import os
import pickle

import numpy as np
import scipy.linalg
import torch
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from loguru import logger

from src.structs import EasyDict, TensorGroup
from src.metrics import metric_main
from src.utils import misc, logging_utils, distributed as dist
from src.utils.os_utils import disable_trivial_warnings, save_json
from src.data import Data, construct_full_cond_from_batch
from src.inference import sample
from src.training.network_utils import load_snapshot
from src.utils.autoencoder_utils import compute_autoencoder_stats, init_autoencoder_stats
from infra.utils import dict_to_hydra_overrides, recursive_instantiate

#----------------------------------------------------------------------------

def _split_eval_seeds(num_fid_samples: int, batch_size: int) -> list[torch.Tensor]:
    all_seeds = torch.arange(num_fid_samples)
    num_batches = ((len(all_seeds) - 1) // (batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = all_seeds.tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    assert all(len(batch) > 0 for batch in rank_batches), f'No seeds assigned to rank {dist.get_rank()}: {rank_batches}'
    return rank_batches


def _run_temf_eval(cfg: EasyDict, device: torch.device):
    """Export decoded PNGs and score them with torch_fidelity."""
    assert cfg.output_dir is not None, 'cfg.output_dir must be specified for eval_backend=temf'
    assert cfg.fid_statistics_file is not None, 'cfg.fid_statistics_file must be specified for eval_backend=temf'
    assert cfg.skip_generation or cfg.gen_dataset is None, 'eval_backend=temf only supports checkpoint evaluation unless skip_generation=true'

    try:
        import torch_fidelity
    except ImportError as exc:
        raise ImportError('torch_fidelity is required for eval_backend=temf') from exc

    # Use output_dir directly instead of output_dir/images
    image_dir = cfg.output_dir
    image_dir_abs = os.path.abspath(str(image_dir))
    fid_stats_path = os.path.abspath(str(cfg.fid_statistics_file))
    dist.loginfo0(f'TEMF eval backend: skip_generation={cfg.skip_generation}, image_dir={image_dir_abs}')

    if cfg.skip_generation:
        dist.loginfo0('TEMF eval backend: skipping generation and scoring existing PNGs only')
    else:
        net, _, experiment_cfg = load_snapshot(cfg.ckpt, verbose=dist.get_rank() == 0, device=device)
        if cfg.get('overwrite_path_remote') is not None:
            logger.info('overwriting path_remote', cfg.overwrite_path_remote)
            if len(experiment_cfg.dataset.video_streams) > 0: experiment_cfg.dataset.video_streams[0].path_remote = cfg.overwrite_path_remote
            if len(experiment_cfg.dataset.image_streams) > 0: experiment_cfg.dataset.image_streams[0].path_remote = cfg.overwrite_path_remote
        net = net.to(device).eval()
        recursive_instantiate(experiment_cfg)
        experiment_cfg = EasyDict.init_recursively(experiment_cfg)

        if 'sampling' not in cfg or cfg.sampling is None:
            cfg.sampling = experiment_cfg.sampling

        if 'dataset' in cfg and cfg.dataset is not None:
            if cfg.dataset.get('resolution') is None:
                cfg.dataset.resolution = experiment_cfg.dataset.resolution
            cfg.dataset.predownload = experiment_cfg.dataset.predownload
            experiment_cfg.dataset = EasyDict.init_recursively(cfg.dataset).to_dict()

        recursive_instantiate(experiment_cfg)
        experiment_cfg = EasyDict.init_recursively(experiment_cfg)
        experiment_cfg.dataset.batch_size = cfg.batch_size * dist.get_world_size()
        experiment_cfg.dataset.batch_gpu = cfg.batch_size
        data = Data.init_from_cfg(experiment_cfg.dataset)
        iterator = data.iterator
        rank_seeds = _split_eval_seeds(cfg.num_fid_samples, cfg.batch_size)

        common_kwargs = dict(verbose=False, device=device, save_traj_len=1)
        for cur_seeds in rank_seeds:
            batch_real = {k: v[:len(cur_seeds)] for k, v in next(iterator).items()}
            cur_cond = construct_full_cond_from_batch(batch_real, misc.unwrap_module(net).cfg.use_x_cond)
            samples = sample(net, cfg.sampling, [cur_seeds], [cur_cond], **common_kwargs, **cfg.model_kwargs)
            samples = TensorGroup(video=samples) if not isinstance(samples, TensorGroup) else samples
            assert samples.video.ndim == 5, f'TEMF eval expects single-frame image outputs, got shape {samples.video.shape}'
            logging_utils.save_samples(
                samples=samples,
                stats=None,
                output_dir=image_dir,
                save_name='sample',
                save_to_wandb=False,
                seeds=cur_seeds.tolist(),
                use_res_suffix=False,
                captions=[None] * len(cur_seeds),
                **dict(cfg.get('save_opts', {})),
            )

    dist.barrier()

    if dist.is_main_process():
        assert os.path.isdir(image_dir_abs), f'Expected generated image directory to exist: {image_dir_abs}'
        image_files = sorted(f for f in os.listdir(image_dir_abs) if f.lower().endswith('.png'))
        assert len(image_files) > 0, f'No PNG samples were written to {image_dir_abs}'
        logger.info({'temf_eval_image_dir': image_dir_abs, 'num_pngs': len(image_files), 'fid_statistics_file': fid_stats_path})

        sample_metrics = {}
        if cfg.compute_isc:
            sample_metrics = torch_fidelity.calculate_metrics(
                input1=image_dir_abs,
                cuda=True,
                fid=False,
                isc=True,
                kid=False,
                prc=False,
                verbose=cfg.verbose,
            )

        try:
            from torch_fidelity.feature_extractor import create_feature_extractor
        except ImportError:
            from torch_fidelity.utils import create_feature_extractor

        feat_extractor = create_feature_extractor(
            'inception-v3-compat',
            ['2048'],
            cuda=True,
            verbose=cfg.verbose,
        )
        featuresdict = torch_fidelity.utils.extract_featuresdict_from_input_id_cached(
            1,
            feat_extractor=feat_extractor,
            input1=image_dir_abs,
            cuda=True,
            verbose=cfg.verbose,
            save_cpu_ram=False,
        )
        feats = np.asarray(featuresdict['2048'])
        mean_gen = feats.mean(axis=0)
        cov_gen = np.cov(feats, rowvar=False)

        stats_npz = np.load(fid_stats_path)
        mean_real = stats_npz['mu'] if 'mu' in stats_npz else stats_npz['mean']
        cov_real = stats_npz['sigma'] if 'sigma' in stats_npz else stats_npz['cov']
        cov_gen = (cov_gen + cov_gen.T) / 2
        cov_real = (cov_real + cov_real.T) / 2
        mean_diff = mean_gen - mean_real
        cov_sqrt, _ = scipy.linalg.sqrtm(cov_gen @ cov_real, disp=False)
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        fid = float(mean_diff @ mean_diff + np.trace(cov_gen + cov_real - 2 * cov_sqrt))

        results = {'fid': float(fid), 'num_pngs': len(image_files)}
        if cfg.compute_isc:
            results['isc_mean'] = float(sample_metrics['inception_score_mean'])
            results['isc_std'] = float(sample_metrics['inception_score_std'])
        if cfg.save_path is not None:
            os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
            save_json(results, cfg.save_path)
        logger.info(results)

    dist.barrier()


@hydra.main(config_path="../configs", config_name="evaluate.yaml", version_base='1.2')
def evaluate(cfg: DictConfig):
    disable_trivial_warnings()
    cfg = EasyDict.init_recursively(cfg)

    # Init torch.distributed and torch settings.
    dist.init()
    dist.init_random_state_and_cuda(seed=cfg.global_seed, cudnn_benchmark=cfg.cudnn_benchmark, allow_tf32=cfg.allow_tf32)
    device = torch.device('cuda')

    if cfg.eval_backend in ('alphaflow', 'temf'):
        _run_temf_eval(cfg, device)
        if cfg.verbose:
            dist.loginfo0('Exiting...')
        return

    # Loading the info about the generated samples.
    if cfg.gen_dataset is None:
        # Loading the network, and we'll use it to run the evals.
        net, snapshot_path, experiment_cfg = load_snapshot(cfg.ckpt, verbose=dist.get_rank() == 0, device=device)
        if cfg.get('overwrite_path_remote') is not None:
            logger.info('overwriting path_remote', cfg.overwrite_path_remote)
            if len(experiment_cfg.dataset.video_streams) > 0: experiment_cfg.dataset.video_streams[0].path_remote = cfg.overwrite_path_remote
            if len(experiment_cfg.dataset.image_streams) > 0: experiment_cfg.dataset.image_streams[0].path_remote = cfg.overwrite_path_remote
        net = net.to(device).eval()
        gen_dataset = None
        recursive_instantiate(experiment_cfg)
    else:
        # We already produced the samples with the model, and we'll use them to run the evals.
        net = snapshot_path = None
        gen_dataset = Data.init_from_cfg(cfg.gen_dataset).dataset
        experiment_cfg = None

    # Loading the info about the training dataset.
    if 'dataset' not in cfg or cfg.dataset is None:
        if experiment_cfg is None:
            # Initialize the GT dataset kwargs and model config from the provided experiment config.
            assert os.path.isfile(cfg.experiment_cfg.path), f'Invalid cfg.experiment_cfg.path: {cfg.experiment_cfg.path} (we need it for GT dataset kwargs)'
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            with initialize(config_path=os.path.join('../', os.path.dirname(cfg.experiment_cfg.path))):
                overrides = None if len(cfg.experiment_cfg.overrides) == 0 else dict_to_hydra_overrides(cfg.experiment_cfg.overrides)
                experiment_cfg = compose(os.path.basename(cfg.experiment_cfg.path), overrides=overrides if overrides is not None else [])
            recursive_instantiate(experiment_cfg)
        gt_dataset_cfg = EasyDict.init_recursively(experiment_cfg.dataset)
    else:
        gt_dataset_cfg = EasyDict.init_recursively(cfg.dataset)

    # TODO: this patching is too hacky, we should do something better.
    experiment_cfg = EasyDict.init_recursively({'dataset': {}} if experiment_cfg is None else experiment_cfg)

    data: Data = Data.init_from_cfg(gt_dataset_cfg)

    if any(m.startswith('lat_noised_reconstruction') for m in cfg.metrics):
        latents_stats_path = os.path.join(cfg.env.latents_stats_dir, f'{experiment_cfg.experiment_id_str}-{os.path.basename(snapshot_path)}-n128.pkl')
        stats, was_loaded = init_autoencoder_stats(experiment_cfg, latents_stats_path=latents_stats_path, keys=['latents'])
        experiment_cfg.dataset.predownload = 0
        if was_loaded:
            latents_std = stats.latents.get_basic_stats().std # [t * c * h * w]
        else:
            latents_std = compute_autoencoder_stats(net, data, experiment_cfg, device, num_samples=128, stats=stats).latents.std # [t * c * h * w]
            # Save the stats.
            with open(latents_stats_path, 'wb') as f:
                pickle.dump(stats, f)
        latents_std = torch.from_numpy(latents_std).float().to(device).view(-1, net.model.latent_channels, *net.model.latent_resolution[1:]) # [t, c, h, w]
    else:
        latents_std = None

    # Validate arguments.
    assert len(cfg.metrics) > 0, f'cfg.metrics must contain at least one value, but got {cfg.metrics}'
    assert all(metric_main.is_valid_metric(m) for m in cfg.metrics), '\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics())
    assert dist.get_world_size() >= 1, f'--gpus must be at least 1, but have {dist.get_world_size()}'
    if gen_dataset is not None:
        if "streams" in cfg.gen_dataset and len(cfg.gen_dataset.streams) > 1:
            gen_path = cfg.gen_dataset.streams[0].src
        else:
            gen_path = cfg.gen_dataset.src
    else:
            gen_path = snapshot_path

    # Calculate each metric.
    results = {}
    for metric in cfg.metrics:
        if cfg.verbose:
            dist.loginfo0(f'Calculating {metric} for {gen_path}...')
        conditioning_dataset = data.dataset_val if data.dataset_val is not None else data.dataset
        metric_dataset = data.dataset_eval
        if metric_main.uses_fixed_fid_stats(metric, cfg.get('fid_statistics_file', None)):
            metric_dataset = conditioning_dataset
        result_dict = metric_main.compute_metric(
            metric,
            net=net,
            dataset=metric_dataset,
            conditioning_dataset=conditioning_dataset,
            model_kwargs=cfg.model_kwargs,
            rank=dist.get_rank(),
            device=torch.device('cuda', dist.get_rank()),
            verbose=cfg.verbose,
            gen_dataset=gen_dataset,
            batch_gen=cfg.batch_size,
            sampling_cfg=cfg.get('sampling'),
            dataset_stats_dir=cfg.env.dataset_stats_dir,
            detector_batch_gpu=cfg.detector_batch_gpu,
            save_partial_stats_freq=cfg.save_partial_stats_freq,
            fid_statistics_file=cfg.get('fid_statistics_file', None),
        )
        results[metric] = result_dict
        if dist.is_main_process():
            metric_main.report_metric(result_dict, run_dir=None, snapshot_path=snapshot_path, save_result=False)
        if cfg.verbose:
            dist.print0()

    if cfg.save_path is not None and dist.is_main_process():
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
        save_json(results, cfg.save_path)

    # Done.
    if cfg.verbose:
        dist.loginfo0('Exiting...')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
