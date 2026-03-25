import os
import json
import pickle
from contextlib import closing
from io import BytesIO
from typing import Any
import traceback
import zipfile

import lmdb

from beartype import beartype
import numpy as np
import torch
from PIL import Image

from src.data.utils import decode_video, convert_pil_image_to_byte_tensor, sample_image_vae_latents, sample_video_vae_latents, lean_resize_frames, path_key, copy_file, get_s3_file_paths, VIDEO_EXTENSIONS
from src.utils.os_utils import file_ext
from src.structs import DataSampleType as DSType
import src.utils.distributed as dist

#----------------------------------------------------------------------------

class FolderDataset(torch.utils.data.Dataset):
    _SRC_TYPES = ['dir', 'zip', 's3', 'lmdb']

    def __init__(self,
        src: str,
        max_size: int | None               = None,       # Artificially limit the size of the dataset. None = no limit.
        label_shape: tuple[int] = (),                    # Shape of the label. Empty tuple = no label.
        shuffle_seed: int | None           = 0,          # Random seed for shuffling the data idx. None = no shuffling.
        data_type: DSType | str  = False,                # Dataset contains precomputed latents instead of images.
        metadata_ext = '.json',
        cache_dir: str | None=None,                             # Copy files from src to the specified local cache directory while iterating.
        s3_listdir_cache_dir: str=None,                  # When using S3 paths, cache the list of files to this local file.
        resolution: tuple[int, int]=None,                # Desired resolution. None = keep original resolution.
        replace_broken_samples_with_random: bool=False,  # If True, replaces broken samples with random samples from the dataset.

        # Video-specific params.
        thread_type: str='NONE',                         # Thread type to use for decoding videos.
        framerate: float=None,                           # Desired framerate. None = use original framerate for each video.
        allow_shorter_videos: bool=False,                # Should we allow videos that are shorter than `load_n_consecutive`?
        random_offset: bool=True,                        # Should we use a random offset when extracting frames from videos?
        frame_seek_timeout_sec: float=10.0,              # Timeout in seconds for seeking to a specific frame in a video.

        # Debugging params.
        print_exceptions: bool   = True,
        print_traceback: bool    = True,                 # Should we print the traceback when an exception occurs?
    ):
        self._src = src
        self.name = path_key(src)
        self.data_type = DSType.from_str(data_type) if isinstance(data_type, str) else data_type
        self._shuffle_seed = shuffle_seed
        self._label_shape = label_shape
        self._max_size = max_size
        self._zipfile = None
        self._lmdb_env = None
        self._lmdb_keys: dict[str, bytes] = {}
        self._lmdb_num_samples: int | None = None
        self._print_exceptions = print_exceptions
        self._print_traceback = print_traceback
        self.resolution = resolution
        self._metadata_ext = metadata_ext
        self._cache_dir = cache_dir
        self._s3_listdir_cache_dir = s3_listdir_cache_dir
        self._replace_broken_samples_with_random = replace_broken_samples_with_random

        # Video-specific params.
        self._thread_type = thread_type
        self._framerate = framerate
        self._allow_shorter_videos = allow_shorter_videos
        self._random_offset = random_offset
        self._frame_seek_timeout_sec = frame_seek_timeout_sec

        # Initializing the index.
        self._all_fnames: list[str] | set[str] = set()
        self._data_fnames: list[str] = []
        self._init_all_fnames()
        self._init_data_fnames()
        print(f"[FolderDataset] type={self._type} src={self._src} all={len(self._all_fnames)} data={len(self._data_fnames)}")

        # Apply max_size.
        if max_size is not None and len(self._data_fnames) > max_size:
            assert self._shuffle_seed is not None, 'When using max_size, shuffle_seed must be specified to get a random subset of the data.'
            self._data_fnames = self._data_fnames[:max_size]

        self.epoch_size = len(self._data_fnames)

    def _init_all_fnames(self):
        if self._src.startswith('s3://'):
            assert self._cache_dir is not None, 'When using S3 paths, `cache_dir` must be specified to cache the files locally.'
            assert self._src.endswith('/'), f'When using S3 paths, the path must end with a slash (e.g. s3://bucket-name/dataset/), but got: {self._src}'
            self._type = 's3'
            s3_contents: list[str] = get_s3_file_paths(self._src, is_main_process=dist.get_rank() == 0, s3_listdir_cache_dir=self._s3_listdir_cache_dir)
            s3_contents = [path[len(self._src):] for path in s3_contents if not path.endswith('/')] # Make paths relative to the S3 "directory".
            self._all_fnames = set(s3_contents)
        elif self._is_lmdb_path(self._src):
            self._type = 'lmdb'
            self._all_fnames = self._get_lmdb_keys()
        elif os.path.isdir(self._src):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._src) for root, _dirs, files in os.walk(self._src) for fname in files}
        elif file_ext(self._src) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError(f'Path must point to a directory, LMDB, or zip, but got: {self._src}')

    def _init_data_fnames(self):
        Image.init()
        if self._type == 'lmdb':
            self._data_fnames = [fname for fname in self._all_fnames if not self._is_lmdb_metadata_key(fname)]
        else:
            data_ext = {DSType.VIDEO: VIDEO_EXTENSIONS, DSType.IMAGE: Image.EXTENSION, DSType.VIDEO_LATENT: {'.pkl'}, DSType.IMAGE_LATENT: {'.pkl'}}[self.data_type]
            self._data_fnames = sorted(fname for fname in self._all_fnames if file_ext(fname) in data_ext) # [num_videos]
        if self._shuffle_seed is not None:
            np.random.RandomState(self._shuffle_seed % (1 << 31)).shuffle(self._data_fnames)
        assert len(self._data_fnames) > 0, f'No data files found in the specified path: {self._src}.'

    def _is_lmdb_path(self, path: str) -> bool:
        if path.endswith('.lmdb'):
            return True
        if not os.path.isdir(path):
            return False
        return any(os.path.exists(os.path.join(path, fname)) for fname in ['data.mdb', 'lock.mdb'])

    def _is_lmdb_metadata_key(self, key: str) -> bool:
        metadata_keys = {'num_samples', 'created_at'}
        if key in metadata_keys:
            return True
        if self._metadata_ext is not None and key.endswith(self._metadata_ext):
            return True
        return False

    def _get_lmdb_env(self):
        assert self._type == 'lmdb'
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._src,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=os.path.isdir(self._src),
                max_readers=256,
            )
        return self._lmdb_env

    def _decode_lmdb_key(self, key: bytes) -> str:
        try:
            return key.decode('utf-8')
        except UnicodeDecodeError:
            return key.hex()

    def _get_lmdb_keys(self) -> list[str]:
        env = self._get_lmdb_env()
        keys: list[str] = []
        with env.begin(write=False) as txn:
            num_samples_raw = txn.get(b'num_samples')
            if num_samples_raw is not None:
                self._lmdb_num_samples = int(num_samples_raw.decode())
                for idx in range(self._lmdb_num_samples):
                    key = f'{idx}'.encode()
                    key_str = str(idx)
                    self._lmdb_keys[key_str] = key
                    keys.append(key_str)
                return keys

            cursor = txn.cursor()
            for key, _value in cursor:
                key_str = self._decode_lmdb_key(key)
                self._lmdb_keys[key_str] = bytes(key)
                keys.append(key_str)
        return keys

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._src)
        return self._zipfile

    def _load_file(self, fname: str, mode: str | None=None):
        if self._type == 'dir':
            src_path = os.path.join(self._src, fname)
            if self._cache_dir is not None:
                # Sometimes, we want to move the files from some slow filesystem to a local node disk.
                dst_path = os.path.join(self._cache_dir, self.name, fname)
                copy_file(src_path, dst_path, skip_if_exists=True)
            else:
                dst_path = src_path
            return open(dst_path, mode or 'rb')
        elif self._type == 'zip':
            return self._get_zipfile().open(fname, mode or 'r')
        elif self._type == 's3':
            # Downloads the file from S3 to local cache_dir and opens it.
            dst_path = os.path.join(self._cache_dir, self.name, fname)
            copy_file(os.path.join(self._src, fname), dst_path, skip_if_exists=True)
            return open(dst_path, mode or 'rb')
        elif self._type == 'lmdb':
            env = self._get_lmdb_env()
            key = self._lmdb_keys.get(fname, fname.encode('utf-8'))
            with env.begin(write=False) as txn:
                value = txn.get(key)
            if value is None:
                raise KeyError(f'LMDB key not found: {fname}')
            data = bytes(value)
            buffer = BytesIO(data)
            if 'b' in (mode or 'rb'):
                return buffer
            return closing(BytesIO(data).getvalue().decode('utf-8'))
        else:
            raise RuntimeError(f'Invalid dataset type: {self._type}')

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
            if self._lmdb_env is not None:
                self._lmdb_env.close()
        finally:
            self._zipfile = None
            self._lmdb_env = None

    def __getstate__(self):
        return dict(self.__dict__, _zipfile=None, _lmdb_env=None)

    def __del__(self):
        try:
            self.close()
        except: # pylint: disable=bare-except
            pass

    def __len__(self):
        return len(self._data_fnames)

    @beartype
    def set_progress(self, epoch: int, sample_in_epoch: int | None=None) -> None:
        pass

    @beartype
    def load_metadata(self, idx) -> tuple[dict[str, Any] | None, np.ndarray | int]:
        if len(self._label_shape) == 0 and self._metadata_ext is None:
            return None, np.array(-1)

        metadata = None
        metadata_path = None
        if self._metadata_ext is not None:
            metadata_path = os.path.splitext(self._data_fnames[idx])[0] + self._metadata_ext
            if self._type == 'lmdb':
                if metadata_path in self._all_fnames:
                    metadata_raw = self._load_file(metadata_path, 'rb')
                    metadata = json.loads(metadata_raw.read().decode('utf-8'))
                else:
                    sample = self._read_lmdb_payload(self._data_fnames[idx])
                    metadata = self._extract_metadata_from_lmdb_sample(sample)
            else:
                assert metadata_path in self._all_fnames, f'Metadata file not found: {metadata_path}'
                with self._load_file(metadata_path, 'r') as f:
                    metadata = json.load(f)

        if len(self._label_shape) == 0:
            label = -1
        else:
            assert metadata is not None and 'class' in metadata, f'No "class" field found in the metadata: {metadata_path}'
            label = np.array(metadata['class'])
            if label.dtype == np.int64:
                assert len(self._label_shape) == 1, f'Cannot convert integer label to one-hot with shape {self._label_shape}'
                onehot = np.zeros(self._label_shape, dtype=np.float32)
                onehot[label] = 1
                label = onehot

        return metadata, label

    def _read_lmdb_payload(self, fname: str) -> Any:
        with self._load_file(fname, 'rb') as f:
            return pickle.load(f)

    def _get_sample_key(self, idx: int, sample_metadata: dict[str, Any] | None=None) -> str:
        if self._type == 'lmdb':
            if sample_metadata is not None and 'filename' in sample_metadata:
                return os.path.splitext(os.path.basename(sample_metadata['filename']))[0].replace('/', '_')
            return self._data_fnames[idx]
        return os.path.splitext(os.path.basename(self._data_fnames[idx]))[0].replace('/', '_')

    def _extract_metadata_from_lmdb_sample(self, sample: Any) -> dict[str, Any] | None:
        if not isinstance(sample, dict):
            return None
        metadata = {}
        if 'input_shape' in sample:
            metadata['input_shape'] = sample['input_shape']
        if 'class' in sample:
            metadata['class'] = sample['class']
        if 'label' in sample and 'class' not in metadata:
            metadata['class'] = sample['label']
        if 'framerate' in sample:
            metadata['framerate'] = sample['framerate']
        return metadata or None

    def __getitem__(self, idx: int, num_retries: int = 10) -> dict:
        assert 0 <= idx < len(self), f'Index out of range: {idx}'
        cur_idx = idx
        for _ in range(num_retries):
            try:
                return self.__unsafe_getitem__(cur_idx)
            except Exception as e: # pylint: disable=broad-except
                if self._print_exceptions:
                    print(f"Exception in __getitem__({cur_idx}): {e}")
                if self._print_traceback:
                    traceback.print_exc()
                if self._replace_broken_samples_with_random:
                    cur_idx = np.random.randint(len(self))
                continue
        raise RuntimeError(f'Failed to read item {idx} after {num_retries} retries.')

    def __unsafe_getitem__(self, idx: int) -> dict:
        with self._load_file(self._data_fnames[idx], 'rb') as sample_file:
            sample_metadata, label = self.load_metadata(idx)
            if self._type == 'lmdb' and sample_metadata is None:
                sample_payload = pickle.load(sample_file)
                sample_metadata = self._extract_metadata_from_lmdb_sample(sample_payload)
                sample_file = BytesIO(pickle.dumps(sample_payload))
            frames = decode_frames_from_sample(
                sample=sample_file,
                data_type=self.data_type,
                sample_metadata=sample_metadata,
                random_offset=self._random_offset,
                frame_seek_timeout_sec=self._frame_seek_timeout_sec,
                allow_shorter_videos=self._allow_shorter_videos,
                framerate=self._framerate,
                resolution=self.resolution,
            ) # [t, c, h, w]

        assert isinstance(frames, torch.Tensor), f"frames is not a torch.Tensor: {type(frames)}"

        return dict(
            video=frames,
            label=label,
            __sample_key__=self._get_sample_key(idx, sample_metadata),
        )

    def get_identifier_desc(self) -> str:
        res_str = "x".join([str(r) for r in self.resolution]) if self.resolution is not None else "orig"
        return f'{self.name}-ms{self._max_size}-r{res_str}-{str(self.data_type).lower()}-fr{self._framerate}-asv{self._allow_shorter_videos}-off{int(self._random_offset)}'

#----------------------------------------------------------------------------
# Some data loading utils.

def _normalize_lmdb_latent_sample(sample: Any, data_type: DSType, sample_metadata: dict[str, Any] | None, random_offset: bool=True) -> tuple[Any, dict[str, Any] | None]:
    if not isinstance(sample, dict):
        return sample, sample_metadata

    if sample_metadata is None:
        sample_metadata = {}
    if 'label' in sample and 'class' not in sample_metadata:
        sample_metadata['class'] = sample['label']
    if 'filename' in sample and 'filename' not in sample_metadata:
        sample_metadata['filename'] = sample['filename']
    if 'input_shape' in sample and 'input_shape' not in sample_metadata:
        sample_metadata['input_shape'] = sample['input_shape']

    if 'moments' not in sample:
        return sample, sample_metadata

    moments_key = 'moments'
    if 'moments_flip' in sample:
        use_flip = np.random.rand() < 0.5 if random_offset else False
        moments_key = 'moments_flip' if use_flip else 'moments'

    moments = torch.as_tensor(sample[moments_key]) if isinstance(sample[moments_key], np.ndarray) else sample[moments_key]
    assert moments.ndim >= 3, f"Unsupported LMDB moments shape: {moments.shape}"
    c = moments.shape[-3] // 2
    assert c > 0 and moments.shape[-3] % 2 == 0, f"Expected moments channel dimension to be even, got shape {moments.shape}"

    mean = moments[..., :c, :, :]
    logvar = moments[..., c:, :, :]
    normalized = {'mean': mean, 'logvar': logvar}

    return normalized, sample_metadata


def decode_frames_from_sample(
    sample: BytesIO,
    data_type: DSType,
    sample_metadata: dict[str, Any] | None,
    random_offset: bool = True,
    frame_seek_timeout_sec: float = 10.0,
    allow_shorter_videos: bool = False,
    framerate: float = None,
    resolution: tuple[int, int, int] = None,
) -> torch.Tensor:
    if data_type == DSType.VIDEO:
        frames = decode_video(
            video_file=sample,
            num_frames_to_extract=resolution[0],
            random_offset=random_offset,
            frame_seek_timeout_sec=frame_seek_timeout_sec,
            allow_shorter_videos=allow_shorter_videos,
            framerate=framerate,
            thread_type='NONE',
        )[0] # (t, Image)
        resize_to = tuple(resolution[1:])
    elif data_type == DSType.IMAGE:
        frames = [Image.open(sample)] # (1, Image)
        resize_to = tuple(resolution[1:])
    elif data_type == DSType.IMAGE_LATENT:
        latent_sample, sample_metadata = _normalize_lmdb_latent_sample(pickle.load(sample), data_type, sample_metadata, random_offset=random_offset)
        frames = sample_image_vae_latents(latent_sample).unsqueeze(0) # [1, lc, lh, lw]
        resize_to = None
    elif data_type == DSType.VIDEO_LATENT:
        latent_sample, sample_metadata = _normalize_lmdb_latent_sample(pickle.load(sample), data_type, sample_metadata, random_offset=random_offset)
        assert sample_metadata is not None, "sample_metadata must be provided for VIDEO_LATENT data_type."
        frames = sample_video_vae_latents(
            latent_sample,
            orig_shape=tuple(sample_metadata['input_shape']),
            num_rgb_frames_to_extract=resolution[0],
            fps_orig=sample_metadata.get('framerate', None),
            fps_trg=framerate,
            random_offset=random_offset,
        ) # [lt, lc, lh, lw]
        resize_to = None
    else:
        raise NotImplementedError(f"Unknown data type: {data_type}")

    frames = lean_resize_frames(frames, resize_to) if resize_to is not None else frames # (t, Image)
    frames = torch.stack([convert_pil_image_to_byte_tensor(f, cut_alpha=True) for f in frames]) if isinstance(frames[0], Image.Image) else frames # [t, c, h, w]

    return frames

#----------------------------------------------------------------------------
