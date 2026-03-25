import torch
from torch.optim.lr_scheduler import LRScheduler

from src.structs import EasyDict

#----------------------------------------------------------------------------

def create_composite_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: EasyDict) -> LRScheduler:
    max_steps = int(cfg.max_steps)
    num_warmup_steps = int(getattr(cfg, 'num_warmup_steps', 0) or 0)

    if max_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    num_warmup_steps = max(min(num_warmup_steps, max_steps), 0)

    if num_warmup_steps == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=cfg.final_lr)

    if num_warmup_steps == max_steps:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / num_warmup_steps,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )

    num_cosine_updates = max_steps - num_warmup_steps
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / num_warmup_steps,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_cosine_updates, eta_min=cfg.final_lr)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])

    return lr_scheduler

#----------------------------------------------------------------------------