import os
from typing import Any, Self, override

import torch

from dno_optimized.options import GenerateOptions

from .callback import Callback


class TensorboardCallback(Callback):
    TB_GLOBAL_VARS = ["lr", "perturb_scale"]
    TB_GROUP_VARS = ["loss", "loss_diff", "loss_decorrelate", "diff_norm", "grad_norm"]
    TB_HIST_VARS = ["x", "z"]

    def __init__(self, log_dir: str | None = None, flush_secs: int | None = None, every_n_steps: int = 1):
        super().__init__(every_n_steps=every_n_steps)

        from torch.utils.tensorboard.writer import SummaryWriter

        # Log in CWD/logs by default
        self.writer = SummaryWriter(log_dir=log_dir or "logs", flush_secs=flush_secs or 10)

    @override
    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        return cls(
            log_dir=config.get("log_dir") or os.path.join(options.out_path, "logs"),
            flush_secs=config.get("flush_secs"),
            every_n_steps=config.get("every_n_steps", 1),
        )

    @override
    def on_step_end(self, step, global_step, info, hist):
        # Function to convert possibly tensor value to scalar
        def convert_value(x: Any):
            if isinstance(x, list):
                return sum(x) / len(x)
            if isinstance(x, torch.Tensor):
                return x.mean().detach().cpu().item()
            return x

        # Used for ordering variables in TB
        group_index = 0
        # Log all variables
        # Scalar value, or value broadcast to batch (e.g. learning rate)
        for var in self.TB_GLOBAL_VARS:
            scalar_value = convert_value(info[var])
            self.writer.add_scalar(f"{group_index:02d}_dno/{var}", scalar_value, global_step=global_step)
        group_index += 1

        # Batched value (one per trial)
        for var in self.TB_GROUP_VARS:
            value = info[var]
            for trial, trial_value in enumerate(value):
                trial_value = convert_value(trial_value)
                self.writer.add_scalar(f"{group_index:02d}_{var}/trial_{trial}", trial_value, global_step=global_step)
            group_index += 1

        # Log noise and output histograms
        for var in self.TB_HIST_VARS:
            for trial, trial_value in enumerate(info[var]):
                self.writer.add_histogram(
                    f"{group_index:02d}_hist_{var}/trial_{trial}", trial_value, global_step=global_step
                )
            group_index += 1
