import os
from typing import TYPE_CHECKING, Any, Self, override

import torch

from dno_optimized.options import GenerateOptions

from .callback import Callback, CallbackStepAction

if TYPE_CHECKING:
    from ..noise_optimizer import DNOInfoDict


class TensorboardCallback(Callback):
    TB_GLOBAL_VARS = ["lr", "perturb_scale"]
    TB_GROUP_VARS = ["loss", "loss_diff", "loss_decorrelate", "diff_norm", "grad_norm"]
    TB_HIST_VARS = ["x", "z"]

    def __init__(
        self,
        log_dir: str | None = None,
        flush_secs: int | None = None,
        every_n_steps: int | None = None,
        start_after: int | None = None,
        enable_profiler: bool | None = None,
    ):
        super().__init__(every_n_steps, start_after)

        self.log_dir = log_dir or "logs"
        self.flush_secs = flush_secs or 2
        self.enable_profiler = enable_profiler or False

        from torch.utils.tensorboard.writer import SummaryWriter

        # Log in CWD/logs by default
        self._writer = SummaryWriter(log_dir=self.log_dir, flush_secs=self.flush_secs)

        self._profiler = None
        if enable_profiler:
            self._profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

    @property
    def writer(self):
        return self._writer

    @override
    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        return cls(
            log_dir=config.get("log_dir") or os.path.join(options.out_path, "logs"),
            flush_secs=config.get("flush_secs"),
            every_n_steps=config.get("every_n_steps"),
            start_after=config.get("start_after"),
            enable_profiler=config.get("enable_profiler")
        )

    @override
    def on_train_begin(self, num_steps: int, batch_size: int):
        if self._profiler:
            self._profiler.start()

    @override
    def on_train_end(self, num_steps: int, batch_size: int, hist: "list[DNOInfoDict]"):
        if self._profiler:
            self._profiler.stop()

    @override
    def on_step_begin(self, step: int) -> CallbackStepAction | None:
        if self._profiler:
            self._profiler.step()

    @override
    def on_step_end(self, step, info, hist):
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
            self._writer.add_scalar(f"{group_index:02d}_dno/{var}", scalar_value, global_step=step)
        group_index += 1

        # Batched value (one per trial)
        for var in self.TB_GROUP_VARS:
            value = info[var]
            for trial, trial_value in enumerate(value):
                trial_value = convert_value(trial_value)
                self._writer.add_scalar(f"{group_index:02d}_{var}/trial_{trial}", trial_value, global_step=step)
            group_index += 1

        # Log noise and output histograms
        for var in self.TB_HIST_VARS:
            for trial, trial_value in enumerate(info[var]):
                self._writer.add_histogram(
                    f"{group_index:02d}_hist_{var}/trial_{trial}", trial_value, global_step=step
                )
            group_index += 1
