import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self, override

import pandas as pd
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from dno_optimized.callbacks.callback import CallbackList, CallbackStepAction
from dno_optimized.callbacks.tensorboard import TensorboardCallback
from dno_optimized.noise_optimizer import DNOInfoDict, DNOStateDict
from dno_optimized.options import GenerateOptions

from .callback import Callback


class ProfilerCallback(Callback):
    def __init__(self, report_dir: str = "profiler", every_n_steps: int | None = None, start_after: int | None = None):
        super().__init__(every_n_steps, start_after)

        self.report_dir = Path(report_dir)

        self._tb_writer: SummaryWriter | None = None

        # State
        self._train_start_time: datetime | None = None
        self._last_step_time: datetime | None = None

        # Reporting
        # Per-step statistics
        self._step_values: dict[int, dict] = defaultdict(dict)
        self._total_train_time: timedelta | None = None
        self._stop_optimize: int | None = None

    def __post_init__(self, callbacks: CallbackList):
        # Try to get tensorboard callback, otherwise will log to progress bar
        tb_callback = callbacks.get(TensorboardCallback)
        if tb_callback:
            self._tb_writer = tb_callback.writer

    @override
    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        return cls(
            report_dir=config.get("report_dir") or str(options.out_path / "profiler"),
            every_n_steps=config.get("every_n_steps"),
            start_after=config.get("start_after"),
        )

    def _log(self, global_step: int, key: str, value: float | int, tb_tag: str | None = None):
        tb_tag = tb_tag or key
        self._step_values[global_step][key] = value
        if self._tb_writer:
            self._tb_writer.add_scalar(f"profiler/{tb_tag}", value, global_step=global_step)
        else:
            self.progress.set_postfix({**self.progress.postfix, **{f"mem_{tb_tag}": f"{value} MB"}})

    def _log_gpu_mem(self, step: int):
        # Log GPU memory consumption
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            gpu_tot = torch.cuda.memory_reserved() / 1024**2
            self._log(step, "mem_gpu_allocated", gpu_mem, tb_tag="mem/gpu_allocated")
            self._log(step, "mem_gpu_reserved", gpu_tot, tb_tag="mem/gpu_reserved")
        else:
            warnings.warn(
                f"{self.__class__.__name__}: CUDA not available, only reporting CPU memory usage.", stacklevel=2
            )

    def _log_cpu_mem(self, step: int):
        # Log CPU memory
        try:
            import psutil

            process = psutil.Process(os.getpid())
            cpu_mem = process.memory_info().rss / 1024**2
            self._log(step, "mem_cpu", cpu_mem, tb_tag="mem/cpu")
        except ImportError:
            warnings.warn(
                f"{self.__class__.__name__}: psutil package not found. Unable to determine CPU RAM usage.", stacklevel=2
            )

    def _log_step_time(self, step: int):
        now = datetime.now()

        if self._last_step_time is None:
            self._last_step_time = now
            return

        assert self._train_start_time is not None
        elapsed = (now - self._train_start_time).total_seconds()
        step_time = (now - self._last_step_time).total_seconds()
        self._log(step, "elapsed", elapsed, tb_tag="time/elapsed")
        self._log(step, "step_time", step_time, tb_tag="time/step_time")
        self._last_step_time = now
        self._step_values[step]["step_time"] = step_time
        self._step_values[step]["elapsed"] = elapsed

    def _generate_report(self):
        # Generate time-series data
        df = pd.DataFrame.from_dict(self._step_values, orient="index")
        df.index.name = "step"

        # Generate summary data
        stats = pd.Series(
            {
                "total_time": self._total_train_time,
                "optimize_steps": self._stop_optimize,
                "peak_mem_cpu": df["mem_cpu"].max(),
                "peak_mem_gpu_allocated": df["mem_gpu_allocated"].max(),
                "peak_mem_gpu_reserved": df["mem_gpu_reserved"].max(),
            }
        )
        # stats.index.name = "key"
        # stats.name = "value"

        self.report_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.report_dir / "step_stats.csv")
        stats.to_csv(self.report_dir / "summary.csv", header=False)

    @override
    def on_step_end(self, step: int, info: DNOInfoDict, hist: list[DNOInfoDict]) -> CallbackStepAction | None:
        self._log_gpu_mem(step)
        self._log_cpu_mem(step)
        self._log_step_time(step)

    @override
    def on_train_begin(self, num_steps: int, batch_size: int):
        self._train_start_time = datetime.now()

    @override
    def on_train_end(self, num_steps: int, batch_size: int, hist: list[DNOInfoDict], state_dict: DNOStateDict):
        assert self._train_start_time is not None
        self._total_train_time = datetime.now() - self._train_start_time
        self._stop_optimize = state_dict["stop_optimize"]
        self._generate_report()
