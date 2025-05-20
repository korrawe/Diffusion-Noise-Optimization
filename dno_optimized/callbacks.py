import os
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Literal, Self, override

import torch

from dno_optimized.options import GenerateOptions

if TYPE_CHECKING:
    from noise_optimizer import DNO, DNOInfoDict


@dataclass
class CallbackStepAction:
    stop: bool = field(default=False, metadata={"help": "Set to True to stop optimization after the current step"})
    # Add more options here if needed


class Callback(ABC):
    """Base class for all callbacks. Override one or more `on_[...]` methods to implement functionality."""

    def __init__(self, every_n_steps: int = 1):
        super().__init__()
        self.dno: "DNO"
        self.every_n_steps = every_n_steps

    def invoke(
        self, callback_stage: Literal["train_begin", "train_end", "step_begin", "step_end"], *args, **kwargs
    ) -> CallbackStepAction | None:
        match callback_stage:
            case "train_begin":
                return self.on_train_begin(*args, **kwargs)
            case "train_end":
                return self.on_train_end(*args, **kwargs)
            case "step_begin":
                if (self.dno.step_count) % self.every_n_steps == 0:
                    return self.on_step_begin(*args, **kwargs)
            case "step_end":
                if (self.dno.step_count) % self.every_n_steps == 0:
                    return self.on_step_end(*args, **kwargs)
        return None

    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        """Initializes the callback from a config dict. Override to implement."""
        raise NotImplementedError(cls.__name__)

    def on_train_begin(self, num_steps: int, batch_size: int):
        """Runs once when the training (optimization loop) starts.

        :param num_steps: Number of optimization steps (effective)
        :param batch_size: Batch size (number of trials)
        """
        pass

    def on_train_end(self, num_steps: int, batch_size: int, hist: "list[DNOInfoDict]"):
        """Runs once when the training is done (completed all iterations or other callback resulted in a stop condition).

        :param num_steps: Number of optimization steps (effective)
        :param batch_size: Batch size (number of trials)
        :param hist List (over batch) of dicts containing lists (over steps) with metrics
        """
        pass

    def on_step_begin(self, step: int, global_step: int) -> CallbackStepAction:
        """Runs once before every optimization step (batch)

        :param step: Step number/training round/batch index
        :param global_step: Effective global optimization step, same as step * batch_size
        """
        return CallbackStepAction()

    def on_step_end(
        self, step: int, global_step: int, info: "DNOInfoDict", hist: "list[DNOInfoDict]"
    ) -> CallbackStepAction:
        """Runs once after every optimization step (batch)

        :param step: Step number/training round/batch index
        :param global_step: Effective global optimization step, same as step * batch_size
        :param info: Current step's info dict
        :param hist: List (over steps) of dicts containing lists (over batch) with metrics
        """
        return CallbackStepAction()


class CallbackList(list[Callback]):
    """Helper list class for invoking a list of callbacks."""

    def __init__(self, callbacks: Iterable[Callback] | None = None):
        super().__init__(callbacks or [])

    def invoke(
        self, dno: "DNO", callback_stage: Literal["train_begin", "train_end", "step_begin", "step_end"], *args, **kwargs
    ):
        for callback in self:
            callback.dno = dno
            callback.invoke(callback_stage, *args, **kwargs)


class TensorboardCallback(Callback):
    TB_GLOBAL_VARS = ["lr", "perturb_scale"]
    TB_GROUP_VARS = ["loss", "loss_diff", "loss_decorrelate", "diff_norm", "grad_norm"]
    TB_HIST_VARS = ["x", "z"]

    def __init__(self, log_dir: str | None = None, flush_secs: int | None = None):
        super().__init__()

        from torch.utils.tensorboard.writer import SummaryWriter

        # Log in CWD/logs by default
        self.writer = SummaryWriter(log_dir=log_dir or "logs", flush_secs=flush_secs or 10)

    @override
    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        return cls(
            log_dir=config.get("log_dir") or os.path.join(options.out_path, "logs"),
            flush_secs=config.get("flush_secs"),
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


def create_callback(name: str, options: GenerateOptions, callback_args: dict) -> Callback:
    match name:
        case "tensorboard":
            return TensorboardCallback.from_config(options, callback_args)
        case _:
            raise KeyError(f"`{name}` is not a valid callback name")
