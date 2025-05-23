import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Self, TypeVar, override

import torch

from dno_optimized.callbacks.callback import CallbackStepAction
from dno_optimized.noise_optimizer import DNOInfoDict, DNOStateDict
from dno_optimized.options import GenerateOptions

from .callback import Callback

T = TypeVar("T")
S = TypeVar("S", bound=float | int)


def argmin(it: Iterable[T], key: Callable[[T], S]) -> tuple[int, S]:
    min_val = None
    min_idx = None
    for i, x in enumerate(it):
        x_val = key(x)
        if min_val is None or x_val <= min_val:
            min_idx = i
            min_val = x_val
    assert min_idx is not None and min_val is not None
    return min_idx, min_val


def argmax(it: Iterable[T], key: Callable[[T], S]) -> tuple[int, S]:
    max_val = None
    max_idx = None
    for i, x in enumerate(it):
        x_val = key(x)
        if max_val is None or x_val >= max_val:
            max_idx = i
            max_val = x_val
    assert max_idx is not None and max_val is not None
    return max_idx, max_val


@dataclass
class ModelCheckpoint:
    step: int
    metric_name: str
    metric_value: float
    state_dict: dict
    type: Literal["best", "latest"]

    def get_save_path(self, save_dir: Path):
        suffix = "" if self.type == "best" else "_latest"
        return save_dir / f"model_{self.step:04d}_{self.metric_name}={self.metric_value:.8f}{suffix}.ckpt"


class SaveTopKCallback(Callback):
    def __init__(
        self,
        save_dir: str,
        k: int = 3,
        metric: str = "loss",
        mode: Literal["min", "max"] = "min",
        save_latest: bool = True,
        flush_every: int = 10,
        every_n_steps: int | None = None,
        start_after: int | None = None,
    ):
        super().__init__(every_n_steps, start_after)

        self.save_dir = Path(save_dir)
        self.k = k
        self.metric_key = metric
        self.mode = mode
        self.save_latest = save_latest
        self.flush_every = flush_every

        self._inf = float("inf") if self.mode == "min" else float("-inf")

        # State
        self._latest_model: ModelCheckpoint | None = None
        self._best_models: list[None | ModelCheckpoint] = [None for i in range(self.k)]
        self._save_queue: list[ModelCheckpoint] = []
        self._delete_queue: list[ModelCheckpoint] = []

    @override
    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        return cls(
            save_dir=config.get("save_dir") or os.path.join(options.out_path, "checkpoints"),
            k=config.get("k", 3),
            metric=config.get("metric", "loss"),
            mode=config.get("mode", "min"),
            save_latest=config.get("save_latest", True),
            flush_every=config.get("flush_every", 10),
            every_n_steps=config.get("every_n_steps"),
            start_after=config.get("start_after"),
        )

    @property
    def best_models(self):
        models = [x for x in self._best_models if x is not None]
        return sorted(models, key=lambda model: model.metric_value, reverse=self.mode == "max")

    def create_checkpoint(self, step: int, metric_value: float, type: Literal["best", "latest"]):
        state = self.dno.state_dict()
        state["stop_optimize"] = step  # Needs to be updated
        return ModelCheckpoint(step, self.metric_key, metric_value, state, type)

    def enqueue_save(self, model: ModelCheckpoint):
        try:
            self._delete_queue.remove(model)
        except ValueError:
            pass
        self._save_queue.append(model)

    def enqueue_delete(self, model: ModelCheckpoint):
        try:
            self._save_queue.remove(model)
        except ValueError:
            pass
        self._delete_queue.append(model)

    def update_latest_model(self, step: int, metric_value: float):
        """Update the latest model, deleting the previous latest"""
        # Create new latest model, add to save queue, update reference
        model_ckpt = self.create_checkpoint(step, metric_value, type="latest")
        self.enqueue_save(model_ckpt)
        # Delete old latest model
        if self._latest_model is not None:
            self.enqueue_delete(self._latest_model)
        self._latest_model = model_ckpt

    def update_top_model(self, index: int, step: int, metric_value: float):
        """Updates the top model at index, deleting old model"""
        # Create checkpoint (in-memory) and add to pending list
        model_ckpt = self.create_checkpoint(step, metric_value, type="best")
        self.enqueue_save(model_ckpt)
        # If index was passed, store in state list and delete replaced model
        if prev_model := self._best_models[index]:
            self.enqueue_delete(prev_model)
        self._best_models[index] = model_ckpt

    def flush(self):
        """Flushes all pending changes to disk (creating/deleting checkpoints)"""
        # Delete all pending models
        for model in self._delete_queue:
            save_path = model.get_save_path(self.save_dir)
            save_path.unlink(missing_ok=True)
        self._delete_queue = []

        # Save new best models
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for model in self._save_queue:
            save_path = model.get_save_path(self.save_dir)
            torch.save(model, save_path)

    @override
    def on_step_end(
        self, step: int, info: DNOInfoDict, hist: list[DNOInfoDict]
    ) -> CallbackStepAction | None:
        value = info[self.metric_key].mean().item()  # Mean over batch (hard-coded)

        if self.save_latest:
            self.update_latest_model(step, value)

        # First check if there are "empty" spots to fill
        for i, model in enumerate(self._best_models):
            if model is None:
                self.update_top_model(i, step, value)
                return

        # No more "empty" spots, find worst model and replace if this model is better
        cmp = argmax if self.mode == "min" else argmin
        best_models = [x for x in self._best_models if x is not None]
        worst_model_idx, worst_metric = cmp(best_models, key=lambda model: model.metric_value if model else self._inf)
        if self.mode == "min" and value < worst_metric or self.mode == "max" and value > worst_metric:
            # Replace as this is better. This overwrites the previous save file
            self.update_top_model(worst_model_idx, step, value)

        # Check if we need to flush
        if (step % self.flush_every) == 0:
            self.flush()

    @override
    def on_train_end(self, num_steps: int, batch_size: int, hist: list[DNOInfoDict], state_dict: DNOStateDict):
        self.flush()
