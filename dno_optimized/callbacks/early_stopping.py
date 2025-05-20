from typing import Literal, Self, override

from dno_optimized.callbacks.callback import CallbackStepAction
from dno_optimized.noise_optimizer import DNOInfoDict
from dno_optimized.options import GenerateOptions

from .callback import Callback


class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        patience: int = 10,
        min_improvement: float = 0.0,
        mode: Literal["min", "max"] = "min",
        metric: str = "loss",
        abs_value: float | None = None,
        every_n_steps: int | None = None,
        start_after: int | None = None,
    ):
        super().__init__(every_n_steps, start_after)

        self.patience = patience
        self.min_improvement = min_improvement
        self.mode = mode
        self.metric = metric
        self.abs_value = abs_value

        # State
        self._best_value: float = float("inf") if mode == "min" else float("-inf")
        self._steps_since_improvement: int = 0

    @override
    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        return cls(
            patience=config.get("patience", 10),
            min_improvement=config.get("min_improvement", 0.0),
            mode=config.get("mode", "min"),
            metric=config.get("metric", "loss"),
            abs_value=config.get("abs_value"),
            every_n_steps=config.get("every_n_steps"),
            start_after=config.get("start_after"),
        )

    @override
    def on_step_end(
        self, step: int, info: DNOInfoDict, hist: list[DNOInfoDict]
    ) -> CallbackStepAction | None:
        value = info[self.metric].mean().item()  # Mean over batch

        if self.abs_value is not None:
            if self.mode == "min" and value < self.abs_value or self.mode == "max" and value > self.abs_value:
                # Surpassed min/max absolute value, stop
                self.progress.write("Early stopping (abs_value surpassed)")
                return CallbackStepAction(stop=True)

        if self.mode == "min" and value < self._best_value - self.min_improvement:
            # Mode=min and got lower value with margin min_improvement, reset state
            self._best_value = value
            self._steps_since_improvement = 0
        elif self.mode == "max" and value > self._best_value + self.min_improvement:
            # Mode=max and got higher value with margin min_improvement, reset state
            self._best_value = value
            self._steps_since_improvement = 0
        else:
            # No improvement, increase counter
            self._steps_since_improvement += 1

        if self._steps_since_improvement > self.patience:
            self.progress.write(f"Early stopping (no improvement over last {self.patience} steps)")
            return CallbackStepAction(stop=True)

        return None
