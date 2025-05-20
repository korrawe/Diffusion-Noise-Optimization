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
        every_n_steps: int = 1,
    ):
        super().__init__(every_n_steps=every_n_steps)

        self.patience = patience
        self.min_improvement = min_improvement
        self.mode = mode
        self.metric = metric
        self.abs_value = abs_value

        # State
        self.best_value: float = float("inf")
        self.steps_since_improvement: int = 0

    @override
    @classmethod
    def from_config(cls, options: GenerateOptions, config: dict) -> Self:
        return cls(
            patience=config.get("patience", 10),
            min_improvement=config.get("min_improvement", 0.0),
            mode=config.get("mode", "min"),
            metric=config.get("metric", "loss"),
            abs_value=config.get("abs_value", None),
            every_n_steps=config.get("every_n_steps", 1),
        )

    @override
    def on_step_end(
        self, step: int, global_step: int, info: DNOInfoDict, hist: list[DNOInfoDict]
    ) -> CallbackStepAction | None:
        value = info[self.metric].mean().item()  # Mean over batch

        if self.abs_value is not None:
            if self.mode == "min" and value < self.abs_value or self.mode == "max" and value > self.abs_value:
                # Surpassed min/max absolute value, stop
                return CallbackStepAction(stop=True)

        if self.mode == "min" and value < self.best_value - self.min_improvement:
            # Mode=min and got lower value with margin min_improvement, reset state
            self.best_value = value
            self.steps_since_improvement = 0
        elif self.mode == "max" and value > self.best_value + self.min_improvement:
            # Mode=max and got higher value with margin min_improvement, reset state
            self.best_value = value
            self.steps_since_improvement = 0
        else:
            # No improvement, increase counter
            self.steps_since_improvement += 1

        if self.steps_since_improvement > self.patience:
            return CallbackStepAction(stop=True)

        return None
