from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, Self

from dno_optimized.options import GenerateOptions

if TYPE_CHECKING:
    from noise_optimizer import DNO, DNOInfoDict


@dataclass
class CallbackStepAction:
    stop: bool = field(default=False, metadata={"help": "Set to True to stop optimization after the current step"})
    # Add more options here if needed

    @staticmethod
    def aggregate(actions: "list[CallbackStepAction | None]"):
        """Method to aggregate a list of callback step actions from multiple callbacks. This defines how to "resolve
        conflicts" (e.g. for stopping criteria). "None" actions are effectively ignored. When there are no actions, the
        method should return the "default action" that effectively does nothing.

        :param actions: Iterable of actions
        :return: Aggregated callback action, or default action if no valid actions in list
        """
        return CallbackStepAction(stop=any(a.stop for a in actions if a is not None))


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

    def on_step_begin(self, step: int, global_step: int) -> CallbackStepAction | None:
        """Runs once before every optimization step (batch)

        :param step: Step number/training round/batch index
        :param global_step: Effective global optimization step, same as step * batch_size
        """
        pass

    def on_step_end(
        self, step: int, global_step: int, info: "DNOInfoDict", hist: "list[DNOInfoDict]"
    ) -> CallbackStepAction | None:
        """Runs once after every optimization step (batch)

        :param step: Step number/training round/batch index
        :param global_step: Effective global optimization step, same as step * batch_size
        :param info: Current step's info dict
        :param hist: List (over steps) of dicts containing lists (over batch) with metrics
        """
        pass

    def __str__(self):
        return repr(self)

    def __repr__(self):
        def format_arg(attr_name: str):
            value = getattr(self, attr_name)
            return f"{attr_name}={repr(value)}"

        attrs = [format_arg(attr) for attr in self.__dict__ if not attr.startswith("_")]  # Only public
        return f"{self.__class__.__name__}({', '.join(attrs)})"


class CallbackList(list[Callback]):
    """Helper list class for invoking a list of callbacks."""

    def __init__(self, callbacks: Iterable[Callback] | None = None):
        super().__init__(callbacks or [])

    def invoke(
        self, dno: "DNO", callback_stage: Literal["train_begin", "train_end", "step_begin", "step_end"], *args, **kwargs
    ):
        actions = []
        for callback in self:
            callback.dno = dno
            action = callback.invoke(callback_stage, *args, **kwargs)
            actions.append(action)
        return CallbackStepAction.aggregate(actions)
