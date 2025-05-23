import inspect
import os
import textwrap
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, Self, Type, TypeVar

from tqdm import tqdm

from dno_optimized.options import GenerateOptions

if TYPE_CHECKING:
    from ..noise_optimizer import DNO, DNOInfoDict, DNOStateDict


def _terminal_width(fallback: int = 80):
    try:
        columns, _ = os.get_terminal_size()
        return columns
    except OSError:
        # Fallback if terminal size cannot be determined
        return fallback


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

    def __init__(self, every_n_steps: int | None = None, start_after: int | None = None):
        """Initialize callback

        :param every_n_steps: Set this value to run the callback only every n training steps. Starts counting at 1. Only
            affects `on_train_[...]` callbacks.
        :param start_after: Set this value to only start running the callback after the first n training steps. Can be
            combined with every_n_steps. Only affects `on_train_[...]` callbacks.
        """
        super().__init__()
        self.dno: "DNO"
        self.progress: tqdm
        self.every_n_steps = every_n_steps or 1
        self.start_after = start_after or 0

    def __post_init__(self, callbacks: "CallbackList"):
        """Called once all callbacks have been instantiated. Can be overridden to depend on other callbacks.

        :param callbacks: List of callbacks in use.
        """
        pass

    def _should_run_step_callback(self):
        step = self.dno.step_count
        return step >= self.start_after and (step % self.every_n_steps) == 0

    def invoke(
        self, callback_stage: Literal["train_begin", "train_end", "step_begin", "step_end"], **kwargs
    ) -> CallbackStepAction | None:
        # Invoke function only with kwargs that are actually accepted by the function. This way, callbacks can choose
        # what kwargs to accept.
        def invoke_with_kwargs(func, kwargs):
            sig = inspect.signature(func)
            accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return func(**accepted)

        match callback_stage:
            case "train_begin":
                return invoke_with_kwargs(self.on_train_begin, kwargs)
            case "train_end":
                return invoke_with_kwargs(self.on_train_end, kwargs)
            case "step_begin":
                if self._should_run_step_callback():
                    return invoke_with_kwargs(self.on_step_begin, kwargs)
            case "step_end":
                if self._should_run_step_callback():
                    return invoke_with_kwargs(self.on_step_end, kwargs)
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

    def on_train_end(self, num_steps: int, batch_size: int, hist: "list[DNOInfoDict]", state_dict: "DNOStateDict"):
        """Runs once when the training is done (completed all iterations or other callback resulted in a stop condition).

        :param num_steps: Number of optimization steps (effective)
        :param batch_size: Batch size (number of trials)
        :param hist List (over batch) of dicts containing lists (over steps) with metrics
        :param state_dict State dict computed after the last optimization step
        """
        pass

    def on_step_begin(self, step: int) -> CallbackStepAction | None:
        """Runs once before every optimization step (batch)

        :param step: Step number/training round/batch index
        """
        pass

    def on_step_end(self, step: int, info: "DNOInfoDict", hist: "list[DNOInfoDict]") -> CallbackStepAction | None:
        """Runs once after every optimization step (batch)

        :param step: Step number/training round/batch index
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
        attrs_str = ", ".join(attrs)
        attrs_str = textwrap.fill(attrs_str, width=_terminal_width(), initial_indent="", subsequent_indent=" " * 6)
        return f"{self.__class__.__name__}({attrs_str})"


T = TypeVar("T")


class CallbackList(list[Callback]):
    """Helper list class for invoking a list of callbacks."""

    def __init__(self, callbacks: Iterable[Callback] | None = None):
        super().__init__(callbacks or [])

    def invoke(
        self, dno: "DNO", callback_stage: Literal["train_begin", "train_end", "step_begin", "step_end"], **kwargs
    ):
        actions = []
        if "pb" in kwargs:
            pb = kwargs["pb"]
            del kwargs["pb"]
        else:
            pb = None
        for callback in self:
            callback.dno = dno
            if pb:
                callback.progress = pb
            action = callback.invoke(callback_stage, **kwargs)
            actions.append(action)
        return CallbackStepAction.aggregate(actions)

    def has(self, callback_type: Type[T]) -> bool:
        return any(isinstance(cb, callback_type) for cb in self)

    def count(self, callback_type: Type[T]) -> int:
        return len([cb for cb in self if isinstance(cb, callback_type)])

    def get_all(self, callback_type: Type[T]) -> list[T]:
        results: list[T] = []
        for callback in self:
            if isinstance(callback, callback_type):
                results.append(callback)
        return results

    def get(self, callback_type: Type[T], index: int | None = None, default: T | None = None) -> T | None:
        results = self.get_all(callback_type)
        if len(results) == 0:
            return default
        if index is not None:
            return results[index]
        return results[0]

    def extend(self, other: "CallbackList", mode: Literal["replace", "append"] = "replace") -> None:
        if mode == "append":
            super(CallbackList, self).extend(other)
        elif mode == "replace":
            final = []
            for cb in self:
                if not other.has(type(cb)):
                    final.append(cb)
            final.extend(other)
            self[:] = final  # Replace own
        else:
            raise ValueError(mode)

    def post_init(self):
        for cb in self:
            cb.__post_init__(self)
