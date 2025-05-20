from dno_optimized.callbacks.callback import Callback
from dno_optimized.callbacks.early_stopping import EarlyStoppingCallback
from dno_optimized.callbacks.tensorboard import TensorboardCallback
from dno_optimized.options import GenerateOptions


def create_callback(name: str, options: GenerateOptions, callback_args: dict) -> Callback:
    match name:
        case "tensorboard":
            return TensorboardCallback.from_config(options, callback_args)
        case "early_stopping":
            return EarlyStoppingCallback.from_config(options, callback_args)
        case _:
            raise KeyError(f"`{name}` is not a valid callback name")