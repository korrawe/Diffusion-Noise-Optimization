
from dno_optimized.callbacks.callback import Callback, CallbackList
from dno_optimized.callbacks.early_stopping import EarlyStoppingCallback
from dno_optimized.callbacks.save_top_k import SaveTopKCallback
from dno_optimized.callbacks.tensorboard import TensorboardCallback
from dno_optimized.options import CallbackConfig, GenerateOptions


def create_callback(name: str, options: GenerateOptions, callback_args: dict) -> Callback:
    match name:
        case "tensorboard":
            return TensorboardCallback.from_config(options, callback_args)
        case "early_stopping":
            return EarlyStoppingCallback.from_config(options, callback_args)
        case "save_top_k":
            return SaveTopKCallback.from_config(options, callback_args)
        case _:
            raise KeyError(f"`{name}` is not a valid callback name")


def default_callbacks(options: GenerateOptions) -> CallbackList:
    return CallbackList(
        [
            TensorboardCallback.from_config(options, {"every_n_steps": 10}),
            EarlyStoppingCallback.from_config(options, {"patience": 50, "min_improvement": 1e-4, "abs_value": 1e-5}),
            SaveTopKCallback.from_config(options, {})
        ]
    )


def callback_list_from_config(configs: list[CallbackConfig], options: GenerateOptions):
    return CallbackList([create_callback(conf.name, options, conf.args) for conf in configs])
