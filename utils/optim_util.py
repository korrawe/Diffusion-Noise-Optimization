from typing import Literal

import torch
from torch.optim.optimizer import ParamsT

from dno import DNOOptions

OptimizerType = Literal['Adam', 'LBFGS', 'SGD', 'Gauss_Newton', 'LevenbergMarquardt']

def create_optimizer(optimizer: OptimizerType, params: ParamsT, config: DNOOptions) -> torch.optim.Optimizer:
    match optimizer:
        case "Adam":
            return torch.optim.Adam(params, lr=config.lr)
        case "LBFGS":
            return torch.optim.LBFGS(params, lr=config.lr, history_size=config.lbfgs.history_size)
        case "SGD":
            return torch.optim.SGD(params, lr=config.lr)
        case "Gauss_Newton":
            raise NotImplementedError(optimizer)
        case "LevenbergMarquardt":
            raise NotImplementedError(optimizer)
        case _:
            raise ValueError(f"`{optimizer}` is not a valid optimizer")