"""
Source: https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.func import functional_call
from torch.optim.optimizer import Optimizer
from typing import List


class GaussNewton(Optimizer):
    r"""Implements Gauss-Newton.
    """

    def __init__(self, params, lr: float, model: nn.Module, hessian_approx: bool = True):

        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)

        super(GaussNewton, self).__init__(params, defaults)

        self.hessian_approx = hessian_approx

        self._model = model
        self._params = self.param_groups[0]['params']
        self._j_list = []
        self._h_list = []

    def __setstate__(self, state):
        super(GaussNewton, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, x: torch.Tensor, closure=None):
        """Performs a single optimization step.

        Args:
            x: Current data batch, which is needed to compute 2nd order derivatives
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            parameters = dict(self._model.named_parameters())
            keys, values = zip(*parameters.items())

            self._h_list = []
            if self.hessian_approx:
                # vectorized jacobian (https://github.com/pytorch/pytorch/issues/49171)
                def func(*params: torch.Tensor):
                    out = functional_call(self._model, {n: p for n, p in zip(keys, params)}, x)
                    return out

                self._j_list: tuple[torch.Tensor] = torch.autograd.functional.jacobian(func, values,
                                                                                       create_graph=False)  # NxCxBxCxHxW
                # create hessian approximation
                for i, j in enumerate(self._j_list):
                    j = j.flatten(end_dim=len(self._j_list[i].shape) - len(d_p_list[i].shape) - 1).flatten(
                        start_dim=1)  # (NC)x(BCHW)
                    try:
                        h = j.T.matmul(j)
                    except RuntimeError:
                        h = None
                    self._h_list.append(h)

            else:
                # vectorized hessian (https://github.com/pytorch/pytorch/issues/49171)
                def func(*params: torch.Tensor):
                    out: torch.Tensor = functional_call(self._model, {n: p for n, p in zip(keys, params)}, x)
                    return out.square().sum()

                self._h_list: tuple[torch.Tensor] = torch.autograd.functional.hessian(func,
                                                                                      tuple(self._model.parameters()),
                                                                                      create_graph=False)
                self._h_list = [self._h_list[i][i] for i in range(len(self._h_list))]  # filter j-th element
                self._h_list = [
                    h.flatten(end_dim=len(self._h_list[i].shape) - len(d_p_list[i].shape) - 1).flatten(start_dim=1) for
                    i, h in enumerate(self._h_list)]  # (NC)x(BCHW)

            self.gna_update(
                params_with_grad,
                d_p_list,
                lr=lr,
            )

        return loss

    def gna_update(
        self,
        params: List[Tensor],
        d_p_list: List[Tensor],
        lr: float,
    ):
        r"""Functional API that performs Gauss-Newton algorithm computation.

        """

        assert len(d_p_list) == len(self._h_list), 'Layer number mismatch'

        # e.g. 4 list elements in raw_train_fit.py for hidden and predict weights and biases
        for i, param in enumerate(params):

            d_p = d_p_list[i]
            h = self._h_list[i]
            if h is None:
                # fall back to stochastic gradient descent
                param.add_(d_p, alpha=-lr)
                break
            # prevent zeros along Hessian diagonal
            diag_vec = h.diagonal() + torch.finfo(h.dtype).eps * 1
            h.as_strided([h.size(0)], [h.size(0) + 1]).copy_(diag_vec)
            h_i = h.pinverse()
            if h_i.shape[-1] == d_p.flatten().shape[0]:
                d2_p = h_i.matmul(d_p.flatten()).reshape(d_p_list[i].shape)
                param.add_(d2_p, alpha=-lr)
            else:
                raise Exception('Tensor dimension mismatch')