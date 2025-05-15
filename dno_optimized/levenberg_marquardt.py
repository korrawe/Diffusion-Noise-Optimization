"""
Sources:
- https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
- https://github.com/hahnec/torchimize/blob/master/torchimize/functions/single/lma_fun_single.py
- https://github.com/fabiodimarco/torch-levenberg-marquardt/blob/main/src/torch_levenberg_marquardt/training.py
- https://github.com/rahulkidambi/AccSGD/blob/master/AccSGD.py
- https://github.com/pytorch/pytorch/blob/main/torch/optim/lbfgs.py
"""

import torch
from torch.optim import Optimizer

from torch.optim.optimizer import ParamsT
from typing import Any

class LevenbergMarquardt(Optimizer):
    def __init__(self, params: ParamsT, lr: float, damping_fac: float, max_iter: int = 10, attempts_per_step: int = 10):
        defaults = dict(
            lr=lr,
            damping_fac=damping_fac,
            max_iter=max_iter,
            attempts_per_step=attempts_per_step,
        )

        super().__init__(params, defaults)

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    @torch.no_grad()
    def step(self, closure: Any) -> Any:
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        # Initialize hyperparameters
        group = self.param_groups[0]
        lr = group['lr']
        damping_fac = group['damping_fac']
        attempts_per_step = group['attempts_per_step']
        max_iter = group['max_iter']
        gtol = group['gtol']
        ptol = group['ptol']
        ftol = group['ftol']

        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        # flat_grad = self._gather_flat_grad()
        # opt_cond = flat_grad.abs().max() <= tolerance_grad

        fun = lambda x: torch.sqrt(x)
        jac_fun = lambda x: torch.autograd.functional.jacobian(x, fun)

        p = orig_loss
        f = fun(p)
        j = jac_fun(p)
        g = torch.matmul(j.T, f)
        H = torch.matmul(j.T, j)
        u = damping_fac * torch.max(torch.diag(H))
        v = 2
        p_list = [p]
        while len(p_list) < max_iter:
            D = torch.eye(j.shape[1], device=j.device)
            h = -torch.linalg.lstsq(H + u * D, g, rcond=None, driver=None)[0]
            f_h = fun(p + h)
            rho_denom = torch.matmul(h, u * h - g)
            rho_nom = torch.matmul(f, f) - torch.matmul(f_h, f_h)
            rho = rho_nom / rho_denom if rho_denom > 0 else torch.inf if rho_nom > 0 else -torch.inf
            if rho > 0:
                p = p + h
                j = jac_fun(p)
                g = torch.matmul(j.T, fun(p))
                H = torch.matmul(j.T, j)
            p_list.append(p.clone())
            f_prev = f.clone()
            f = fun(p)
            u, v = (u * torch.max(torch.tensor([1 / 3, 1 - (2 * rho - 1) ** 3])), 2) if rho > 0 else (u * v, v * 2)

            # stop conditions
            gcon = max(abs(g)) < gtol
            pcon = (h ** 2).sum() ** .5 < ptol * (ptol + (p ** 2).sum() ** .5)
            fcon = ((f_prev - f) ** 2).sum() < ((ftol * f) ** 2).sum() if rho > 0 else False
            if gcon or pcon or fcon:
                break


        return orig_loss
