import math
import torch
from torch.optim import Optimizer
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class NoiseOptOptions:
    unroll_steps: int
    opt_steps: int
    optimizer: str = "adam"
    last_unroll_steps: int = None
    lr: float = 5e-2
    momentum: float = 0.9
    adam_beta2: float = 0.999
    perturb_scale: float = 5e-4
    diff_penalty_scale: float = 0
    grad_clip_norm: float = 0
    grad_clip_value: float = 0
    grad_mode: str = "sign"
    lr_warm_up_steps: int = 0
    lr_scheduler: str = "constant"
    lr_decay_steps: int = 0
    sgdr_mult_factor: float = 1
    noise_scheduler: str = "match"
    noise_decay_steps: int = 0
    standardize_z_before_step: bool = False
    standardize_z_after_step: bool = False
    decorrelate_scale: float = 1e6
    decorrelate_stop_at: int = 2
    decorrelate_feature_dim: bool = False
    starting_noise_seed: int = 0
    separate_backward_for_ode: bool = True
    model_has_text: bool = False
    postfix: str = ""

    @property
    def name(self):
        out = [f"noiseopt{self.opt_steps}"]
        out.append(f"unroll{self.unroll_steps}")
        if self.last_unroll_steps is not None and self.last_unroll_steps != self.unroll_steps:
            out.append(f"lastunroll{self.last_unroll_steps}")
        # list all properties
        if self.optimizer == "adam":
            out.append(f"adam-{self.momentum}-{self.adam_beta2}")
        elif self.optimizer == "sgd":
            out.append(f"sgd-{self.momentum}")
        out.append(f"lr{self.lr}")
        if self.lr_warm_up_steps > 0:
            out.append(f"warmup{self.lr_warm_up_steps}")
        if self.lr_scheduler != "constant" and self.lr_decay_steps > 0:
            if self.lr_scheduler == 'sgdr' and self.sgdr_mult_factor != 1:
                out.append(f"{self.lr_scheduler}{self.lr_decay_steps}mult{self.sgdr_mult_factor}")
            else:
                out.append(f"{self.lr_scheduler}{self.lr_decay_steps}")
        out.append(f"noise{self.perturb_scale}")
        if self.diff_penalty_scale > 0:
            out.append(f"diff{self.diff_penalty_scale}")
        if self.noise_scheduler != "constant" and self.noise_decay_steps > 0:
            if self.noise_scheduler == "match":
                if self.lr_scheduler != "constant" and self.lr_decay_steps > 0:
                    out.append(f"noise{self.noise_scheduler}")
            else:
                out.append(f"noise{self.noise_scheduler}{self.noise_decay_steps}")
        if self.grad_clip_norm > 0:
            out.append(f"clipnorm{self.grad_clip_norm}")
        if self.grad_clip_value > 0:
            out.append(f"clipvalue{self.grad_clip_value}")
        if self.grad_mode != "normal":
            out.append(f"grad{self.grad_mode}")
        if self.standardize_z_before_step:
            out.append("standardizebf")
        if self.standardize_z_after_step:
            out.append("standardize")
        if self.decorrelate_scale > 0:
            if self.decorrelate_feature_dim:
                out.append(f"decorr-feat{self.decorrelate_scale}f")
            else:
                out.append(f"decorr{self.decorrelate_scale}")
            if self.decorrelate_stop_at != 8:
                out.append(f"stop{self.decorrelate_stop_at}")
        if self.separate_backward_for_ode:
            out.append("odegrad")
        if self.model_has_text:
            out.append("wtext")
        if self.starting_noise_seed != 0:
            out.append(f"seed{self.starting_noise_seed}")
        if self.postfix != "":
            out.append(self.postfix)
        return "_".join(out)

class NoiseOptimizer:
    """
    Args:
        start_z: (N, 263, 1, 120)
    """

    def __init__(
        self,
        model,
        criterion,
        start_z,
        conf: NoiseOptOptions,
        noise_seeds: list = None,
    ):
        self.model = model
        self.criterion = criterion
        self.start_z = start_z.detach()
        self.conf = conf
        self.noise_seeds = noise_seeds

        self.current_z = self.start_z.clone().requires_grad_(True)
        # excluding the first dimension (batch size)
        self.dims = list(range(1, len(self.start_z.shape)))

        if conf.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                [self.current_z], lr=conf.lr, betas=(conf.momentum, conf.adam_beta2)
            )
        elif conf.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                [self.current_z], lr=conf.lr, momentum=conf.momentum
            )
        else:
            raise NotImplementedError(f"Optimizer {conf.optimizer} not implemented")

        self.lr_scheduler = []
        if conf.lr_warm_up_steps > 0:
            self.lr_scheduler.append(
                lambda step: warmup_scheduler(step, conf.lr_warm_up_steps)
            )
        if conf.lr_scheduler == "constant" or conf.lr_decay_steps == 0:
            pass
        elif conf.lr_scheduler == "cosine":
            self.lr_scheduler.append(
                lambda step: cosine_decay_scheduler(
                    step, conf.lr_decay_steps, conf.opt_steps, decay_first=False
                )
            )
        elif conf.lr_scheduler == "quadratic":
            self.lr_scheduler.append(
                lambda step: quadratic_decay_scheduler(
                    step, conf.lr_decay_steps, conf.opt_steps, decay_first=False
                )
            )
        elif conf.lr_scheduler == "linear":
            self.lr_scheduler.append(
                lambda step: linear_decay_scheduler(
                    step, conf.lr_decay_steps, conf.opt_steps, decay_first=False
                )
            )
        elif conf.lr_scheduler == 'sgdr':
            self.lr_scheduler.append(
                lambda step: sgdr_scheduler(
                    step, conf.lr_decay_steps, conf.opt_steps, mult_factor=conf.sgdr_mult_factor
                )
            )
        else:
            raise NotImplementedError(
                f"LR scheduler {conf.lr_scheduler} not implemented"
            )

        self.noise_scheduler = []
        if conf.noise_scheduler == "constant" or conf.noise_decay_steps == 0:
            pass
        elif conf.noise_scheduler == "quadratic":
            self.noise_scheduler.append(
                lambda step: quadratic_decay_scheduler(
                    step, conf.noise_decay_steps, conf.opt_steps, decay_first=True
                )
            )
        elif conf.noise_scheduler == "linear":
            self.noise_scheduler.append(
                lambda step: linear_decay_scheduler(
                    step, conf.noise_decay_steps, conf.opt_steps, decay_first=True
                )
            )
        elif conf.noise_scheduler == "match":
            # match the noise decay with the lr decay
            pass
        else:
            raise NotImplementedError()

        self.step_count = 0
        # history of the optimization (for each step and each instance in the batch)
        # hist = {
        #    "step": [step] * batch_size,
        #    "lr": [lr] * batch_size,
        #    ...
        # }
        self.hist = []

    def __call__(self, num_steps: int = None):
        if num_steps is None:
            num_steps = self.conf.opt_steps

        batch_size = self.start_z.shape[0]
        with tqdm(range(num_steps)) as prog:
            for i in prog:
                # info = {"step": self.step_count}
                info = {"step": [self.step_count] * batch_size}

                # learning rate scheduler
                lr_frac = 1
                if len(self.lr_scheduler) > 0:
                    for scheduler in self.lr_scheduler:
                        lr_frac *= scheduler(self.step_count)
                    self.set_lr(self.conf.lr * lr_frac)
                info["lr"] = [self.conf.lr * lr_frac] * batch_size

                # standardize z (before)
                if self.conf.standardize_z_before_step:
                    mean = self.current_z.mean(dim=self.dims, keepdim=True)
                    std = self.current_z.std(dim=self.dims, keepdim=True)
                    self.current_z.data = (self.current_z - mean) / std

                # forward
                x = self.model(self.current_z)
                # [batch_size,]
                loss = self.criterion(x)
                assert loss.shape == (batch_size,)
                info["loss"] = loss.detach().cpu()
                loss = loss.sum()

                def aux_loss():
                    aux_loss = 0
                    # diff penalty
                    if self.conf.diff_penalty_scale > 0:
                        # [batch_size,]
                        loss_diff = (
                            (self.current_z - self.start_z).norm(p=2, dim=self.dims)
                        )
                        assert loss_diff.shape == (batch_size,)
                        aux_loss += self.conf.diff_penalty_scale * loss_diff.sum()
                        info["loss_diff"] = loss_diff.detach().cpu()
                    else:
                        info["loss_diff"] = [0] * batch_size

                    # decorrelate
                    if self.conf.decorrelate_scale > 0:
                        if self.noise_seeds is None:
                            seed_list = None
                        else:
                            seed_list = [seed * 10_000 + i for seed in self.noise_seeds]
                        loss_decorrelate = noise_regularize_1d(self.current_z, stop_at=self.conf.decorrelate_stop_at, dim=3, seeds=seed_list)
                        if self.conf.decorrelate_feature_dim:
                            loss_decorrelate += noise_regularize_1d(self.current_z, stop_at=self.conf.decorrelate_stop_at, dim=1, seeds=seed_list)
                        assert loss_decorrelate.shape == (batch_size,)
                        aux_loss += self.conf.decorrelate_scale * loss_decorrelate.sum()
                        info["loss_decorrelate"] = loss_decorrelate.detach().cpu()
                    else:
                        info["loss_decorrelate"] = [0] * batch_size

                    return aux_loss

                if not self.conf.separate_backward_for_ode:
                    loss += aux_loss()

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # grad mode
                if self.conf.grad_mode == "normal":
                    pass
                elif self.conf.grad_mode == "unit":
                    self.current_z.grad.data /= self.current_z.grad.norm(
                        p=2, dim=self.dims, keepdim=True
                    )
                elif self.conf.grad_mode == "sign":
                    self.current_z.grad.data = self.current_z.grad.sign()
                else:
                    raise NotImplementedError(
                        f"Grad mode {self.conf.grad_mode} not implemented"
                    )

                # backward for aux loss
                if self.conf.separate_backward_for_ode:
                    # only apply if aux_loss is tensor
                    _loss = aux_loss()
                    if isinstance(_loss, torch.Tensor):
                        _loss.backward()

                # log grad norm (before)
                info["grad_norm"] = self.current_z.grad.norm(p=2, dim=self.dims).detach().cpu()

                # clip grad
                if self.conf.grad_clip_norm > 0:
                    batch_clip_grad_norm(self.current_z, self.conf.grad_clip_norm)
                if self.conf.grad_clip_value > 0:
                    raise NotImplementedError(f"grad_clip_value not implemented")
                    torch.nn.utils.clip_grad_value_(
                        self.current_z, self.conf.grad_clip_value
                    )
                # log grad norm (after)
                info["grad_norm_after"] = self.current_z.grad.norm(p=2, dim=self.dims).detach().cpu()

                # optimize z
                self.optimizer.step()

                # noise perturbation
                noise_frac = 1
                if self.conf.noise_scheduler == "match":
                    noise_frac = lr_frac
                else:
                    if len(self.noise_scheduler) > 0:
                        noise_frac = 1
                        for scheduler in self.noise_scheduler:
                            noise_frac *= scheduler(self.step_count)
                info["perturb_scale"] = [self.conf.perturb_scale * noise_frac] * batch_size

                if self.noise_seeds is None:
                    noise = torch.randn_like(self.current_z)
                else:
                    # deterministic noise
                    seed_list = [seed * 10_000 + i for seed in self.noise_seeds]
                    noise = generate_det_noise(self.current_z.shape, seed_list).to(self.current_z.device)
                    
                self.current_z.data += (
                    noise
                    * self.conf.perturb_scale
                    * noise_frac
                )

                # standardize z
                if self.conf.standardize_z_after_step:
                    mean = self.current_z.mean(dim=self.dims, keepdim=True)
                    std = self.current_z.std(dim=self.dims, keepdim=True)
                    self.current_z.data = (self.current_z - mean) / std

                # log the norm(z - start_z)
                info["diff_norm"] = (
                    (self.current_z - self.start_z)
                    .norm(p=2, dim=self.dims)
                    .detach()
                    .cpu()
                )

                # log current z
                info["z"] = self.current_z.detach().cpu()
                info["x"] = x.detach().cpu()

                self.step_count += 1
                self.hist.append(info)
                prog.set_postfix({"loss": info["loss"].mean().item()})

            # output is a list (over batch) of dict (over keys) of lists (over steps)
            hist = []
            for i in range(batch_size):
                hist.append({})
                for k in self.hist[0].keys():
                    hist[-1][k] = [info[k][i] for info in self.hist]
            # hist = {}
            # for k in self.hist[0].keys():
            #     hist[k] = [info[k] for info in self.hist]

            return {
                "z": self.current_z.detach(),
                "x": x.detach(),
                "hist": hist,
            }

    def set_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr

def generate_det_noise(shape, seeds):
    noise = []
    for seed in seeds:
        torch.manual_seed(seed)
        noise.append(torch.randn(shape[1:]))
    noise = torch.stack(noise, dim=0)
    return noise


def warmup_scheduler(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1


def cosine_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using cosine decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return (math.cos((step) / decay_steps * math.pi) + 1) / 2
    else:
        if step < total_steps - decay_steps:
            return 1
        return (
            math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1
        ) / 2

def sgdr_scheduler(step, cycle_length, total_steps, mult_factor=1):
    """
    SGDR learning rate scheduler.
    :param step: int, current step.
    :param cycle_length: int, cycle length.
    :param min_lr: float, minimum learning rate.
    :param max_lr: float, maximum learning rate.
    :param mult_factor: float, scaling factor for the next cycle length.
    :return: float, learning rate.
    """
    offset = 0
    if offset + cycle_length + mult_factor * cycle_length > total_steps:
        cycle_length = total_steps - offset
    while step - cycle_length > 0:
        offset += cycle_length
        step -= cycle_length
        cycle_length *= mult_factor
        if offset + cycle_length + mult_factor * cycle_length > total_steps:
            cycle_length = total_steps - offset

    x = (step % cycle_length) / cycle_length
    return (1 + math.cos(x * math.pi)) / 2

def quadratic_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using quadratic decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return ((decay_steps - step) / decay_steps) ** 2
    else:
        if step < total_steps - decay_steps:
            return 1
        return ((decay_steps - (step - (total_steps - decay_steps))) / decay_steps) ** 2


def linear_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using linear decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return (decay_steps - step) / decay_steps
    else:
        if step < total_steps - decay_steps:
            return 1
        return (decay_steps - (step - (total_steps - decay_steps))) / decay_steps


def noise_regularize_1d(noise, stop_at=8, dim=3, seeds=None):
    """
    Args:
        noise (torch.Tensor): (N, C, 1, size)
        stop_at (int): stop decorrelating when size is less than or equal to stop_at
        dim (int): the dimension to decorrelate
    """
    loss = 0
    size = noise.shape[dim]
    
    # pad noise in the size dimention so that it is the power of 2
    if size != 2 ** int(math.log2(size)):
        new_size = 2 ** int(math.log2(size) + 1)
        pad = (new_size - size)
        pad_shape = list(noise.shape)
        pad_shape[dim] = pad
        
        if seeds is not None:
            pad_noise = generate_det_noise(pad_shape, seeds).to(noise.device)
        else:
            pad_noise = torch.randn(*pad_shape).to(noise.device)

        noise = torch.cat([noise, pad_noise], dim=dim)
        size = noise.shape[dim] 

    while True:
        # this loss penalizes spatially correlated noise
        # the noise is rolled in the size direction and the dot product is taken
        # (bs, )
        loss = (
            loss
            + (noise * torch.roll(noise, shifts=1, dims=dim)).mean(dim=[1,2,3]).pow(2)
        )

        # stop when size is 8
        if size <= stop_at:
            break

        # (N, C, 1, size) -> (N, C, 1, size // 2, 2)
        noise_shape = list(noise.shape)
        noise_shape[dim] = size // 2
        noise_shape.insert(dim+1, 2)
        noise = noise.reshape(noise_shape)
        # average pool over (2,) window
        noise = noise.mean([dim+1])
        size //= 2

    return loss

def batch_clip_grad_norm(
        parameters, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    assert isinstance(parameters, torch.Tensor), f"parameters must be a tensor"
    assert parameters.requires_grad, f"parameters must require grad"
    # (bs, x, x, x)
    grads = parameters.grad
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    dims = list(range(1, len(grads.shape)))
    device = grads.device
    if norm_type == float('inf'):
        raise NotImplementedError("inf norm is not implemented")
    else:
        # (bs, 1, 1, 1)
        total_norm = torch.norm(grads, norm_type, dim=dims, keepdim=True)

    if torch.logical_or(total_norm.isnan(), total_norm.isinf()).any():
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            f"parameters {parameters} is non-finite, so it cannot be clipped."
        )

    # (bs, 1, 1, 1)
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    grads.detach().mul_(clip_coef_clamped.to(grads.device))
    return total_norm