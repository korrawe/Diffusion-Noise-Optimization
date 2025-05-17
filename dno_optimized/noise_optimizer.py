import math
from typing import TYPE_CHECKING, Any

import torch
from torch.optim.optimizer import ParamsT
from tqdm import tqdm

from dno_optimized.gauss_newton import GaussNewton
from dno_optimized.levenberg_marquardt import LevenbergMarquardt
from dno_optimized.options import DNOOptions, OptimizerType

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter


def create_optimizer(
    optimizer: OptimizerType, params: ParamsT, config: DNOOptions, model, criterion
) -> torch.optim.Optimizer:
    print("Config:", config)
    match optimizer:
        case OptimizerType.Adam:
            return torch.optim.Adam(params, lr=config.lr)
        case OptimizerType.LBFGS:
            return torch.optim.LBFGS(
                params,
                lr=config.lr,
                line_search_fn=config.lbfgs.line_search_fn,
                max_iter=config.lbfgs.max_iter,
                history_size=config.lbfgs.history_size,
            )
        case OptimizerType.SGD:
            return torch.optim.SGD(params, lr=config.lr)
        case "GaussNewton":
            return GaussNewton(params, lr=config.lr)
        case "LevenbergMarquardt":
            # loss_fn = tlm.loss.LossWrapper(criterion)
            # return tlm.training.LevenbergMarquardtModule(
            #     model=Model(model, params),
            #     loss_fn=loss_fn,
            #     learning_rate=config.lr,
            #     attempts_per_step=config.levenbergMarquardt.attempts_per_step,
            #     solve_method='qr')
            return LevenbergMarquardt(
                params,
                lr=config.lr,
                damping_fac=config.levenbergMarquardt.damping_fac,
                attempts_per_step=config.levenbergMarquardt.attempts_per_step,
            )
        case _:
            raise ValueError(f"`{optimizer}` is not a valid optimizer")


class Model(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params

    def forward(self, z):
        return self.model(z)


class DNO:
    """
    Args:
        start_z: (N, 263, 1, 120)
    """

    # Variables from info dict to be logged to tensorboard (if enabled)
    TB_GLOBAL_VARS = ["lr", "perturb_scale"]
    TB_GROUP_VARS = ["loss", "loss_diff", "loss_decorrelate", "diff_norm", "grad_norm"]
    TB_HIST_VARS = ["x", "z"]

    def __init__(self, model, criterion, start_z, conf: DNOOptions, tb_writer: "SummaryWriter | None" = None):
        self.model = model
        self.criterion = criterion
        # for diff penalty
        self.start_z = start_z.detach()
        self.conf = conf

        self.current_z = self.start_z.clone().requires_grad_(True)
        # excluding the first dimension (batch size)
        self.dims = list(range(1, len(self.start_z.shape)))

        self.optimizer = create_optimizer(self.conf.optimizer, [self.current_z], self.conf, model, criterion)
        print(f"INFO: Using {self.conf.optimizer.name} optimizer with LR of {self.conf.lr:.2g}")

        self.lr_scheduler = []
        if conf.lr_warm_up_steps > 0:
            self.lr_scheduler.append(lambda step: warmup_scheduler(step, conf.lr_warm_up_steps))
            print(f"INFO: Using linear learning rate warmup over {conf.lr_warm_up_steps} steps")
        if conf.lr_decay_steps > 0:
            self.lr_scheduler.append(
                lambda step: cosine_decay_scheduler(step, conf.lr_decay_steps, conf.num_opt_steps, decay_first=False)
            )
            print(f"INFO: Using cosine learning rate decay over {conf.lr_decay_steps} steps")

        self.step_count = 0

        # Optimizer closure running variables
        self.last_x: torch.Tensor | None = None
        self.lr_frac: float | None = None

        # history of the optimization (for each step and each instance in the batch)
        # hist = {
        #    "step": [step] * batch_size,
        #    "lr": [lr] * batch_size,
        #    ...
        # }
        self.hist = []
        self.info = {}

        self.tb_writer = tb_writer

    @property
    def batch_size(self):
        return self.start_z.size(0)

    @property
    def global_step(self):
        return self.step_count * self.batch_size

    def __call__(self, num_steps: int | None = None):
        return self.optimize(num_steps=num_steps)

    def optimize(self, num_steps: int | None = None):
        if num_steps is None:
            num_steps = self.conf.num_opt_steps

        batch_size = self.start_z.shape[0]

        for i in (pb := tqdm(range(num_steps))):

            def closure():
                # Reset gradients
                self.optimizer.zero_grad()
                # Single step forward and backward
                self.last_x, self.lr_frac, loss = self.compute_loss(batch_size=batch_size)
                return loss

            # Perform optimization round
            self.optimizer.step(closure)

            # if isinstance(self.optimizer, torch.optim.Optimizer):
            #     self.compute_torch_optimizer(batch_size=batch_size)
            # elif isinstance(self.optimizer, tlm.training.LevenbergMarquardtModule):
            #     self.compute_levenberg_marquardt_optimizer()
            # else:
            #     raise NotImplementedError(f"Optimizer {self.optimizer} not supported")

            # Add noise after optimization step
            self.noise_perturbation(self.lr_frac, batch_size=batch_size)
            # Logging
            self.log_data(self.last_x)
            self.log_tensorboard(self.last_x, batch_size)
            pb.set_postfix({"loss": self.info["loss"].mean().item()})

        hist = self.compute_hist(batch_size=batch_size)

        assert self.last_x is not None, "Missing result"
        return {
            # last step's z
            "z": self.current_z.detach(),
            # previous step's x
            "x": self.last_x.detach(),
            "hist": hist,
        }

    # def compute_torch_optimizer(self, batch_size):
    #     def closure():
    #         # Reset gradients
    #         self.optimizer.zero_grad()
    #         # Single step forward and backward
    #         self.last_x, self.lr_frac, loss = self.compute_loss(batch_size=batch_size)
    #         return loss
    #
    #     # Perform optimization round
    #     self.optimizer.step(closure)
    #
    # def compute_levenberg_marquardt_optimizer(self):
    #     inputs = self.current_z
    #     targets = self.criterion.target
    #     self.optimizer.training_step(inputs, targets)

    def set_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr

    def compute_loss(self, batch_size):
        self.info = {"step": [self.step_count] * batch_size}

        # learning rate scheduler
        lr_frac = 1
        if len(self.lr_scheduler) > 0:
            for scheduler in self.lr_scheduler:
                lr_frac *= scheduler(self.step_count)
            self.set_lr(self.conf.lr * lr_frac)
        self.info["lr"] = [self.conf.lr * lr_frac] * batch_size

        # criterion
        x = self.model(self.current_z)
        # [batch_size,]
        loss = self.criterion(x)
        assert loss.shape == (batch_size,)
        self.info["loss"] = loss.detach().cpu()
        loss = loss.sum()

        # diff penalty
        if self.conf.diff_penalty_scale > 0:
            # [batch_size,]
            loss_diff = (self.current_z - self.start_z).norm(p=2, dim=self.dims)
            assert loss_diff.shape == (batch_size,)
            loss += self.conf.diff_penalty_scale * loss_diff.sum()
            self.info["loss_diff"] = loss_diff.detach().cpu()
        else:
            self.info["loss_diff"] = [0] * batch_size

        # decorrelate
        if self.conf.decorrelate_scale > 0:
            loss_decorrelate = noise_regularize_1d(
                self.current_z,
                dim=self.conf.decorrelate_dim,
            )
            assert loss_decorrelate.shape == (batch_size,)
            loss += self.conf.decorrelate_scale * loss_decorrelate.sum()
            self.info["loss_decorrelate"] = loss_decorrelate.detach().cpu()
        else:
            self.info["loss_decorrelate"] = [0] * batch_size

        # backward
        loss.backward()

        # log grad norm (before)
        self.info["grad_norm"] = self.current_z.grad.norm(p=2, dim=self.dims).detach().cpu()

        # grad mode
        self.current_z.grad.data /= self.current_z.grad.norm(p=2, dim=self.dims, keepdim=True)

        return x, lr_frac, loss

    def noise_perturbation(self, lr_frac, batch_size):
        # noise perturbation
        # match the noise fraction to the learning rate fraction
        noise_frac = lr_frac
        self.info["perturb_scale"] = [self.conf.perturb_scale * noise_frac] * batch_size

        noise = torch.randn_like(self.current_z)
        self.current_z.data += noise * self.conf.perturb_scale * noise_frac

    def log_data(self, x):
        # log the norm(z - start_z)
        self.info["diff_norm"] = (self.current_z - self.start_z).norm(p=2, dim=self.dims).detach().cpu()

        # log current z
        self.info["z"] = self.current_z.detach().cpu()
        self.info["x"] = x.detach().cpu()

        self.step_count += 1
        self.hist.append(self.info)

    def log_tensorboard(self, x, batch_size: int):
        if self.tb_writer is None:
            return

        # Functino to convert possibly tensor value to scalar
        def convert_value(x: Any):
            if isinstance(x, list):
                return sum(x) / len(x)
            if isinstance(x, torch.Tensor):
                return x.mean().detach().cpu().item()
            return x

        # Used for ordering variables in TB
        group_index = 0
        # Log all variables
        # Scalar value, or value broadcast to batch (e.g. learning rate)
        for var in self.TB_GLOBAL_VARS:
            scalar_value = convert_value(self.info[var])
            self.tb_writer.add_scalar(f"{group_index:02d}_dno/{var}", scalar_value, global_step=self.global_step)
        group_index += 1

        # Batched value (one per trial)
        for var in self.TB_GROUP_VARS:
            value = self.info[var]
            for trial, trial_value in enumerate(value):
                trial_value = convert_value(trial_value)
                self.tb_writer.add_scalar(
                    f"{group_index:02d}_{var}/trial_{trial}", trial_value, global_step=self.global_step
                )
            group_index += 1

        # Log noise and output histograms
        for var in self.TB_HIST_VARS:
            for trial, trial_value in enumerate(self.info[var]):
                self.tb_writer.add_histogram(
                    f"{group_index:02d}_hist_{var}/trial_{trial}", trial_value, global_step=self.global_step
                )
            group_index += 1

    def compute_hist(self, batch_size):
        # output is a list (over batch) of dict (over keys) of lists (over steps)
        hist = []
        for i in range(batch_size):
            hist.append({})
            for k in self.hist[0].keys():
                hist[-1][k] = [info[k][i] for info in self.hist]
        return hist


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
        return (math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1) / 2


def noise_regularize_1d(noise, stop_at=2, dim=3):
    """
    Args:
        noise (torch.Tensor): (N, C, 1, size)
        stop_at (int): stop decorrelating when size is less than or equal to stop_at
        dim (int): the dimension to decorrelate
    """
    all_dims = set(range(len(noise.shape)))
    loss = 0
    size = noise.shape[dim]

    # pad noise in the size dimention so that it is the power of 2
    if size != 2 ** int(math.log2(size)):
        new_size = 2 ** int(math.log2(size) + 1)
        pad = new_size - size
        pad_shape = list(noise.shape)
        pad_shape[dim] = pad
        pad_noise = torch.randn(*pad_shape).to(noise.device)

        noise = torch.cat([noise, pad_noise], dim=dim)
        size = noise.shape[dim]

    while True:
        # this loss penalizes spatially correlated noise
        # the noise is rolled in the size direction and the dot product is taken
        # (bs, )
        loss = loss + (noise * torch.roll(noise, shifts=1, dims=dim)).mean(
            # average over all dimensions except 0 (batch)
            dim=list(all_dims - {0})
        ).pow(2)

        # stop when size is 8
        if size <= stop_at:
            break

        # (N, C, 1, size) -> (N, C, 1, size // 2, 2)
        noise_shape = list(noise.shape)
        noise_shape[dim] = size // 2
        noise_shape.insert(dim + 1, 2)
        noise = noise.reshape(noise_shape)
        # average pool over (2,) window
        noise = noise.mean([dim + 1])
        size //= 2

    return loss
