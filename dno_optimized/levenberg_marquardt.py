# Adapted from: https://github.com/fabiodimarco/torch-levenberg-marquardt/blob/main/src/torch_levenberg_marquardt/training.py

from typing import Any, Literal, Callable

import torch
from torch import Tensor

from torch.func import jacrev, vmap
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from torch_levenberg_marquardt.damping import DampingStrategy, StandardDampingStrategy
from torch_levenberg_marquardt.selection import ParamSelectionStrategy
from torch_levenberg_marquardt.tree import tree_cat


class LevenbergMarquardt(Optimizer):
    """Levenberg-Marquardt training module."""

    def __init__(
        self,
        params: ParamsT,
        model: Callable[[Tensor], Tensor],
        loss_fn: Callable[[Tensor], Tensor],
        damping_strategy: DampingStrategy | None = None,
        learning_rate: float = 1.0,
        attempts_per_step: int = 10,
        solve_method: Literal['qr', 'cholesky', 'solve'] = 'qr',
        param_selection_strategy: ParamSelectionStrategy | None = None,
        use_vmap: bool = False,
        max_batch_size: int | None = None,
    ) -> None:
        """Initializes `LevenbergMarquardtModule` instance.

        Args:
            model: The model to be trained, expected to inherit from `torch.nn.Module`.
            loss_fn: A custom loss function inheriting from `Loss`.
                Defaults to `MSELoss()`.
            damping_strategy: Damping strategy to use during training.
                Defaults to `StandardDampingStrategy`.
            learning_rate: Specifies the step size for updating the model parameters.
                The update is performed using the formula `w = w - lr * updates`,
                where `updates` are calculated by the Levenberg-Marquardt algorithm.
            attempts_per_step: Defines the maximum number of attempts allowed during a
                training step to compute a valid model update that reduces the loss on
                the current batch. During each attempt, new model parameters are
                computed, and the resulting loss (`new_loss`) is compared to the
                previous loss. If `new_loss < loss`, the new parameters are accepted.
                Otherwise, the old parameters are restored, and a new attempt is made
                with an adjusted damping factor. If the maximum number of attempts is
                reached without reducing the loss, the step is finalized with the last
                computed parameters, even if they do not decrease the loss.
            solve_method: Solver to use for the linear system. Options:
                - 'qr': QR decomposition (robust, slower).
                - 'cholesky': Cholesky decomposition (fast, less stable).
                - 'solve': Direct solve (balanced speed and robustness).
            param_selection_strategy: A `ParamSelectionStrategy` instance defining how
                subsets of parameters are chosen each training_step. If None, all
                parameters are used.
            use_vmap: Specifies whether to use `torch.vmap` for Jacobian computation.
                Enabling `vmap` is generally the preferred choice as it is faster
                and requires less memory, especially for medium to large models.
                For very small models or simple cases, computing the Jacobian
                without `vmap` might be marginally more efficient. Defaults to `True`.
            max_batch_size: If set, the input batch is divided into smaller sub-batches
                of this size when computing the Jacobian and forming the Gauss-Newton
                approximations. Each sub-batch processes a portion of the input data at
                a time, allowing the Hessian approximation and the RHS vector to be
                constructed incrementally. This reduces peak memory usage but can limit
                parallelism, as the computations are partially serialized rather than
                fully utilizing hardware resources for parallel computation.
        """
        defaults = dict(
            lr=learning_rate,
        )
        super().__init__(params, defaults)

        self._model = model

        # Set up loss function and damping strategy
        self.loss_fn = loss_fn
        self.damping_strategy = damping_strategy or StandardDampingStrategy()

        self.learning_rate = learning_rate
        self.attempts_per_step = attempts_per_step
        self.solve_method = solve_method
        self.param_selection_strategy = param_selection_strategy
        self.use_vmap = use_vmap
        self.max_batch_size = max_batch_size

        # Extract trainable parameters
        self._params = params
        self._num_params = sum(p.numel() for p in self._params)

        # Precompute splits for flat_params
        self._splits = [p.numel() for p in self._params]

        # Flatten all trainable parameters into a single tensor
        self.flat_params = torch.cat(
            [p.detach().flatten() for p in self._params]
        )

        # Bind model parameters to slices of the flat parameter tensor
        start = 0
        for p in self._params:
            size = p.numel()
            p.data = self.flat_params[start : start + size].view_as(p)
            start += size

        # Backup storage for parameters
        self._flat_params_backup: Tensor = self.flat_params.clone()

        self._batch_size: int | None = None
        self._num_residuals: int | None = None

    @torch.no_grad()
    def backup_parameters(self) -> None:
        """Backs up the current model parameters into a separate tensor."""
        self._flat_params_backup = self.flat_params.clone()

    @torch.no_grad()
    def restore_parameters(self) -> None:
        """Restores model parameters from the backup tensor."""
        self.flat_params.copy_(self._flat_params_backup)

    @torch.no_grad()
    def reset(self) -> None:
        """Resets internal state, including the damping factor and outputs."""
        self._batch_size = None
        self._num_residuals = None
        self.damping_strategy.reset()

    def forward(self, params) -> Any:
        """Performs a forward pass using the current model parameters."""
        # return functional_call(self._model, self._params_and_buffers, inputs)
        return self._model(params)

    @torch.no_grad()
    def _solve(self, matrix: Tensor, rhs: Tensor) -> Tensor:
        """Solves the linear system using the specified solver.

        Args:
            matrix: The matrix representing the linear system.
            rhs: The right-hand side vector.

        Returns:
            The solution vector.
        """

        if self.solve_method == 'qr':
            q, r = torch.linalg.qr(matrix)
            y = torch.matmul(q.transpose(-2, -1), rhs)
            return torch.linalg.solve_triangular(r, y, upper=True)
        elif self.solve_method == 'cholesky':
            L = torch.linalg.cholesky(matrix)
            y = torch.linalg.solve_triangular(L, rhs, upper=False)
            return torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        elif self.solve_method == 'solve':
            return torch.linalg.solve(matrix, rhs)
        else:
            raise ValueError(
                f"Invalid solve_method '{self.solve_method}'. "
                "Choose from 'qr', 'cholesky', 'solve'."
            )

    @torch.no_grad()
    def _apply_updates(self, updates: Tensor) -> None:
        """Applies parameter updates directly to flat_params.

        Args:
            updates: The computed parameter updates.
        """
        self.flat_params.add_(-self.learning_rate * updates)

    @torch.no_grad()
    def _compute_jacobian(
        self,
        param_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor, Any]:
        """Computes the Jacobian of the residuals with respect to model parameters.

        Args:
            # inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            # targets: Target tensor of shape `(batch_size, target_dim, ...)`.
            param_indices: Indices of selected parameters if using a subset, else None.

        Returns:
            tuple: A tuple containing:
                - jacobian: The Jacobian matrix of shape `(num_residuals, num_params)`.
                - residuals: Residual vector of shape `(num_residuals, 1)`.
                - outputs: Model outputs of shape `(batch_size, target_dim, ...)`.
        """
        def compute_residuals(flat_params: Tensor) -> Tensor:
            x = self.forward(flat_params.view(next(iter(self._params)).shape))
            return torch.sqrt(self.loss_fn(x))

        # Compute outputs and residuals for the full batch
        outputs = self.forward(next(iter(self._params)))
        residuals = torch.sqrt(self.loss_fn(outputs))

        # Adjust flat_params to focus on the selected subset, if provided
        flat_params = self.flat_params

        jacobians: Tensor
        if self.use_vmap:
            # Compute per-sample Jacobian with vmap and jacrev
            jacobian_func = jacrev(compute_residuals)
            jacobians = vmap(
                jacobian_func, in_dims=0, randomness='different'
            )(
                flat_params,
            )
            jacobians = jacobians.squeeze(1)
        else:
            # Compute per-batch Jacobian with jacrev
            jacobian_func = jacrev(lambda p: compute_residuals(p))
            jacobians = jacobian_func(flat_params)  # type: ignore

        # Flatten batches and outputs into a matrix to solve least-squares problems.
        residuals = residuals.view(-1, 1)
        jacobians = jacobians.view(-1, flat_params.numel())

        return jacobians, residuals, outputs

    @torch.no_grad()
    def _sliced_gauss_newton_overdetermined(
        self,
        param_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor, Any]:
        """Gauss-Newton approximation for overdetermined systems using slicing.

        This method handles large overdetermined systems by dividing the input into
        smaller slices. For each slice, the Jacobian matrix is computed and used to
        incrementally build the full Gauss-Newton Hessian approximation and the
        right-hand side (RHS) vector.

        Args:
            # inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            # targets: Target tensor of shape `(batch_size, output_dim, ...)`.
            param_indices: Indices of selected parameters if using a subset, else None.

        Returns:
            tuple:
                - JJ: `(num_parameters, num_parameters) = J' * J`
                - rhs: `(num_parameters, 1) = J' * residuals`.
                - outputs: `(batch_size, output_dim, ...)`
        """
        assert self.max_batch_size is not None
        assert self._batch_size is not None

        batch_size = self._batch_size
        num_params = (
            param_indices.numel() if param_indices is not None else self._num_params
        )

        # Use one tensor from the inputs to obtain device and dtype.
        device = self.flat_params.device
        dtype = self.flat_params.dtype

        JJ = torch.zeros((num_params, num_params), dtype=dtype, device=device)
        rhs = torch.zeros((num_params, 1), dtype=dtype, device=device)

        outputs_slices: list[Any] = []

        for start in range(0, batch_size, self.max_batch_size):
            J_slice, residuals_slice, outputs_slice = self._compute_jacobian(
                param_indices
            )
            outputs_slices.append(outputs_slice)

            JJ += J_slice.t().matmul(J_slice)
            rhs += J_slice.t().matmul(residuals_slice)

        outputs = tree_cat(outputs_slices, dim=0)
        return JJ, rhs, outputs

    @torch.no_grad()
    def _sliced_gauss_newton_underdetermined(
        self,
        # inputs: Any,
        # targets: Any,
        param_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Any]:
        """Gauss-Newton approximation for underdetermined systems using slicing.

        This method handles large underdetermined systems by dividing the input into
        smaller slices. For each slice, it computes the local Jacobian and residuals,
        concatenating them into a full J and residuals.

        Args:
            # inputs: Input tensor `(batch_size, input_dim, ...)`.
            # targets: Target tensor `(batch_size, output_dim, ...)`.
            param_indices: Indices of selected parameters if using a subset, else None.

        Returns:
            tuple:
                - J: Full Jacobian `(num_residuals, num_parameters)`
                - JJ: `(num_residuals, num_residuals) = J * J'`
                - rhs: `(num_residuals, 1) = residuals`
                - outputs: `(batch_size, output_dim, ...)`
        """
        assert self.max_batch_size is not None
        assert self._batch_size is not None

        batch_size = self._batch_size

        J_slices: list[Tensor] = []
        residuals_slices: list[Tensor] = []
        outputs_slices: list[Any] = []

        for start in range(0, batch_size, self.max_batch_size):
            J_slice, residuals_slice, outputs_slice = self._compute_jacobian(
                param_indices
            )

            J_slices.append(J_slice)
            residuals_slices.append(residuals_slice)
            outputs_slices.append(outputs_slice)

        # Concatenate all slices to form the full J, residuals, and outputs
        J = torch.cat(J_slices, dim=0)
        residuals = torch.cat(residuals_slices, dim=0)
        outputs = tree_cat(outputs_slices, dim=0)

        # Compute JJ and rhs as in the non-sliced scenario for underdetermined case
        JJ = J @ J.t()  # JJ = J * J'
        rhs = residuals  # rhs = residuals
        return J, JJ, rhs, outputs

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Tensor],
    ) -> tuple[Any, Tensor, bool, dict[str, Any]]:
        """Performs a single training step.

        Args:
            # inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            # targets: Target tensor of shape `(batch_size, target_dim, ...)`.

        Returns:
            tuple: A tuple containing:
                - outputs: Model outputs for the given inputs.
                - loss: The computed loss value.
                - stop_training: Whether training should stop.
                - logs: Additional metadata (e.g., damping factor, attempts).
        """
        with torch.enable_grad():
            closure()

        if self._batch_size is None:
            # Initialize during the first train step
            outputs = self.forward(next(iter(self._params)))
            residuals = torch.sqrt(self.loss_fn(outputs))
            self._batch_size = residuals.shape[0]
            self._num_residuals = residuals.numel()

        assert self._batch_size
        assert self._num_residuals

        batch_size = self._batch_size
        num_residuals = self._num_residuals
        num_params = self._num_params

        param_indices = None
        if self.param_selection_strategy is not None:
            param_indices = self.param_selection_strategy.select_parameters()
            num_params = param_indices.numel()

        overdetermined = num_residuals >= num_params

        if self.max_batch_size is not None and self.max_batch_size < batch_size:
            # reduced memory sliced computation
            if overdetermined:
                JJ, rhs, outputs = self._sliced_gauss_newton_overdetermined(
                    param_indices
                )
                J = None
            else:
                J, JJ, rhs, outputs = self._sliced_gauss_newton_underdetermined(
                    param_indices
                )
        else:
            J, residuals, outputs = self._compute_jacobian(
                param_indices
            )
            if overdetermined:
                # overdetermined
                JJ = J.t() @ J  # JJ = J' * J
                rhs = J.t() @ residuals  # rhs = J' * residuals
            else:
                # underdetermined
                JJ = J @ J.t()  # JJ = J * J'
                rhs = residuals  # rhs = residuals

        # Normalize for numerical stability
        normalization_factor = 1.0 / batch_size
        JJ *= normalization_factor
        rhs *= normalization_factor

        # Compute the current loss value
        loss = self.loss_fn(outputs)

        stop_training = False
        attempt = 0
        self.damping_strategy.initialize_step(loss)

        for attempt in range(self.attempts_per_step):
            params_updated = False

            # Try to update the parameters
            try:
                # Apply damping to the Gauss-Newton Hessian approximation
                JJ_damped = self.damping_strategy.apply(JJ)

                # Compute the updates:
                # - Overdetermined: updates = (J' * J + damping)^-1 * J'*residuals
                # - Underdetermined: updates = J' * (J * J' + damping)^-1 * residuals
                updates = self._solve(JJ_damped, rhs)

                if not overdetermined:
                    assert J is not None
                    updates = J.t().matmul(updates)

                updates = updates.view(-1)

                if param_indices is not None:
                    full_updates = torch.zeros(
                        self._num_params,
                        device=updates.device,
                        dtype=updates.dtype,
                    )
                    full_updates[param_indices] = updates
                    updates = full_updates

                # Check if updates are finite
                if torch.all(torch.isfinite(updates)):
                    params_updated = True
                    self._apply_updates(updates)

            except Exception as e:
                print(f'[ERROR] levenberg_marquardt: An exception occurred: {e}')

            if params_updated:
                # Compute the new loss value
                new_outputs = self.forward(next(iter(self._params)))
                new_loss = self.loss_fn(new_outputs)

                if new_loss < loss:
                    # Accept the new model parameters and backup them
                    loss = new_loss
                    self.damping_strategy.on_successful_update(loss)
                    self.backup_parameters()
                    break

                # Restore the old parameters and try a new damping factor
                self.restore_parameters()

            # Adjust the damping factor for the next attempt
            self.damping_strategy.on_unsuccessful_update(loss)

            # Check if we should stop making more attempts and simply take the current update
            stop_attempts = self.damping_strategy.stop_attempts(loss)

            # Check if training should stop
            stop_training = self.damping_strategy.stop_training(loss)
            if stop_training or stop_attempts:
                break

        logs = {
            'damping': self.damping_strategy.get_current_damping(),
            'attempts': attempt,
        }

        return outputs, loss, stop_training, logs
