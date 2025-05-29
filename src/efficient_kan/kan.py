import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def _b_spline_basis_recursion(self, x: torch.Tensor, p: int, knots: torch.Tensor) -> torch.Tensor:
        """
        Computes B-spline basis functions of degree p using the Cox-de Boor recursion.
        This is an internal helper function.

        Args:
            x (torch.Tensor): Input tensor after unsqueezing, typically (batch_size, in_features, 1).
            p (int): Degree of the B-spline to compute.
            knots (torch.Tensor): Knot vector of shape (in_features, num_knots).

        Returns:
            torch.Tensor: B-spline bases of degree p.
                          Shape: (batch_size, in_features, num_basis_functions).
                          num_basis_functions = num_knots - (p + 1).
        """
        # B-splines of negative degree are zero. This handles recursive calls for derivatives
        # that reduce the degree below zero.
        if p < 0:
            # The number of "basis functions" for formal consistency in recursion.
            num_bases_formal = knots.shape[1] - (p + 1) # num_knots - p - 1
            return torch.zeros((x.size(0), self.in_features, num_bases_formal),
                               device=x.device, dtype=x.dtype)

        # Base case: degree 0 B-splines
        # B_{i,0}(x) = 1 if knots_i <= x < knots_{i+1}, else 0
        # Output shape for degree 0: (batch_size, in_features, num_knots - 1)
        bases = ((x >= knots[:, :-1].unsqueeze(0)) & (x < knots[:, 1:].unsqueeze(0))).to(x.dtype)

        # Cox-de Boor recursion for degree p > 0
        # k_iter is the current degree being computed, from 1 to p
        for k_iter in range(1, p + 1):
            # `bases` at the start of this iteration are for degree (k_iter - 1)
            # Shape of `bases` (last dim): num_knots - (k_iter - 1 + 1) = num_knots - k_iter

            # Denominators for the two terms in the recursion
            # Term 1: (x - t_i) / (t_{i+k_iter} - t_i) * B_{i, k_iter-1}(x)
            den1 = knots[:, k_iter:-1] - knots[:, :-(k_iter + 1)]
            den1 = torch.where(den1 == 0, torch.full_like(den1, 1e-8), den1) # Avoid division by zero

            # Term 2: (t_{i+k_iter+1} - x) / (t_{i+k_iter+1} - t_{i+1}) * B_{i+1, k_iter-1}(x)
            den2 = knots[:, (k_iter + 1):] - knots[:, 1:-k_iter]
            den2 = torch.where(den2 == 0, torch.full_like(den2, 1e-8), den2) # Avoid division by zero

            # Numerators and combining terms
            term1_factor = (x - knots[:, :-(k_iter + 1)].unsqueeze(0)) / den1.unsqueeze(0)
            term1 = term1_factor * bases[:, :, :-1] # Uses B_{i, k_iter-1}

            term2_factor = (knots[:, (k_iter + 1):].unsqueeze(0) - x) / den2.unsqueeze(0)
            term2 = term2_factor * bases[:, :, 1:] # Uses B_{i+1, k_iter-1}
            
            bases = term1 + term2
            # Shape of `bases` after this iteration (last dim): num_knots - (k_iter + 1)
        return bases
    
    def _b_spline_derivative_recursion(self, x: torch.Tensor, m_remaining: int, current_p: int, knots: torch.Tensor) -> torch.Tensor:
        """
        Recursively computes the m_remaining-th derivative of B-splines that originally had degree `current_p`
        at this stage of recursion.

        Args:
            x (torch.Tensor): Input tensor after unsqueezing, typically (batch_size, in_features, 1).
            m_remaining (int): Remaining derivative order to compute.
            current_p (int): The degree of the B-splines for which the m_remaining-th derivative is sought.
            knots (torch.Tensor): Original knot vector, shape (in_features, num_knots).

        Returns:
            torch.Tensor: m_remaining-th derivatives of B-splines of degree `current_p`.
                          Shape: (batch_size, in_features, num_knots - (current_p + 1)).
        """
        num_knots = knots.shape[1]

        # Base case for derivative recursion: if no more derivatives needed (m_remaining = 0),
        # evaluate the B-spline of degree current_p.
        if m_remaining == 0:
            return self._b_spline_basis_recursion(x, current_p, knots)

        # If current_p < 0, it implies we've differentiated a degree 0 spline. Its derivative is 0.
        # (The _b_spline_basis_recursion also handles p < 0 by returning zeros).
        # More generally, the derivative of B_{i,p} is p * (...). So if p=0, derivative is 0.
        if current_p < 0: # Should be captured by _b_spline_basis_recursion if m_remaining=0
                          # Or by current_p=0 in the multiplication step below.
                          # For safety, if current_p becomes negative and m_remaining > 0.
            num_basis_functions = num_knots - (current_p + 1)
            return torch.zeros((x.size(0), self.in_features, num_basis_functions),
                               device=x.device, dtype=x.dtype)


        # Recursive step for derivative:
        # D^m B_{i,p} = p * ( D^{m-1}B_{i,p-1}/(t_{i+p}-t_i) - D^{m-1}B_{i+1,p-1}/(t_{i+p+1}-t_{i+1}) )
        # We need (m_remaining-1)-th derivatives of splines of degree (current_p - 1).
        
        bases_lower_deg_deriv = self._b_spline_derivative_recursion(x, m_remaining - 1, current_p - 1, knots)
        # Shape of bases_lower_deg_deriv (last dim): num_knots - (current_p - 1 + 1) = num_knots - current_p

        # Denominators for the derivative formula.
        # For D B_{i,current_p}, the first term involves B_{i,current_p-1} / (t_{i+current_p} - t_i)
        den1 = knots[:, current_p : num_knots - 1] - knots[:, : num_knots - (current_p + 1)]
        den1 = torch.where(den1 == 0, torch.full_like(den1, 1e-8), den1)

        den2 = knots[:, current_p + 1 :] - knots[:, 1 : num_knots - current_p]
        den2 = torch.where(den2 == 0, torch.full_like(den2, 1e-8), den2)

        # bases_lower_deg_deriv has shape (..., num_knots - current_p)
        # bases_lower_deg_deriv[:, :, :-1] corresponds to D^{m-1}B_{i, current_p-1}
        # bases_lower_deg_deriv[:, :, 1:] corresponds to D^{m-1}B_{i+1, current_p-1}
        # These slices result in (num_knots - current_p - 1) elements, which is the
        # number of basis functions for degree current_p.
        term1 = bases_lower_deg_deriv[:, :, :-1] / den1.unsqueeze(0)
        term2 = bases_lower_deg_deriv[:, :, 1:] / den2.unsqueeze(0)
        
        # The factor is `current_p` (degree of the splines B_{i,current_p} before this differentiation step).
        # If current_p is 0, this factor makes the derivative 0, which is correct for D B_{i,0}.
        derivatives = current_p * (term1 - term2)
        
        return derivatives

    def b_splines(self, x: torch.Tensor, m: int = 0):
        """
        Compute the B-spline bases or their m-th derivatives.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            m (int): Order of the derivative. m=0 means B-spline values.
                     m must be non-negative.

        Returns:
            torch.Tensor: B-spline bases/derivatives tensor.
                          Shape (batch_size, in_features, num_out_bases).
                          num_out_bases = self.grid_size + self.spline_order.
                          This is the number of basis functions for B-splines of degree self.spline_order.
        """
        if not (x.dim() == 2 and x.size(1) == self.in_features):
             raise ValueError(
                 f"Input x must have shape (batch_size, in_features). Got {x.shape}, expected ({x.size(0)}, {self.in_features})"
            )
        if not isinstance(m, int) or m < 0:
            raise ValueError("Derivative order m must be a non-negative integer.")

        knots = self.grid
        initial_degree = self.spline_order # This is p, the original degree of the splines

        # If derivative order m is greater than spline degree p, the result is zero.
        if m > initial_degree:
            num_out_bases = self.grid_size + self.spline_order
            return torch.zeros((x.size(0), self.in_features, num_out_bases),
                               device=x.device, dtype=x.dtype)

        # Unsqueeze x for broadcasting within helper functions
        x_unsqueezed = x.unsqueeze(-1) # Shape: (batch_size, in_features, 1)
        
        # Call the recursive helper.
        # `initial_degree` is the degree of B-splines for which derivatives are sought.
        # `m` is the total order of derivative needed.
        result_bases = self._b_spline_derivative_recursion(x_unsqueezed, m, initial_degree, knots)

        # Assert final shape.
        # The number of basis functions for degree `initial_degree` (self.spline_order) is
        # num_knots - (initial_degree + 1).
        # Given num_knots = self.grid_size + 2 * initial_degree + 1,
        # num_out_bases = (self.grid_size + 2 * initial_degree + 1) - (initial_degree + 1)
        #               = self.grid_size + initial_degree
        # This matches `self.grid_size + self.spline_order` from the original docstring.
        expected_num_out_bases = self.grid_size + self.spline_order
        
        # Validate output shape
        if not (result_bases.ndim == 3 and \
                result_bases.size(0) == x.size(0) and \
                result_bases.size(1) == self.in_features and \
                result_bases.size(2) == expected_num_out_bases):
             raise AssertionError(
                 f"Output shape mismatch. Expected {(x.size(0), self.in_features, expected_num_out_bases)}, "
                 f"got {result_bases.shape}. (m={m}, initial_degree={initial_degree})"
            )
        
        return result_bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def second_derivative_regularization(self):
        """
        Compute the L2 norm of second derivatives of spline functions at fixed points.
        Uses a fixed grid of points evenly spaced in the domain [-1, 1].
        
        Returns:
            torch.Tensor: Mean L2 norm of second derivatives
        """
        # Create fixed evaluation points (100 points evenly spaced in [-1, 1])
        n_points = 100
        eval_points = torch.linspace(-1, 1, n_points, device=self.base_weight.device)
        # Expand for each input feature
        eval_points = eval_points.unsqueeze(1).expand(-1, self.in_features)  # (n_points, in_features)
        
        # Get second derivatives of B-splines at these points
        d2_bases = self.b_splines(eval_points, m=2)  # (n_points, in_features, grid_size + spline_order)
        
        # Compute spline values using second derivatives
        d2_output = F.linear(
            d2_bases.view(n_points, -1),
            self.scaled_spline_weight.view(self.out_features, -1)
        )  # (n_points, out_features)
        
        # Compute mean L2 norm
        return torch.mean(d2_output.pow(2))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0, regularize_smoothness=0.0):
        """
        Compute the regularization loss.

        This includes three terms:
        1. L1 regularization on spline weights (activation)
        2. Entropy regularization on spline weights
        3. L2 regularization on second derivatives (smoothness)

        Args:
            regularize_activation (float): Weight for L1 regularization
            regularize_entropy (float): Weight for entropy regularization
            regularize_smoothness (float): Weight for second derivative regularization
        """
        # Original regularization terms
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        
        # Second derivative regularization
        regularization_loss_smoothness = (
            self.second_derivative_regularization() if regularize_smoothness > 0
            else torch.tensor(0.0, device=self.base_weight.device)
        )
        
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
            + regularize_smoothness * regularization_loss_smoothness
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        random_seed=None
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Set random seed if provided
        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0, regularize_smoothness=0.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy, regularize_smoothness)
            for layer in self.layers
        )
