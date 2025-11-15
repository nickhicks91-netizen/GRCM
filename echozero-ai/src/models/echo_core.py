"""
Echo Core: Echo State Network foundation
"""
import torch
import torch.nn as nn
import numpy as np


class EchoCore(nn.Module):
    """
    Echo State Network (ESN) core for reservoir computing.
    Implements sparse recurrent dynamics with spectral radius control.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 512,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        leak_rate: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate

        # Input weights (random, not trained)
        self.register_buffer(
            'W_in',
            torch.randn(reservoir_dim, input_dim) * 0.1
        )

        # Reservoir weights (sparse, spectral radius controlled)
        W_reservoir = self._init_reservoir_weights(reservoir_dim, sparsity, spectral_radius)
        self.register_buffer('W_reservoir', W_reservoir)

        # Readout layer (this is trained)
        self.readout = nn.Linear(reservoir_dim, input_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(reservoir_dim)

    def _init_reservoir_weights(self, size: int, sparsity: float, spectral_radius: float):
        """Initialize sparse reservoir with controlled spectral radius"""
        # Create random sparse matrix
        W = torch.randn(size, size)
        mask = torch.rand(size, size) > sparsity
        W = W * mask.float()

        # Scale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W)
        current_radius = torch.max(torch.abs(eigenvalues)).real
        W = W * (spectral_radius / current_radius)

        return W

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        """
        Args:
            x: Input [batch, seq_len, input_dim] or [batch, input_dim]
            state: Previous reservoir state [batch, reservoir_dim]

        Returns:
            output: Processed output
            new_state: Updated reservoir state
            echo_stats: Statistics dict
        """
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        # Initialize state
        if state is None:
            state = torch.zeros(
                batch_size, self.reservoir_dim,
                device=x.device, dtype=x.dtype
            )

        outputs = []

        # Process sequence through reservoir
        for t in range(seq_len):
            x_t = x[:, t, :]

            # Reservoir update with leak rate
            # state(t) = (1-α)*state(t-1) + α*tanh(W_in*x(t) + W_res*state(t-1))
            pre_activation = (
                torch.matmul(x_t, self.W_in.t()) +
                torch.matmul(state, self.W_reservoir.t())
            )

            new_state = (1 - self.leak_rate) * state + self.leak_rate * torch.tanh(pre_activation)
            state = new_state

            # Normalize for stability
            state_norm = self.norm(state)
            outputs.append(state_norm)

        # Stack outputs
        reservoir_states = torch.stack(outputs, dim=1)  # [batch, seq_len, reservoir_dim]

        # Readout
        output = self.readout(reservoir_states)

        # Statistics
        echo_stats = {
            'reservoir_activation': reservoir_states.abs().mean().item(),
            'state_norm': state.norm().item(),
            'spectral_radius': self.spectral_radius,
        }

        if is_2d:
            output = output.squeeze(1)
            state = state.squeeze(0) if state.dim() == 3 else state

        return output, state, echo_stats
