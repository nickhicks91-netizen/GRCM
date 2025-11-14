"""Body simulator for proprioceptive feedback and embodied agency."""
import torch
import torch.nn.functional as F


class BodySimulator:
    """
    Simple physics-based body simulation for proprioceptive grounding.

    State: [position(8), velocity(8)] = 16D
    Dynamics: F = desire_align * action
              accel = F / mass
              vel += accel * dt
              pos += vel * dt

    Args:
        state_dim: Total state dimension (default 16: 8 pos + 8 vel)
        mass: Body mass for force→acceleration (default 1.0)
        dt: Time step (default 0.1)
    """

    def __init__(self, state_dim: int = 16, mass: float = 1.0, dt: float = 0.1):
        assert state_dim % 2 == 0, "state_dim must be even (pos + vel)"
        self.state_dim = state_dim
        self.state = torch.zeros(1, state_dim)
        self.mass = mass
        self.dt = dt
        self.half_dim = state_dim // 2

    def update(self, action: torch.Tensor, desire_align: torch.Tensor) -> torch.Tensor:
        """
        Update body state with action modulated by desire alignment.

        Args:
            action: (batch, action_dim) Control signals
            desire_align: (batch, 1) Desire alignment (gates action)

        Returns:
            new_state: (batch, state_dim) Updated proprioceptive state
        """
        # Pad action if smaller than half_dim
        if action.size(1) < self.half_dim:
            action = F.pad(action, (0, self.half_dim - action.size(1)))

        # Force = desire-gated action
        force = desire_align * action

        # Acceleration
        accel = force / self.mass

        # Clone for safety
        self.state = self.state.clone()

        # Velocity update: v += a * dt
        self.state[:, self.half_dim:] += accel[:, :self.half_dim] * self.dt

        # Position update: p += v * dt
        self.state[:, :self.half_dim] += self.state[:, self.half_dim:] * self.dt

        return self.state.clone()

    def reset(self) -> None:
        """Reset body to zero state."""
        self.state = torch.zeros(1, self.state_dim)

    def get_state(self) -> torch.Tensor:
        """Get current proprioceptive state."""
        return self.state.clone()
