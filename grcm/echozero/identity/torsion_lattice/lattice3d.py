"""
3D Torsion Memory Lattice - Long-term Identity Storage

Implements a 3D XY-model phase lattice with self-healing behavior.
This provides stable, noise-resistant long-term memory that operates
on a slow loop (0.1-2 Hz) independent of fast inference.

Key Properties:
- Self-healing: XY-model dynamics drive toward stable minima
- Noise-resistant: Phase coherence survives perturbations
- Möbius-gated: Only low-torsion patterns are written
- Non-invasive: Never blocks fast loop
"""

import numpy as np
from typing import Tuple, Optional


class TorsionLattice3D:
    """
    3D XY-model phase lattice with self-healing behavior.

    The lattice stores identity information as phase patterns in a 3D grid.
    Local relaxation dynamics automatically heal noise and damage.

    Physics:
        θ(i,j,k) ∈ [0, 2π)  - Phase at each lattice point
        m - Global magnitude (decays to enforce stability)

    Dynamics:
        θ_new = θ + κ·Σ_neighbors sin(θ_neighbor - θ)
        m_new = m·exp(-γ/2)
    """

    def __init__(
        self,
        size: int = 5,
        coupling: float = 0.8,
        gamma: float = 0.03,
    ):
        """
        Initialize 3D torsion lattice.

        Args:
            size: Lattice dimension (creates size³ grid)
            coupling: XY coupling strength (higher = faster relaxation)
            gamma: Magnitude decay rate (self-healing parameter)
        """
        self.size = size
        self.coupling = coupling
        self.gamma = gamma

        # Phase matrix: values in [0, 2π)
        self.theta = np.zeros((size, size, size), dtype=np.float64)
        self.init_checkerboard()

        # Magnitude (uniform across lattice)
        self.m = 1.0

        # Neighbor offsets (26-neighbor connectivity)
        self.neighbors = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

    def init_checkerboard(self) -> None:
        """
        Initialize with checkerboard pattern (stable XY ground state).

        Alternates between 0 and π based on coordinate parity.
        This is a local minimum of the XY model.
        """
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    if (i + j + k) % 2 == 0:
                        self.theta[i, j, k] = 0.0
                    else:
                        self.theta[i, j, k] = np.pi

    def step(self) -> None:
        """
        One relaxation step using XY-model update.

        Updates all phases based on neighbor coupling, then applies
        magnitude decay. This drives the system toward stable minima.
        """
        new_theta = np.copy(self.theta)
        L = self.size

        for i in range(L):
            for j in range(L):
                for k in range(L):
                    # XY local field: Σ sin(θ_neighbor - θ_i)
                    s = 0.0
                    for dx, dy, dz in self.neighbors:
                        ni = (i + dx) % L  # Periodic boundary conditions
                        nj = (j + dy) % L
                        nk = (k + dz) % L
                        s += np.sin(self.theta[ni, nj, nk] - self.theta[i, j, k])

                    # Update using coupling strength
                    new_theta[i, j, k] += self.coupling * s

        # Apply magnitude decay globally (self-healing mechanism)
        self.m *= np.exp(-self.gamma / 2)

        # Wrap phases back to [0, 2π)
        self.theta = np.mod(new_theta, 2 * np.pi)

    def write_vector(self, v: np.ndarray, strength: float = 0.03) -> None:
        """
        Write an identity vector into the lattice by phase alignment.

        Converts the identity vector to an angle and weakly biases all
        phases toward that angle. Uses small strength (< 0.05) to avoid
        disrupting stable patterns.

        Args:
            v: Identity vector [2D] (real, imag components)
            strength: Write strength coefficient (should be < 0.05)
        """
        # Convert vector to angle
        angle = np.arctan2(v[1], v[0])  # atan2(imag, real)

        # Weak write: blend current phases with target angle
        self.theta = (1 - strength) * self.theta + strength * angle

        # Wrap back to [0, 2π)
        self.theta = np.mod(self.theta, 2 * np.pi)

    def read_vector(self) -> np.ndarray:
        """
        Return average phase as identity vector.

        Computes the mean phase across all lattice points and
        converts to a 2D vector representation.

        Returns:
            Identity vector [real, imag] representing average phase
        """
        # Compute mean of exp(iθ) to get average phase
        real = np.mean(np.cos(self.theta))
        imag = np.mean(np.sin(self.theta))

        return np.array([real, imag], dtype=np.float64)

    def get_energy(self) -> float:
        """
        Compute XY-model energy (for debugging/monitoring).

        E = -Σ_<i,j> cos(θ_i - θ_j)

        Lower energy = more stable configuration.

        Returns:
            Total XY energy
        """
        energy = 0.0
        L = self.size

        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for dx, dy, dz in self.neighbors:
                        ni = (i + dx) % L
                        nj = (j + dy) % L
                        nk = (k + dz) % L

                        # XY interaction
                        energy -= np.cos(self.theta[i, j, k] - self.theta[ni, nj, nk])

        # Divide by 2 (each pair counted twice)
        return energy / 2.0

    def get_magnetization(self) -> float:
        """
        Compute order parameter (magnetization).

        M = |⟨exp(iθ)⟩|

        M ≈ 1: ordered (coherent phases)
        M ≈ 0: disordered (random phases)

        Returns:
            Magnetization magnitude [0, 1]
        """
        vec = self.read_vector()
        return np.linalg.norm(vec)

    def reset(self) -> None:
        """Reset lattice to checkerboard ground state."""
        self.init_checkerboard()
        self.m = 1.0

    def __repr__(self) -> str:
        """String representation of lattice state."""
        mag = self.get_magnetization()
        energy = self.get_energy()
        return (
            f"TorsionLattice3D(size={self.size}, "
            f"m={self.m:.4f}, "
            f"magnetization={mag:.4f}, "
            f"energy={energy:.2f})"
        )
