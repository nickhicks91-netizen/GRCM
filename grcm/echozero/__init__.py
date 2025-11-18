"""
EchoZero: Complex-valued resonant dynamics module.

This module implements coupled nonlinear oscillator dynamics with:
- Local dynamics (-α + iω)ψ
- Nearest-neighbor coupling
- Nonlinear damping -β|ψ|²ψ
- Want modulation γψ
- External drive I(t)
- Hub constraint -λΣψ
"""

from .dynamics import (
    echozero_dynamics,
    EchoZeroSystem,
    compute_coherence,
    compute_want_modulation,
)
from .lattice import LatticeBuilder, build_ring_lattice
from .coupling import CouplingMatrix, build_coupling_matrix
from .ode_solver import integrate_echozero, RK4Solver

__all__ = [
    "echozero_dynamics",
    "EchoZeroSystem",
    "compute_coherence",
    "compute_want_modulation",
    "LatticeBuilder",
    "build_ring_lattice",
    "CouplingMatrix",
    "build_coupling_matrix",
    "integrate_echozero",
    "RK4Solver",
]
