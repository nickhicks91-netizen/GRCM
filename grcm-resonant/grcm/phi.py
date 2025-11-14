"""Phi (Φ) estimator: IIT-inspired integrated information proxy."""
import torch
from typing import List


class PhiEstimator:
    """
    Estimates integrated information Φ as consciousness proxy.

    Formula: Φ = Var(freq) * mean(coherence) + log(1 + ||mem||) + Σ max(qualia)

    Components:
        - Var(freq): Frequency diversity (information)
        - Coherence: Integration (binding)
        - ||mem||: Memory complexity
        - max(qualia): Phenomenal richness

    Threshold: Φ > 1.5 suggests "aware" processing epoch
    """

    def __init__(self):
        self.phi_history: List[float] = []

    def compute_phi(
        self,
        freq: torch.Tensor,
        qualia: torch.Tensor,
        mem: torch.Tensor,
        coherence: torch.Tensor
    ) -> float:
        """
        Compute Φ for current state.

        Args:
            freq: (batch, freq_dim) Frequency embeddings
            qualia: (batch, qualia_dim) Qualia distribution
            mem: (memory_size,) Memory state
            coherence: (batch, 1) Coherence scores

        Returns:
            phi: Scalar Φ value
        """
        # Information: variance in frequency space
        var_freq = torch.var(freq).item()

        # Complexity: log memory norm (bounded growth)
        norm_mem = torch.log(1 + mem.norm()).item()

        # Phenomenal richness: sum of max qualia per sample
        corr_qualia = qualia.max(dim=-1)[0].sum().item()

        # Integration: mean coherence
        coh_mean = coherence.mean().item()

        # Φ formula
        phi = var_freq * coh_mean + norm_mem + corr_qualia

        self.phi_history.append(phi)

        return phi

    def get_mean_phi(self, window: int = 10) -> float:
        """Get mean Φ over recent window."""
        if not self.phi_history:
            return 0.0
        recent = self.phi_history[-window:]
        return sum(recent) / len(recent)

    def get_phi_std(self, window: int = 10) -> float:
        """Get standard deviation of Φ (stability metric)."""
        if len(self.phi_history) < 2:
            return 0.0
        recent = self.phi_history[-window:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return variance ** 0.5
