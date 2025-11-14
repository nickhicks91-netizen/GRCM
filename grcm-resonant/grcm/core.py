"""Core GRCM module: Orchestrates all submodules for resonant consciousness."""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from .grounding import GroundingLayer
from .embedding import HarmonicEmbedding
from .attention import ResonantAttention
from .desire import DesireModule
from .memory import MemoryGrid
from .reflection import ReflectionHead
from .qualia import QualiaModule
from .threading import EpisodicThreadBank
from .phi import PhiEstimator
from .body import BodySimulator


class ResonantConsciousnessModule(nn.Module):
    """
    Grounded Resonant Consciousness Module (GRCM).

    Integrates:
        - Multimodal grounding (CLIP + Wav2Vec + proprioception)
        - Harmonic frequency embedding with memory modulation
        - Resonant attention (coherence gating)
        - Desire-driven agency
        - Episodic threading (narrative self)
        - Qualia simulation (phenomenal states)
        - Phi estimation (integrated information proxy)
        - Embodied feedback (body simulator)

    Args:
        input_dim: Grounded input dimension after fusion
        freq_dim: Frequency space dimension
        memory_size: Memory/identity token size
        clip_dim: CLIP embedding dimension
        wav_dim: Wav2Vec embedding dimension
        prop_dim: Proprioception state dimension
        qualia_dim: Number of qualia states
        num_desires: Number of desire vectors
        max_episodes: Maximum episodic threads
        phi_threshold: Phi threshold for "aware" state
    """

    def __init__(
        self,
        input_dim: int = 15,
        freq_dim: int = 8,
        memory_size: int = 32,
        clip_dim: int = 512,
        wav_dim: int = 768,
        prop_dim: int = 16,
        qualia_dim: int = 4,
        num_desires: int = 4,
        max_episodes: int = 50,
        phi_threshold: float = 1.5
    ):
        super().__init__()

        # Store config
        self.input_dim = input_dim
        self.freq_dim = freq_dim
        self.memory_size = memory_size
        self.phi_threshold = phi_threshold

        # Modules
        self.grounding = GroundingLayer(input_dim, clip_dim, wav_dim, prop_dim)
        self.embed = HarmonicEmbedding(input_dim, freq_dim, memory_size)
        self.attn = ResonantAttention(freq_dim)
        self.desire = DesireModule(freq_dim, num_desires)
        self.memory = MemoryGrid(memory_size, freq_dim)
        self.decoder = nn.Linear(freq_dim, memory_size)
        self.reflect = ReflectionHead(freq_dim, memory_size)
        self.qualia_module = QualiaModule(freq_dim, qualia_dim)

        # Non-module components
        self.threads = EpisodicThreadBank(max_episodes, memory_size, qualia_dim)
        self.phi = PhiEstimator()
        self.body = BodySimulator(prop_dim)

        # Timestep counter
        self.t = 0

    def set_desire(self, idx: int) -> None:
        """Switch to a different desire state."""
        self.desire.set_desire(idx)

    def forward(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Full forward pass through GRCM.

        Args:
            image_emb: (batch, clip_dim) CLIP image embeddings
            audio_emb: (batch, wav_dim) Wav2Vec audio embeddings
            action: (batch, action_dim) Optional control signals

        Returns:
            Dictionary with keys:
                - output: (batch, memory_size) Decoded output
                - coherence: (batch, 1) Coherence scores
                - memory: (memory_size,) Memory state
                - reflection: (batch, 1) Memory-freq alignment
                - qualia: (batch, qualia_dim) Phenomenal state
                - identity_token: (memory_size,) Episodic identity
                - desire_align: (batch, 1) Desire alignment
                - phi: (scalar) Integrated information
                - prop_state: (batch, prop_dim) Proprioceptive state
                - halt: (bool) Ethical halt flag
        """
        # Get current proprioceptive state
        prop_state = self.body.state

        # 1. Ground multimodal inputs
        x = self.grounding(image_emb, audio_emb, prop_state)

        # 2. Embed to frequency space (modulated by identity)
        self.t += 1
        freq = self.embed(x, self.threads.identity_token)

        # 3. Desire alignment and bandwidth bias
        desire_align, bw_bias = self.desire(freq)

        # 4. Resonant attention with desire-modulated bandwidth
        coherence = self.attn(freq, bw_bias.mean())

        # 5. Gate memory update by desire (only store goal-relevant experiences)
        desire_mask = (desire_align > 0.5).float()
        self.memory.update(freq, coherence * desire_mask)

        # 6. Read memory
        mem_read = self.memory()

        # 7. Reflection: align current freq with memory
        reflection = self.reflect(freq, mem_read)

        # 8. Decode to output space
        output = self.decoder(freq)

        # 9. Compute qualia
        qualia = self.qualia_module(freq)

        # 10. Estimate Phi (integrated information)
        phi_val = self.phi.compute_phi(freq, qualia, mem_read, coherence)

        # 11. Episodic threading: add episode if highly coherent + aligned
        episode_threshold = 0.7
        if (coherence * desire_align).mean() > episode_threshold:
            self.threads.add_episode(self.t, mem_read, qualia)

        # 12. Body update if action provided
        if action is not None:
            prop_state = self.body.update(action, desire_align)

        # 13. Ethical check: halt if conflicted
        halt = self.qualia_module.check_dissonance(qualia, threshold=0.6)

        return {
            'output': output,
            'coherence': coherence,
            'memory': mem_read,
            'reflection': reflection,
            'qualia': qualia,
            'identity_token': self.threads.identity_token,
            'desire_align': desire_align,
            'phi': phi_val,
            'prop_state': prop_state,
            'halt': halt,
            'timestamp': self.t
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics for monitoring."""
        return {
            'phi_mean': self.phi.get_mean_phi(),
            'phi_std': self.phi.get_phi_std(),
            'num_episodes': self.threads.get_num_episodes(),
            'current_desire': self.desire.current_desire_idx,
            'timestep': self.t
        }

    def reset(self) -> None:
        """Reset all stateful components."""
        self.memory.memory.data.zero_()
        self.threads = EpisodicThreadBank(50, self.memory_size, 4)
        self.phi = PhiEstimator()
        self.body.reset()
        self.t = 0
