"""Grounding layer for multimodal sensory fusion (CLIP + Wav2Vec + proprioception)."""
import torch
import torch.nn as nn
from typing import Optional


class GroundingLayer(nn.Module):
    """
    Fuses multimodal embeddings: vision (CLIP), audio (Wav2Vec), proprioception.
    Uses cross-attention for modality harmony.

    Args:
        input_dim: Total dimension after fusion
        clip_dim: CLIP image embedding dimension (default 512)
        wav_dim: Wav2Vec audio embedding dimension (default 768)
        prop_dim: Proprioception state dimension (default 16)
        num_heads: Number of attention heads (default 3)
    """

    def __init__(
        self,
        input_dim: int,
        clip_dim: int = 512,
        wav_dim: int = 768,
        prop_dim: int = 16,
        num_heads: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim

        # Ensure input_dim is divisible by 3 for equal splits
        third = input_dim // 3
        self.clip_proj = nn.Linear(clip_dim, third)
        self.wav_proj = nn.Linear(wav_dim, third)
        self.prop_proj = nn.Linear(prop_dim, input_dim - 2 * third)  # Remainder

        self.cross_attn = nn.MultiheadAttention(input_dim, num_heads)

    def forward(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        prop_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_emb: (batch, clip_dim) CLIP embeddings
            audio_emb: (batch, wav_dim) Wav2Vec embeddings
            prop_state: (batch, prop_dim) Proprioception state

        Returns:
            grounded: (batch, input_dim) Fused representation
        """
        clip = self.clip_proj(image_emb)
        wav = self.wav_proj(audio_emb)
        prop = self.prop_proj(prop_state)

        # Concatenate modalities
        grounded = torch.cat([clip, wav, prop], dim=-1)

        # Self-attention for cross-modal binding
        grounded = grounded.unsqueeze(0)  # (1, batch, input_dim) for MHA
        grounded, _ = self.cross_attn(grounded, grounded, grounded)
        grounded = grounded.squeeze(0)  # Back to (batch, input_dim)

        return grounded
