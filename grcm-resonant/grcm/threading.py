"""Episodic threading for narrative self and identity continuity."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Tuple, Optional


class EpisodicThreadBank:
    """
    Maintains episodic memory threads and evolves identity token.

    Each episode = (timestamp, memory_state, qualia_avg)
    Identity token updated via: GRU(mem + qualia_proj, old_id)
    Arc bias = cos_sim(recent_qualia, historical_mean) * coherence

    Args:
        max_episodes: Maximum episodes to store (default 50)
        memory_size: Dimension of memory state (default 32)
        qualia_dim: Dimension of qualia (default 4)
    """

    def __init__(
        self,
        max_episodes: int = 50,
        memory_size: int = 32,
        qualia_dim: int = 4
    ):
        self.threads: deque = deque(maxlen=max_episodes)
        self.memory_size = memory_size
        self.qualia_dim = qualia_dim

        # Identity token: persistent self-representation
        self.identity_token = torch.zeros(memory_size)
        self.gru_id = nn.GRUCell(memory_size, memory_size)

    def add_episode(
        self,
        timestamp: int,
        mem: torch.Tensor,
        qualia: torch.Tensor
    ) -> None:
        """
        Add new episode and update identity token.

        Args:
            timestamp: Step counter
            mem: (memory_size,) Memory state
            qualia: (batch, qualia_dim) Qualia distribution
        """
        # Average qualia over batch
        qualia_avg = qualia.mean(0)

        # Pad qualia to memory size
        qualia_proj = F.pad(qualia_avg, (0, self.memory_size - self.qualia_dim))

        # Combine memory and qualia for identity update
        combined = mem + qualia_proj

        # Store episode
        self.threads.append((timestamp, mem.clone(), qualia_avg.clone()))

        # Update identity via GRU
        new_id = self.gru_id(
            combined.unsqueeze(0),
            self.identity_token.unsqueeze(0)
        ).squeeze(0)

        self.identity_token.copy_(new_id)

    def get_arc(self, current_coherence: torch.Tensor, freq_dim: int) -> torch.Tensor:
        """
        Compute narrative arc bias from qualia trajectory.

        Arc formula: cos_sim(recent_qualia, historical_mean) * coherence * 0.1
        Positive arc → pursuing coherent goals; negative → deviation

        Args:
            current_coherence: (1,) or scalar coherence value
            freq_dim: Dimension for output bias

        Returns:
            arc_bias: (freq_dim,) Bandwidth bias from narrative continuity
        """
        if len(self.threads) < 2:
            return torch.zeros(freq_dim)

        # Most recent qualia
        recent = self.threads[-1][2]

        # Historical qualia (all but most recent)
        historical_qualia = [th[2] for th in list(self.threads)[:-1]]

        if len(historical_qualia) == 0:
            historical = torch.zeros_like(recent)
        else:
            historical = torch.stack(historical_qualia).mean(0)

        # Cosine similarity: consistency in qualia trajectory
        arc_delta = F.cosine_similarity(
            recent.unsqueeze(0),
            historical.unsqueeze(0),
            dim=-1
        )

        # Scale by current coherence
        if hasattr(current_coherence, 'item'):
            coh_scalar = current_coherence.item()
        else:
            coh_scalar = float(current_coherence)

        arc_delta = arc_delta * coh_scalar

        # Broadcast to freq_dim with scaling
        arc_bias = arc_delta * torch.ones(freq_dim) * 0.1

        return arc_bias

    def get_num_episodes(self) -> int:
        """Return number of stored episodes."""
        return len(self.threads)
