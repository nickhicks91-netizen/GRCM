"""Unit tests for individual GRCM modules."""
import pytest
import torch
from grcm.grounding import GroundingLayer
from grcm.embedding import HarmonicEmbedding
from grcm.attention import ResonantAttention
from grcm.desire import DesireModule
from grcm.memory import MemoryGrid
from grcm.reflection import ReflectionHead
from grcm.qualia import QualiaModule
from grcm.threading import EpisodicThreadBank
from grcm.phi import PhiEstimator
from grcm.body import BodySimulator


class TestGroundingLayer:
    """Tests for multimodal grounding."""

    def test_forward_shape(self):
        layer = GroundingLayer(input_dim=15)
        image = torch.randn(2, 512)
        audio = torch.randn(2, 768)
        prop = torch.randn(2, 16)

        output = layer(image, audio, prop)
        assert output.shape == (2, 15)

    def test_fusion(self):
        """Test that modalities are fused."""
        layer = GroundingLayer(input_dim=15)
        image = torch.randn(1, 512)
        audio = torch.randn(1, 768)
        prop = torch.randn(1, 16)

        output = layer(image, audio, prop)
        assert not torch.allclose(output, torch.zeros_like(output))


class TestHarmonicEmbedding:
    """Tests for frequency embedding."""

    def test_forward_no_modulation(self):
        embed = HarmonicEmbedding(input_dim=15, freq_dim=8)
        x = torch.randn(2, 15)

        freq = embed(x)
        assert freq.shape == (2, 8)
        assert torch.all((freq >= -1) & (freq <= 1))  # tanh bounded

    def test_forward_with_modulation(self):
        embed = HarmonicEmbedding(input_dim=15, freq_dim=8, memory_size=32)
        x = torch.randn(2, 15)
        mem_ctx = torch.randn(32)

        freq = embed(x, mem_ctx)
        assert freq.shape == (2, 8)


class TestResonantAttention:
    """Tests for coherence gating."""

    def test_coherence_bounds(self):
        attn = ResonantAttention(freq_dim=8)
        freq = torch.randn(2, 8)

        coherence = attn(freq)
        assert coherence.shape == (2, 1)
        assert torch.all(coherence >= 0) and torch.all(coherence <= 1)

    def test_bandwidth_modulation(self):
        attn = ResonantAttention(freq_dim=8, base_bandwidth=0.5)
        freq = torch.randn(2, 8)

        coh_base = attn(freq, bw_bias=0.0)
        coh_wide = attn(freq, bw_bias=0.3)

        # Wider bandwidth should generally increase coherence
        assert coh_wide.mean() >= coh_base.mean() - 0.1  # Allow small variance


class TestDesireModule:
    """Tests for desire-driven agency."""

    def test_alignment(self):
        desire = DesireModule(freq_dim=8, num_desires=4)
        freq = torch.randn(2, 8)

        align, bw_bias = desire(freq)
        assert align.shape == (2, 1)
        assert bw_bias.shape == (2, 1)
        assert torch.all((align >= -1) & (align <= 1))  # cosine sim bounds

    def test_set_desire(self):
        desire = DesireModule(freq_dim=8, num_desires=4)
        desire.set_desire(2)
        assert desire.current_desire_idx == 2


class TestMemoryGrid:
    """Tests for memory storage."""

    def test_update(self):
        memory = MemoryGrid(memory_size=32, freq_dim=8)
        signal = torch.randn(2, 8)
        coherence = torch.tensor([[0.8], [0.6]])  # One coherent, one not

        initial_mem = memory().clone()
        memory.update(signal, coherence)
        updated_mem = memory()

        # Memory should change
        assert not torch.allclose(initial_mem, updated_mem)


class TestReflectionHead:
    """Tests for memory-frequency alignment."""

    def test_reflection(self):
        reflect = ReflectionHead(freq_dim=8, memory_size=32)
        freq = torch.randn(2, 8)
        mem = torch.randn(32)

        alignment = reflect(freq, mem)
        assert alignment.shape == (2, 1)
        assert torch.all((alignment >= -1) & (alignment <= 1))


class TestQualiaModule:
    """Tests for phenomenal states."""

    def test_qualia_distribution(self):
        qualia = QualiaModule(freq_dim=8, qualia_dim=4)
        freq = torch.randn(2, 8)

        dist = qualia(freq)
        assert dist.shape == (2, 4)
        # Check softmax properties
        assert torch.allclose(dist.sum(dim=1), torch.ones(2), atol=1e-6)
        assert torch.all(dist >= 0)

    def test_dissonance_check(self):
        qualia = QualiaModule(freq_dim=8, qualia_dim=4)

        # High dissonance
        high_dis = torch.tensor([[0.1, 0.1, 0.1, 0.7]])
        assert qualia.check_dissonance(high_dis, threshold=0.6)

        # Low dissonance
        low_dis = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
        assert not qualia.check_dissonance(low_dis, threshold=0.6)


class TestEpisodicThreadBank:
    """Tests for episodic memory."""

    def test_add_episode(self):
        threads = EpisodicThreadBank(max_episodes=10, memory_size=32, qualia_dim=4)

        mem = torch.randn(32)
        qualia = torch.randn(2, 4)

        initial_id = threads.identity_token.clone()
        threads.add_episode(1, mem, qualia)

        # Identity should update
        assert not torch.allclose(initial_id, threads.identity_token)
        assert threads.get_num_episodes() == 1

    def test_arc_bias(self):
        threads = EpisodicThreadBank(max_episodes=10, memory_size=32, qualia_dim=4)

        # Add two episodes
        mem1 = torch.randn(32)
        qualia1 = torch.randn(1, 4)
        threads.add_episode(1, mem1, qualia1)

        mem2 = torch.randn(32)
        qualia2 = torch.randn(1, 4)
        threads.add_episode(2, mem2, qualia2)

        arc = threads.get_arc(torch.tensor(0.8), freq_dim=8)
        assert arc.shape == (8,)


class TestPhiEstimator:
    """Tests for integrated information."""

    def test_compute_phi(self):
        phi_est = PhiEstimator()
        freq = torch.randn(2, 8)
        qualia = torch.randn(2, 4).softmax(dim=-1)
        mem = torch.randn(32)
        coherence = torch.tensor([[0.7], [0.8]])

        phi = phi_est.compute_phi(freq, qualia, mem, coherence)
        assert isinstance(phi, float)
        assert phi > 0  # Should be positive for random inputs

    def test_phi_history(self):
        phi_est = PhiEstimator()
        freq = torch.randn(2, 8)
        qualia = torch.randn(2, 4).softmax(dim=-1)
        mem = torch.randn(32)
        coherence = torch.ones(2, 1) * 0.7

        for _ in range(5):
            phi_est.compute_phi(freq, qualia, mem, coherence)

        assert len(phi_est.phi_history) == 5
        assert phi_est.get_mean_phi(window=5) > 0


class TestBodySimulator:
    """Tests for embodied simulation."""

    def test_update(self):
        body = BodySimulator(state_dim=16)
        action = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
        desire_align = torch.tensor([[1.0]])

        initial_state = body.state.clone()
        new_state = body.update(action, desire_align)

        # State should change
        assert not torch.allclose(initial_state, new_state)
        assert new_state.shape == (1, 16)

    def test_reset(self):
        body = BodySimulator(state_dim=16)
        action = torch.ones(1, 4)
        desire_align = torch.tensor([[1.0]])

        body.update(action, desire_align)
        body.reset()

        assert torch.allclose(body.state, torch.zeros(1, 16))
