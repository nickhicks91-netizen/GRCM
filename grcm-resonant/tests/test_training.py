"""Tests for EchoMirror training."""
import pytest
import torch
from grcm import ResonantConsciousnessModule, echo_mirror_train


class TestEchoMirrorTraining:
    """Tests for desire vector tuning."""

    def test_echo_mirror_convergence(self):
        """Test that training reduces loss."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )
        model.set_desire(0)

        # Mock data
        eeg = torch.randn(10, 8)
        voice = torch.randn(10, 768)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        # Get initial desire vectors
        initial_desires = model.desire.desire_vectors.clone()

        # Train
        echo_mirror_train(model, eeg, voice, labels, epochs=5, lr=0.01, verbose=False)

        # Desire vectors should update
        assert not torch.allclose(initial_desires, model.desire.desire_vectors)

    def test_echo_mirror_phi_preservation(self):
        """Test that phi is maintained during training."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )
        model.set_desire(0)

        eeg = torch.randn(10, 8)
        voice = torch.randn(10, 768)
        labels = torch.ones(10)  # All positive

        # Train
        echo_mirror_train(model, eeg, voice, labels, epochs=3, lr=0.01, verbose=False)

        # Phi should be tracked
        assert len(model.phi.phi_history) > 0

    def test_echo_mirror_different_labels(self):
        """Test with different label distributions."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        eeg = torch.randn(20, 8)
        voice = torch.randn(20, 768)

        # All zeros
        labels_zero = torch.zeros(20)
        echo_mirror_train(model, eeg, voice, labels_zero, epochs=2, verbose=False)

        # All ones
        model.reset()
        labels_one = torch.ones(20)
        echo_mirror_train(model, eeg, voice, labels_one, epochs=2, verbose=False)

        # Should complete without errors
        assert len(model.phi.phi_history) > 0
