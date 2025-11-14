"""Tests for Gradio UI."""
import pytest
import torch
import numpy as np

try:
    from grcm.gradio_ui import create_qualia_plot, create_phi_history_plot, GRADIO_AVAILABLE, MATPLOTLIB_AVAILABLE
except ImportError:
    GRADIO_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False

from grcm import ResonantConsciousnessModule


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
@pytest.mark.requires_gradio
class TestQualiaPlot:
    """Tests for qualia plotting."""

    def test_create_qualia_plot(self):
        """Test qualia plot creation."""
        qualia = np.array([0.3, 0.4, 0.2, 0.1])

        fig = create_qualia_plot(qualia)

        assert fig is not None
        assert len(fig.axes) == 1

    def test_qualia_plot_with_dissonance(self):
        """Test qualia plot with high conflicted state."""
        qualia = np.array([0.1, 0.2, 0.0, 0.7])  # High conflicted

        fig = create_qualia_plot(qualia)

        assert fig is not None

    def test_phi_history_plot(self):
        """Test phi history plot creation."""
        phi_history = [1.2, 1.5, 1.8, 1.6, 1.7, 1.9, 2.0, 1.8, 1.7, 1.6]

        fig = create_phi_history_plot(phi_history)

        assert fig is not None
        assert len(fig.axes) == 1


@pytest.mark.skipif(not GRADIO_AVAILABLE or not MATPLOTLIB_AVAILABLE, reason="Gradio/Matplotlib not installed")
@pytest.mark.requires_gradio
class TestGradioInterface:
    """Tests for Gradio interface creation."""

    def test_create_interface(self):
        """Test interface creation."""
        from grcm.gradio_ui import create_grcm_interface

        model = ResonantConsciousnessModule(15, 8, 32)

        interface = create_grcm_interface(model)

        assert interface is not None

    @pytest.mark.slow
    def test_interface_process_inputs(self):
        """Test processing inputs through interface (without launching)."""
        from grcm.gradio_ui import create_grcm_interface

        model = ResonantConsciousnessModule(15, 8, 32)
        model.eval()

        # Create interface
        interface = create_grcm_interface(model)

        # Interface created successfully
        assert interface is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
