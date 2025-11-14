"""Tests for MLflow integration."""
import pytest
import torch
import tempfile
import shutil
from pathlib import Path

try:
    from grcm.mlflow_logger import GRCMMLflowLogger, create_mlflow_experiment, MLFLOW_AVAILABLE
except ImportError:
    MLFLOW_AVAILABLE = False

from grcm import ResonantConsciousnessModule


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
@pytest.mark.requires_mlflow
class TestMLflowIntegration:
    """Tests for MLflow logging."""

    @pytest.fixture
    def temp_mlruns(self):
        """Create temporary MLflow tracking directory."""
        temp_dir = tempfile.mkdtemp()
        yield f"file://{temp_dir}"
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_logger_creation(self, temp_mlruns):
        """Test GRCMMLflowLogger creation."""
        logger = GRCMMLflowLogger(
            experiment_name="test-experiment",
            tracking_uri=temp_mlruns
        )
        assert logger is not None
        assert logger.experiment_name == "test-experiment"

    def test_context_manager(self, temp_mlruns):
        """Test logger as context manager."""
        logger = GRCMMLflowLogger(
            experiment_name="test-context",
            tracking_uri=temp_mlruns
        )

        with logger:
            assert logger.run is not None

        # Run should be ended
        assert logger.run is not None  # Still has reference

    def test_log_model_config(self, temp_mlruns):
        """Test logging model configuration."""
        logger = GRCMMLflowLogger(
            experiment_name="test-config",
            tracking_uri=temp_mlruns
        )

        config = {
            "input_dim": 15,
            "freq_dim": 8,
            "memory_size": 32
        }

        with logger:
            logger.log_model_config(config)

    def test_log_step(self, temp_mlruns):
        """Test logging a single step."""
        model = ResonantConsciousnessModule(15, 8, 32)
        logger = GRCMMLflowLogger(
            experiment_name="test-step",
            tracking_uri=temp_mlruns
        )

        image_emb = torch.randn(1, 512)
        audio_emb = torch.randn(1, 768)

        with logger:
            result = model(image_emb, audio_emb)
            logger.log_step(result, step=0)

    def test_log_batch(self, temp_mlruns):
        """Test logging multiple steps."""
        model = ResonantConsciousnessModule(15, 8, 32)
        logger = GRCMMLflowLogger(
            experiment_name="test-batch",
            tracking_uri=temp_mlruns
        )

        results = []
        for _ in range(5):
            image_emb = torch.randn(1, 512)
            audio_emb = torch.randn(1, 768)
            result = model(image_emb, audio_emb)
            results.append(result)

        with logger:
            logger.log_batch(results, start_step=0)

    def test_log_phi_statistics(self, temp_mlruns):
        """Test logging phi statistics."""
        logger = GRCMMLflowLogger(
            experiment_name="test-phi",
            tracking_uri=temp_mlruns
        )

        phi_history = [1.2, 1.5, 1.8, 1.6, 1.7, 1.9, 2.0, 1.8, 1.7, 1.6]

        with logger:
            logger.log_phi_statistics(phi_history, window=10)

    def test_log_coherence_quality(self, temp_mlruns):
        """Test logging coherence quality."""
        logger = GRCMMLflowLogger(
            experiment_name="test-coherence",
            tracking_uri=temp_mlruns
        )

        coherence_scores = [0.8, 0.9, 0.75, 0.6, 0.85, 0.7, 0.95, 0.8, 0.9, 0.85]

        with logger:
            logger.log_coherence_quality(coherence_scores, threshold=0.7)

    def test_create_experiment(self, temp_mlruns):
        """Test creating a complete experiment."""
        model = ResonantConsciousnessModule(15, 8, 32)

        run_id = create_mlflow_experiment(
            model,
            num_steps=20,
            experiment_name="test-full-experiment"
        )

        assert run_id is not None
        assert isinstance(run_id, str)


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
@pytest.mark.requires_mlflow
@pytest.mark.integration
class TestMLflowWorkflow:
    """Integration tests for MLflow workflows."""

    @pytest.fixture
    def temp_mlruns(self):
        """Create temporary MLflow tracking directory."""
        temp_dir = tempfile.mkdtemp()
        yield f"file://{temp_dir}"
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_experiment_workflow(self, temp_mlruns):
        """Test complete experiment workflow."""
        model = ResonantConsciousnessModule(15, 8, 32)
        logger = GRCMMLflowLogger(
            experiment_name="integration-test",
            tracking_uri=temp_mlruns
        )

        with logger:
            # Log config
            config = {
                "input_dim": 15,
                "freq_dim": 8,
                "memory_size": 32,
                "phi_threshold": 1.5
            }
            logger.log_model_config(config)

            # Run steps
            for step in range(10):
                image_emb = torch.randn(1, 512)
                audio_emb = torch.randn(1, 768)
                result = model(image_emb, audio_emb)

                logger.log_step(result, step=step)

            # Log statistics
            logger.log_phi_statistics(model.phi.phi_history)

            coherence_scores = [0.8] * 10
            logger.log_coherence_quality(coherence_scores)

        # Experiment completed successfully
        assert len(model.phi.phi_history) == 10
