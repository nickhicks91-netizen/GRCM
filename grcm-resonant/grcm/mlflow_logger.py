"""MLflow integration for GRCM experiment tracking and monitoring."""
import warnings
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed. Install with: pip install mlflow")


class GRCMMLflowLogger:
    """
    MLflow integration for GRCM experiment tracking.

    Logs:
        - Phi (Φ) values over time
        - Coherence metrics
        - Qualia distributions
        - Desire alignment
        - Model parameters and hyperparameters
        - Performance metrics (latency, throughput)

    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking server URI (default: local ./mlruns)
        auto_log: Enable automatic logging (default: True)
    """

    def __init__(
        self,
        experiment_name: str = "GRCM-Resonant",
        tracking_uri: Optional[str] = None,
        auto_log: bool = True
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create/get experiment
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

        self.auto_log = auto_log
        self.run = None
        self.step_counter = 0

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Dictionary of tags
            description: Run description
        """
        self.run = mlflow.start_run(run_name=run_name, description=description)

        if tags:
            mlflow.set_tags(tags)

        # Default tags
        mlflow.set_tag("framework", "PyTorch")
        mlflow.set_tag("model", "GRCM-Resonant")

        print(f"✓ Started MLflow run: {self.run.info.run_id}")

    def log_model_config(self, config: Dict[str, Any]) -> None:
        """
        Log model configuration parameters.

        Args:
            config: Dictionary of model hyperparameters
        """
        mlflow.log_params(config)
        print(f"✓ Logged {len(config)} config parameters")

    def log_step(
        self,
        result: Dict[str, Any],
        step: Optional[int] = None,
        log_qualia_distribution: bool = True
    ) -> None:
        """
        Log a single forward pass result.

        Args:
            result: Dictionary from GRCM forward pass
            step: Step number (auto-increments if None)
            log_qualia_distribution: Log full qualia distribution
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        # Core metrics
        mlflow.log_metric("phi", result['phi'], step=step)
        mlflow.log_metric("coherence_mean", result['coherence'].mean().item(), step=step)
        mlflow.log_metric("desire_align_mean", result['desire_align'].mean().item(), step=step)
        mlflow.log_metric("reflection_mean", result['reflection'].mean().item(), step=step)

        # Qualia distribution
        if log_qualia_distribution:
            qualia = result['qualia'][0].detach().cpu().numpy()
            mlflow.log_metric("qualia_calm", float(qualia[0]), step=step)
            mlflow.log_metric("qualia_alert", float(qualia[1]), step=step)
            mlflow.log_metric("qualia_curious", float(qualia[2]), step=step)
            mlflow.log_metric("qualia_conflicted", float(qualia[3]), step=step)

        # Ethical monitoring
        if result['halt']:
            mlflow.log_metric("ethical_halt", 1.0, step=step)

    def log_batch(
        self,
        results: List[Dict[str, Any]],
        start_step: int = 0
    ) -> None:
        """
        Log multiple results efficiently.

        Args:
            results: List of GRCM forward pass results
            start_step: Starting step number
        """
        for i, result in enumerate(results):
            self.log_step(result, step=start_step + i, log_qualia_distribution=False)

        # Log aggregated qualia at last step
        if results:
            last_result = results[-1]
            qualia = last_result['qualia'][0].detach().cpu().numpy()
            step = start_step + len(results) - 1

            mlflow.log_metric("qualia_calm", float(qualia[0]), step=step)
            mlflow.log_metric("qualia_alert", float(qualia[1]), step=step)
            mlflow.log_metric("qualia_curious", float(qualia[2]), step=step)
            mlflow.log_metric("qualia_conflicted", float(qualia[3]), step=step)

        print(f"✓ Logged {len(results)} steps")

    def log_benchmark(
        self,
        benchmark_result,
        prefix: str = "benchmark"
    ) -> None:
        """
        Log benchmark results.

        Args:
            benchmark_result: BenchmarkResult instance
            prefix: Metric prefix
        """
        metrics = {
            f"{prefix}_mean_latency_ms": benchmark_result.mean_latency_ms,
            f"{prefix}_std_latency_ms": benchmark_result.std_latency_ms,
            f"{prefix}_p95_latency_ms": benchmark_result.p95_latency_ms,
            f"{prefix}_p99_latency_ms": benchmark_result.p99_latency_ms,
            f"{prefix}_throughput": benchmark_result.throughput_samples_per_sec,
        }

        mlflow.log_metrics(metrics)
        print(f"✓ Logged benchmark metrics: {benchmark_result.name}")

    def log_phi_statistics(
        self,
        phi_history: List[float],
        window: int = 10
    ) -> None:
        """
        Log Phi statistics (mean, std, stability).

        Args:
            phi_history: List of Phi values
            window: Window for computing statistics
        """
        if len(phi_history) < 2:
            return

        recent = phi_history[-window:]
        mean_phi = sum(recent) / len(recent)
        variance = sum((x - mean_phi) ** 2 for x in recent) / len(recent)
        std_phi = variance ** 0.5

        mlflow.log_metrics({
            "phi_mean": mean_phi,
            "phi_std": std_phi,
            "phi_stable": 1.0 if std_phi < 0.2 else 0.0
        })

    def log_coherence_quality(
        self,
        coherence_scores: List[float],
        threshold: float = 0.7
    ) -> None:
        """
        Log coherence quality metrics.

        Args:
            coherence_scores: List of coherence values
            threshold: Coherence threshold (default 0.7)
        """
        above_threshold = sum(1 for c in coherence_scores if c > threshold)
        percentage = (above_threshold / len(coherence_scores)) * 100

        mlflow.log_metrics({
            "coherence_mean": sum(coherence_scores) / len(coherence_scores),
            "coherence_percentage_above_threshold": percentage,
            "coherence_threshold": threshold
        })

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log PyTorch model to MLflow.

        Args:
            model: GRCM model instance
            artifact_path: Path within run artifacts
            registered_model_name: Name for model registry
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        print(f"✓ Logged model to {artifact_path}")

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Log a local file as an artifact.

        Args:
            local_path: Local file path
            artifact_path: Path within run artifacts
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        filename: str
    ) -> None:
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Artifact filename (e.g., "config.json")
        """
        mlflow.log_dict(dictionary, filename)

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.run:
            mlflow.end_run(status=status)
            print(f"✓ Ended MLflow run: {status}")

    def __enter__(self):
        """Context manager entry."""
        if not self.run:
            self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)


def create_mlflow_experiment(
    model,
    num_steps: int = 100,
    experiment_name: str = "GRCM-Evaluation",
    log_model: bool = False
) -> str:
    """
    Create a complete MLflow experiment with a GRCM model.

    Args:
        model: ResonantConsciousnessModule instance
        num_steps: Number of evaluation steps
        experiment_name: MLflow experiment name
        log_model: Whether to log the model itself

    Returns:
        run_id: MLflow run ID
    """
    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow not installed")

    import torch

    logger = GRCMMLflowLogger(experiment_name=experiment_name)

    with logger:
        # Log model config
        config = {
            "input_dim": model.input_dim,
            "freq_dim": model.freq_dim,
            "memory_size": model.memory_size,
            "phi_threshold": model.phi_threshold,
            "num_steps": num_steps
        }
        logger.log_model_config(config)

        # Run evaluation
        print(f"\nRunning {num_steps} evaluation steps...")

        for step in range(num_steps):
            image_emb = torch.randn(1, 512)
            audio_emb = torch.randn(1, 768)
            action = torch.randn(1, 4)

            with torch.no_grad():
                result = model(image_emb, audio_emb, action)

            logger.log_step(result, step=step)

            if (step + 1) % 25 == 0:
                print(f"  Step {step + 1}/{num_steps}")

        # Log final statistics
        logger.log_phi_statistics(model.phi.phi_history)

        # Log model if requested
        if log_model:
            logger.log_model(model, registered_model_name="GRCM-Resonant")

        run_id = logger.run.info.run_id

    print(f"\n✓ Experiment complete!")
    print(f"  View results: mlflow ui")
    print(f"  Run ID: {run_id}")

    return run_id


if __name__ == "__main__":
    print("GRCM MLflow Integration")
    print("=" * 50)

    if MLFLOW_AVAILABLE:
        print("\n✓ MLflow available")
        print("\nUsage:")
        print("  from grcm.mlflow_logger import GRCMMLflowLogger")
        print("  logger = GRCMMLflowLogger('my-experiment')")
        print("  with logger:")
        print("      result = model(image_emb, audio_emb)")
        print("      logger.log_step(result)")
    else:
        print("\n✗ MLflow not installed")
        print("  Install with: pip install mlflow")
