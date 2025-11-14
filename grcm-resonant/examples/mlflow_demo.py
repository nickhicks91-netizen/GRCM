#!/usr/bin/env python3
"""
MLflow Experiment Tracking Demo for GRCM

Demonstrates:
    - Experiment creation
    - Metric logging (phi, coherence, qualia)
    - Model versioning
    - Run comparison

Usage:
    python examples/mlflow_demo.py
    mlflow ui  # View results at http://localhost:5000
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    from grcm import ResonantConsciousnessModule
    from grcm.mlflow_logger import GRCMMLflowLogger, create_mlflow_experiment
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_OK = False


def demo_basic_logging():
    """Demo: Basic metric logging."""
    print("=" * 70)
    print("1. BASIC METRIC LOGGING")
    print("=" * 70)

    model = ResonantConsciousnessModule(15, 8, 32)
    logger = GRCMMLflowLogger(experiment_name="GRCM-Basic-Demo")

    with logger:
        # Log config
        config = {
            "input_dim": 15,
            "freq_dim": 8,
            "memory_size": 32,
            "phi_threshold": 1.5,
            "demo_type": "basic"
        }
        logger.log_model_config(config)

        # Run 50 steps
        print("\nRunning 50 steps...")
        for step in range(50):
            image_emb = torch.randn(1, 512)
            audio_emb = torch.randn(1, 768)
            action = torch.randn(1, 4)

            with torch.no_grad():
                result = model(image_emb, audio_emb, action)

            logger.log_step(result, step=step)

            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/50")

        # Log final statistics
        logger.log_phi_statistics(model.phi.phi_history)

        coherence_scores = [r for r in range(50)]  # Dummy for demo
        logger.log_coherence_quality([0.8] * 50, threshold=0.7)

    print("\n✓ Basic logging complete!")
    print(f"  Run ID: {logger.run.info.run_id}")


def demo_desire_comparison():
    """Demo: Compare different desire states."""
    print("\n" + "=" * 70)
    print("2. DESIRE STATE COMPARISON")
    print("=" * 70)

    desire_names = ["Curiosity", "Safety", "Social", "Exploration"]

    for desire_idx, desire_name in enumerate(desire_names):
        print(f"\n  Testing {desire_name}...")

        model = ResonantConsciousnessModule(15, 8, 32)
        model.set_desire(desire_idx)

        logger = GRCMMLflowLogger(experiment_name="GRCM-Desire-Comparison")

        with logger:
            # Tag with desire name
            logger.run.set_tag("desire", desire_name)
            logger.run.set_tag("desire_idx", str(desire_idx))

            # Log config
            config = {
                "input_dim": 15,
                "freq_dim": 8,
                "memory_size": 32,
                "desire": desire_name,
                "desire_idx": desire_idx
            }
            logger.log_model_config(config)

            # Run steps
            for step in range(30):
                image_emb = torch.randn(1, 512)
                audio_emb = torch.randn(1, 768)

                with torch.no_grad():
                    result = model(image_emb, audio_emb)

                logger.log_step(result, step=step)

            # Log statistics
            logger.log_phi_statistics(model.phi.phi_history)

        print(f"    ✓ {desire_name} complete (Run: {logger.run.info.run_id})")

    print("\n✓ Desire comparison complete!")
    print("  Compare runs in MLflow UI")


def demo_model_registry():
    """Demo: Model versioning and registry."""
    print("\n" + "=" * 70)
    print("3. MODEL REGISTRY")
    print("=" * 70)

    model = ResonantConsciousnessModule(15, 8, 32)
    model.eval()

    logger = GRCMMLflowLogger(experiment_name="GRCM-Model-Registry")

    with logger:
        # Log config
        config = {"input_dim": 15, "freq_dim": 8, "memory_size": 32}
        logger.log_model_config(config)

        # Run brief evaluation
        print("\nRunning brief evaluation...")
        for step in range(20):
            image_emb = torch.randn(1, 512)
            audio_emb = torch.randn(1, 768)

            with torch.no_grad():
                result = model(image_emb, audio_emb)

            logger.log_step(result, step=step)

        # Log model to registry
        print("\nLogging model to registry...")
        logger.log_model(
            model,
            artifact_path="model",
            registered_model_name="GRCM-Resonant-v1"
        )

    print("\n✓ Model registered!")
    print("  View in MLflow UI under Models tab")


def demo_full_experiment():
    """Demo: Complete experiment using helper function."""
    print("\n" + "=" * 70)
    print("4. FULL EXPERIMENT (Helper Function)")
    print("=" * 70)

    model = ResonantConsciousnessModule(15, 8, 32)

    run_id = create_mlflow_experiment(
        model,
        num_steps=100,
        experiment_name="GRCM-Full-Experiment",
        log_model=False  # Don't log model for demo
    )

    print(f"\n✓ Full experiment complete!")
    print(f"  Run ID: {run_id}")


def main():
    """Run all demos."""
    if not IMPORTS_OK:
        print("✗ Import error. Install dependencies:")
        print("  pip install mlflow")
        return 1

    print("=" * 70)
    print("GRCM MLFLOW DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how to use MLflow with GRCM for:")
    print("  1. Basic metric logging (phi, coherence, qualia)")
    print("  2. Comparing desire states")
    print("  3. Model versioning and registry")
    print("  4. Complete experiment workflows")
    print("\nAll experiments will be logged to ./mlruns")
    print("View results with: mlflow ui")

    # Run demos
    try:
        demo_basic_logging()
        demo_desire_comparison()
        demo_model_registry()
        demo_full_experiment()

        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Start MLflow UI: mlflow ui")
        print("  2. Open browser: http://localhost:5000")
        print("  3. Compare runs and visualize metrics")
        print("  4. Explore model registry")

        return 0

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
