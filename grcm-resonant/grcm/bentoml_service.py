"""BentoML service for GRCM production deployment."""
import warnings
from typing import Dict, Any, Optional
import numpy as np

try:
    import bentoml
    from bentoml.io import JSON, NumpyNdarray
    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False
    warnings.warn("BentoML not installed. Install with: pip install bentoml")

import torch
from pathlib import Path


if BENTOML_AVAILABLE:
    # Define the BentoML service
    @bentoml.service(
        name="grcm-resonant",
        resources={
            "cpu": "2",
            "memory": "2Gi",
        },
        traffic={
            "timeout": 30,
        }
    )
    class GRCMService:
        """
        BentoML service for GRCM inference.

        Endpoints:
            - /predict: Single inference
            - /predict_batch: Batch inference
            - /health: Health check
            - /metrics: Prometheus metrics
        """

        def __init__(self):
            """Initialize GRCM model."""
            from grcm import ResonantConsciousnessModule
            from grcm.config import load_config

            # Load config
            config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
            if config_path.exists():
                config = load_config(str(config_path))
            else:
                config = load_config()

            # Initialize model
            self.model = ResonantConsciousnessModule(
                input_dim=config.input_dim,
                freq_dim=config.freq_dim,
                memory_size=config.memory_size,
                phi_threshold=config.phi_threshold
            )
            self.model.eval()

            # Metrics
            self.request_count = 0
            self.error_count = 0

        @bentoml.api
        def predict(
            self,
            image_emb: NumpyNdarray,
            audio_emb: NumpyNdarray,
            action: Optional[NumpyNdarray] = None,
            desire_idx: int = 0
        ) -> JSON:
            """
            Single inference endpoint.

            Args:
                image_emb: (512,) CLIP image embedding
                audio_emb: (768,) Wav2Vec audio embedding
                action: Optional (4,) action vector
                desire_idx: Desire state (0-3)

            Returns:
                JSON with outputs: coherence, phi, qualia, halt, etc.
            """
            self.request_count += 1

            try:
                # Set desire
                self.model.set_desire(desire_idx)

                # Convert to tensors
                image_tensor = torch.from_numpy(image_emb).float().unsqueeze(0)
                audio_tensor = torch.from_numpy(audio_emb).float().unsqueeze(0)

                if action is not None:
                    action_tensor = torch.from_numpy(action).float().unsqueeze(0)
                else:
                    action_tensor = None

                # Inference
                with torch.no_grad():
                    result = self.model(image_tensor, audio_tensor, action_tensor)

                # Format response
                response = {
                    "coherence": float(result['coherence'].mean().item()),
                    "phi": float(result['phi']),
                    "desire_align": float(result['desire_align'].mean().item()),
                    "reflection": float(result['reflection'].mean().item()),
                    "qualia": result['qualia'][0].cpu().numpy().tolist(),
                    "halt": bool(result['halt']),
                    "timestamp": int(result['timestamp']),
                    "prop_state": result['prop_state'][0].cpu().numpy().tolist()
                }

                return response

            except Exception as e:
                self.error_count += 1
                return {
                    "error": str(e),
                    "request_count": self.request_count,
                    "error_count": self.error_count
                }

        @bentoml.api
        def predict_batch(
            self,
            image_emb_batch: NumpyNdarray,
            audio_emb_batch: NumpyNdarray,
            action_batch: Optional[NumpyNdarray] = None,
            desire_idx: int = 0
        ) -> JSON:
            """
            Batch inference endpoint.

            Args:
                image_emb_batch: (batch, 512) CLIP embeddings
                audio_emb_batch: (batch, 768) Wav2Vec embeddings
                action_batch: Optional (batch, 4) actions
                desire_idx: Desire state

            Returns:
                JSON with batch outputs
            """
            self.request_count += 1

            try:
                # Set desire
                self.model.set_desire(desire_idx)

                # Convert to tensors
                image_tensor = torch.from_numpy(image_emb_batch).float()
                audio_tensor = torch.from_numpy(audio_emb_batch).float()

                if action_batch is not None:
                    action_tensor = torch.from_numpy(action_batch).float()
                else:
                    action_tensor = None

                # Batch inference
                with torch.no_grad():
                    result = self.model(image_tensor, audio_tensor, action_tensor)

                # Format batch response
                batch_size = image_emb_batch.shape[0]
                responses = []

                for i in range(batch_size):
                    responses.append({
                        "coherence": float(result['coherence'][i].item()),
                        "desire_align": float(result['desire_align'][i].item()),
                        "reflection": float(result['reflection'][i].item()),
                        "qualia": result['qualia'][i].cpu().numpy().tolist(),
                    })

                return {
                    "batch_size": batch_size,
                    "phi": float(result['phi']),
                    "halt": bool(result['halt']),
                    "timestamp": int(result['timestamp']),
                    "results": responses
                }

            except Exception as e:
                self.error_count += 1
                return {
                    "error": str(e),
                    "request_count": self.request_count,
                    "error_count": self.error_count
                }

        @bentoml.api
        def health(self) -> JSON:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model": "grcm-resonant",
                "version": "0.1.0",
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1)
            }

        @bentoml.api
        def metrics(self) -> JSON:
            """Prometheus-compatible metrics."""
            metrics = self.model.get_metrics()
            metrics.update({
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1)
            })
            return metrics


def create_bentoml_service():
    """Create and return BentoML service for deployment."""
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML not installed")

    return GRCMService()


if __name__ == "__main__":
    if BENTOML_AVAILABLE:
        print("=" * 60)
        print("GRCM BentoML Service")
        print("=" * 60)
        print("\nStarting BentoML server...")
        print("API endpoints:")
        print("  POST /predict - Single inference")
        print("  POST /predict_batch - Batch inference")
        print("  GET /health - Health check")
        print("  GET /metrics - Prometheus metrics")
        print("\nServe with: bentoml serve grcm.bentoml_service:GRCMService")
    else:
        print("BentoML not installed. Install with: pip install bentoml")
