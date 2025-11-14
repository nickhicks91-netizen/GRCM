"""
GRCM-Resonant: Grounded Resonant Consciousness Module
======================================================

A PyTorch-based consciousness simulation kernel featuring:
- Resonant attention with frequency-based coherence gating
- Desire-driven agency and goal-directed behavior
- Qualia simulation (phenomenal states)
- Phi estimation (integrated information proxy)
- Episodic threading for narrative self-continuity
- Multimodal grounding (vision, audio, proprioception)
- EchoMirror training for human-aligned desires

Usage:
    from grcm import ResonantConsciousnessModule

    model = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    )

    result = model(image_emb, audio_emb, action)
    print(f"Phi: {result['phi']:.3f}")
    print(f"Qualia: {result['qualia']}")
"""

__version__ = "0.1.0"
__author__ = "Nicholas"
__license__ = "MIT"

# Core module
from .core import ResonantConsciousnessModule

# Submodules (optional imports)
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

# Training
from .training import echo_mirror_train

# Optimization (optional imports - may not be available in all environments)
try:
    from .optimization import OptimizedGRCM, export_to_onnx, test_onnx_inference
    from .benchmark import GRCMBenchmark, BenchmarkResult
    _optimization_available = True
except ImportError:
    _optimization_available = False

__all__ = [
    # Main
    "ResonantConsciousnessModule",
    # Components
    "GroundingLayer",
    "HarmonicEmbedding",
    "ResonantAttention",
    "DesireModule",
    "MemoryGrid",
    "ReflectionHead",
    "QualiaModule",
    "EpisodicThreadBank",
    "PhiEstimator",
    "BodySimulator",
    # Training
    "echo_mirror_train",
]

# Add optimization to __all__ if available
if _optimization_available:
    __all__.extend([
        "OptimizedGRCM",
        "export_to_onnx",
        "test_onnx_inference",
        "GRCMBenchmark",
        "BenchmarkResult",
    ])
