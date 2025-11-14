#!/usr/bin/env python3
"""
GRCM Optimization Demonstration

Shows how to use all optimization features:
    - torch.compile
    - INT8 quantization
    - ONNX export
    - Benchmarking

Usage:
    python examples/optimization_demo.py
"""
import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import ResonantConsciousnessModule
from grcm.optimization import OptimizedGRCM, export_to_onnx, test_onnx_inference
from grcm.benchmark import GRCMBenchmark


def demo_torch_compile():
    """Demonstrate torch.compile optimization."""
    print("=" * 70)
    print("1. TORCH.COMPILE OPTIMIZATION")
    print("=" * 70)

    # Create base model
    print("\nCreating base model...")
    base_model = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    )
    base_model.eval()

    # Wrap with torch.compile
    print("Applying torch.compile...")
    optimized = OptimizedGRCM(
        base_model,
        mode='reduce-overhead',
        quantize=False
    )

    # Test inference
    print("\nTesting inference...")
    image_emb = torch.randn(1, 512)
    audio_emb = torch.randn(1, 768)
    action = torch.tensor([[0.1, 0.2, 0.0, 0.0]])

    result = optimized(image_emb, audio_emb, action)

    print(f"✓ Inference successful")
    print(f"  Coherence: {result['coherence'].item():.3f}")
    print(f"  Phi: {result['phi']:.3f}")
    print(f"  Halt: {result['halt']}")

    # Optimization info
    info = optimized.get_optimization_info()
    print(f"\nOptimization info:")
    for key, val in info.items():
        print(f"  {key}: {val}")

    return optimized


def demo_quantization():
    """Demonstrate INT8 quantization."""
    print("\n" + "=" * 70)
    print("2. INT8 QUANTIZATION")
    print("=" * 70)

    print("\nCreating quantized model...")
    base_model = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    )

    quantized = OptimizedGRCM(
        base_model,
        mode='default',
        quantize=True
    )

    # Test inference
    print("Testing quantized inference...")
    image_emb = torch.randn(1, 512)
    audio_emb = torch.randn(1, 768)

    result = quantized(image_emb, audio_emb)

    print(f"✓ Quantized inference successful")
    print(f"  Coherence: {result['coherence'].item():.3f}")
    print(f"  Qualia: {result['qualia'].detach().numpy()[0]}")

    return quantized


def demo_benchmarking(models):
    """Demonstrate benchmarking."""
    print("\n" + "=" * 70)
    print("3. BENCHMARKING")
    print("=" * 70)

    benchmark = GRCMBenchmark(models['baseline'], warmup_iterations=5)

    print("\nRunning quick benchmark (50 iterations)...")
    result = benchmark.benchmark_latency(
        batch_size=1,
        num_iterations=50,
        name="Quick Benchmark",
        optimization="baseline"
    )

    print("\nLatency Distribution:")
    print(f"  Mean: {result.mean_latency_ms:.2f} ms")
    print(f"  Std:  {result.std_latency_ms:.2f} ms")
    print(f"  P95:  {result.p95_latency_ms:.2f} ms")
    print(f"  P99:  {result.p99_latency_ms:.2f} ms")

    # Compare optimizations
    if len(models) > 1:
        print("\n" + "=" * 70)
        print("COMPARING OPTIMIZATIONS")
        print("=" * 70)

        benchmark.compare_optimizations(models, num_iterations=30)


def demo_onnx_export():
    """Demonstrate ONNX export."""
    print("\n" + "=" * 70)
    print("4. ONNX EXPORT")
    print("=" * 70)

    # Create model
    print("\nCreating model for export...")
    model = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    )

    # Export
    output_path = '/tmp/grcm_demo.onnx'
    print(f"Exporting to {output_path}...")

    try:
        export_to_onnx(
            model,
            output_path=output_path,
            opset_version=18,
            dynamic_batch=True,
            simplify=False
        )

        # Test ONNX inference
        print("\nTesting ONNX inference...")
        test_onnx_inference(output_path, num_samples=3)

    except Exception as e:
        print(f"⚠ ONNX export/test skipped: {e}")
        print("  Install with: pip install onnx onnxruntime")


def demo_quality_metrics():
    """Demonstrate quality metrics."""
    print("\n" + "=" * 70)
    print("5. QUALITY METRICS")
    print("=" * 70)

    model = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    )

    benchmark = GRCMBenchmark(model)

    # Coherence quality
    print("\nCoherence Quality:")
    coh_metrics = benchmark.benchmark_coherence_quality(num_samples=50)

    # Phi stability
    print("\nPhi Stability:")
    phi_metrics = benchmark.benchmark_phi_stability(num_iterations=50)

    # Memory usage
    print("\nMemory Usage:")
    mem_metrics = benchmark.memory_usage()


def main():
    print("=" * 70)
    print("GRCM OPTIMIZATION DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo showcases all optimization features:")
    print("  1. torch.compile for 2-3x speedup")
    print("  2. INT8 quantization for lower memory")
    print("  3. Comprehensive benchmarking")
    print("  4. ONNX export for cross-platform deployment")
    print("  5. Quality metrics (coherence, phi)")

    # Collect models for comparison
    models = {}

    # Baseline
    print("\n" + "=" * 70)
    print("CREATING MODELS")
    print("=" * 70)

    baseline = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    )
    baseline.eval()
    models['baseline'] = baseline
    print("✓ Baseline model created")

    # Torch compile
    try:
        compiled = demo_torch_compile()
        models['torch.compile'] = compiled
    except Exception as e:
        print(f"⚠ torch.compile demo skipped: {e}")

    # Quantization
    try:
        quantized = demo_quantization()
        models['quantized'] = quantized
    except Exception as e:
        print(f"⚠ Quantization demo skipped: {e}")

    # Benchmarking
    try:
        demo_benchmarking(models)
    except Exception as e:
        print(f"⚠ Benchmarking demo error: {e}")

    # ONNX export
    demo_onnx_export()

    # Quality metrics
    demo_quality_metrics()

    # Final summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run full benchmark: python scripts/benchmark_all.py")
    print("  2. Export for production: python scripts/export_onnx.py --simplify --test")
    print("  3. Deploy with TensorRT: See docs/TENSORRT_GUIDE.md")
    print("  4. Serve with BentoML: Coming in Phase 4")


if __name__ == "__main__":
    main()
