#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for GRCM

Compares:
    - Baseline (eager mode)
    - torch.compile
    - INT8 quantization
    - torch.compile + quantization

Usage:
    python scripts/benchmark_all.py
    python scripts/benchmark_all.py --iterations 200 --save results.json
"""
import argparse
import sys
from pathlib import Path
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from grcm import ResonantConsciousnessModule
from grcm.optimization import OptimizedGRCM
from grcm.benchmark import GRCMBenchmark


def main():
    parser = argparse.ArgumentParser(description='Benchmark GRCM optimizations')
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations (default 100)'
    )
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help='Batch sizes to test (default: 1 2 4 8)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )
    parser.add_argument(
        '--skip-compile',
        action='store_true',
        help='Skip torch.compile benchmark'
    )
    parser.add_argument(
        '--skip-quantize',
        action='store_true',
        help='Skip quantization benchmark'
    )

    args = parser.parse_args()

    # Check device
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')

    print("=" * 70)
    print("GRCM OPTIMIZATION BENCHMARK SUITE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device:       {device}")
    print(f"  Iterations:   {args.iterations}")
    print(f"  Batch Sizes:  {args.batch_sizes}")
    print(f"  Save Results: {args.save or 'None'}")

    # Create base model
    print("\n" + "=" * 70)
    print("Creating Models...")
    print("=" * 70)

    base_model = ResonantConsciousnessModule(
        input_dim=15,
        freq_dim=8,
        memory_size=32
    ).to(device)
    base_model.eval()
    print("✓ Baseline model created")

    models = {'Baseline (Eager)': base_model}

    # torch.compile
    if not args.skip_compile:
        try:
            compiled_model = OptimizedGRCM(
                base_model,
                mode='reduce-overhead',
                quantize=False
            )
            models['torch.compile'] = compiled_model
            print("✓ torch.compile model created")
        except Exception as e:
            warnings.warn(f"torch.compile failed: {e}")

    # Quantization (CPU only)
    if not args.skip_quantize and device.type == 'cpu':
        try:
            quantized_model = OptimizedGRCM(
                ResonantConsciousnessModule(
                    input_dim=15,
                    freq_dim=8,
                    memory_size=32
                ),
                mode='default',
                quantize=True
            )
            models['INT8 Quantized'] = quantized_model
            print("✓ Quantized model created")
        except Exception as e:
            warnings.warn(f"Quantization failed: {e}")

        # torch.compile + quantization
        if not args.skip_compile:
            try:
                compiled_quant_model = OptimizedGRCM(
                    ResonantConsciousnessModule(
                        input_dim=15,
                        freq_dim=8,
                        memory_size=32
                    ),
                    mode='reduce-overhead',
                    quantize=True
                )
                models['torch.compile + INT8'] = compiled_quant_model
                print("✓ torch.compile + quantized model created")
            except Exception as e:
                warnings.warn(f"Combined optimization failed: {e}")

    # Run benchmarks
    print("\n" + "=" * 70)
    print("Running Benchmarks...")
    print("=" * 70)

    all_results = {}

    for name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {name}")
        print(f"{'=' * 70}")

        benchmark = GRCMBenchmark(model, device=device, warmup_iterations=10)

        # Run full benchmark
        save_path = f"{args.save}.{name.replace(' ', '_').replace('+', '_')}.json" if args.save else None
        results = benchmark.full_benchmark(
            optimization=name,
            save_path=save_path
        )

        all_results[name] = results

    # Comparison summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    baseline_latency = all_results['Baseline (Eager)']['latency']['mean_latency_ms']

    print(f"\n{'Optimization':<30} {'Latency (ms)':<15} {'Speedup':<10} {'Throughput (samp/s)':<20}")
    print("-" * 70)

    for name, results in all_results.items():
        lat = results['latency']['mean_latency_ms']
        speedup = baseline_latency / lat
        throughput = results['latency']['throughput_samples_per_sec']

        print(f"{name:<30} {lat:<15.2f} {speedup:<10.2f}x {throughput:<20.1f}")

    # Quality metrics comparison
    print("\n" + "=" * 70)
    print("QUALITY METRICS COMPARISON")
    print("=" * 70)

    print(f"\n{'Optimization':<30} {'Coherence >0.7':<15} {'Phi Std':<10}")
    print("-" * 70)

    for name, results in all_results.items():
        coh_pct = results['coherence_quality']['percentage_above_threshold']
        phi_std = results['phi_stability']['std_phi']

        print(f"{name:<30} {coh_pct:<15.1f}% {phi_std:<10.3f}")

    # Save combined results
    if args.save:
        import json
        combined_path = args.save
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Combined results saved to {combined_path}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if device.type == 'cpu':
        print("\n✓ For CPU deployment:")
        print("  - Use torch.compile for best performance (2-3x speedup)")
        print("  - Add INT8 quantization for lower memory (minimal accuracy loss)")
        print("  - Combined: torch.compile + INT8 for production")
        print("\n✓ For edge devices:")
        print("  - Export to ONNX: python scripts/export_onnx.py")
        print("  - Convert to TensorRT for 5-10x speedup on NVIDIA GPUs")
    else:
        print("\n✓ For GPU deployment:")
        print("  - Use torch.compile with mode='max-autotune'")
        print("  - Enable FP16 mixed precision for 2x speedup")
        print("  - Consider TensorRT for inference servers")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
