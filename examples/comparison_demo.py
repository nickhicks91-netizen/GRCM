"""
GRCM vs Transformer Baseline Comparison
Demonstrates efficiency gains of GRCM over standard transformers
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from grcm import ModularGRCM, load_config, ModelComparator
from baselines import TransformerBaseline, BaselineConfig
from baselines.transformer_baseline import MultimodalTransformerBaseline, create_comparable_baseline


def main():
    print("=" * 70)
    print("GRCM EFFICIENCY COMPARISON")
    print("GRCM vs Standard Transformer Baseline")
    print("=" * 70)

    # 1. Load GRCM model
    print("\n[1/4] Loading GRCM model...")
    grcm_config = load_config("config/grcm_default.yaml")
    grcm_model = ModularGRCM(grcm_config)
    grcm_model.eval()

    grcm_params = sum(p.numel() for p in grcm_model.parameters() if p.requires_grad)
    print(f"    ✓ GRCM loaded: {grcm_params:,} parameters")

    # 2. Create comparable transformer baseline
    print("\n[2/4] Creating transformer baseline...")
    baseline_model = create_comparable_baseline(grcm_config)
    baseline_model.eval()

    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    print(f"    ✓ Baseline loaded: {baseline_params:,} parameters")

    # 3. Create comparator
    print("\n[3/4] Initializing comparator...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")

    comparator = ModelComparator(grcm_model, baseline_model, device=device)
    print("    ✓ Comparator ready")

    # 4. Run comparison
    print("\n[4/4] Running comprehensive comparison...")
    print("    (This may take 1-2 minutes...)")

    result = comparator.compare_models(
        batch_size=4,
        num_iterations=100,
        platform="A100" if device == "cuda" else "CPU"
    )

    # Print summary
    comparator.print_summary(result)

    # Additional analysis
    print("\n" + "=" * 70)
    print("DETAILED METRICS")
    print("=" * 70)

    print("\n[Latency Breakdown]")
    print(f"  GRCM latency:     {result.grcm_latency_ms:.3f}ms")
    print(f"  Baseline latency: {result.baseline_latency_ms:.3f}ms")
    print(f"  Absolute gain:    {result.baseline_latency_ms - result.grcm_latency_ms:.3f}ms")
    print(f"  Relative speedup: {result.speedup:.2f}x")

    if result.speedup >= 10:
        print(f"  🔥 EXCELLENT: {result.speedup:.1f}x speedup achieved!")
    elif result.speedup >= 5:
        print(f"  ✓ GOOD: {result.speedup:.1f}x speedup")
    elif result.speedup >= 2:
        print(f"  ✓ MODERATE: {result.speedup:.1f}x speedup")
    else:
        print(f"  ⚠ LIMITED: {result.speedup:.1f}x speedup")

    print("\n[Memory Breakdown]")
    print(f"  GRCM memory:     {result.grcm_memory_mb:.2f} MB")
    print(f"  Baseline memory: {result.baseline_memory_mb:.2f} MB")
    print(f"  Memory saved:    {result.baseline_memory_mb - result.grcm_memory_mb:.2f} MB")
    print(f"  Reduction:       {result.memory_reduction:.1f}%")

    print("\n[Computational Efficiency]")
    print(f"  GRCM FLOPs:     {result.grcm_flops:,}")
    print(f"  Baseline FLOPs: {result.baseline_flops:,}")
    print(f"  FLOPs saved:    {result.baseline_flops - result.grcm_flops:,}")
    print(f"  Reduction:      {result.flop_reduction:.1f}%")

    print("\n[Energy Efficiency]")
    print(f"  GRCM energy:     {result.grcm_energy_mj:.3f} mJ")
    print(f"  Baseline energy: {result.baseline_energy_mj:.3f} mJ")
    print(f"  Energy saved:    {result.baseline_energy_mj - result.grcm_energy_mj:.3f} mJ")
    print(f"  Savings:         {result.energy_savings:.1f}%")

    # Batch scaling analysis
    print("\n" + "=" * 70)
    print("BATCH SCALING ANALYSIS")
    print("=" * 70)
    print("Testing how efficiency scales with batch size...")

    batch_results = comparator.compare_batch_scaling(
        batch_sizes=[1, 2, 4, 8],
        num_iterations=50
    )

    print("\n[Batch Scaling Summary]")
    print(f"{'Batch':<10} {'GRCM (ms)':<12} {'Baseline (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    for bs, res in batch_results.items():
        print(f"{bs:<10} {res.grcm_latency_ms:<12.2f} {res.baseline_latency_ms:<15.2f} {res.speedup:<10.2f}x")

    # Export results
    print("\n" + "=" * 70)
    print("EXPORT RESULTS")
    print("=" * 70)

    Path("comparison_results").mkdir(exist_ok=True)
    comparator.export_results(result, "comparison_results/grcm_vs_transformer.json")
    print("    ✓ Results exported to comparison_results/grcm_vs_transformer.json")

    # Value proposition summary
    print("\n" + "=" * 70)
    print("VALUE PROPOSITION SUMMARY")
    print("=" * 70)

    if result.speedup >= 10:
        efficiency_level = "BREAKTHROUGH"
        market_impact = "TRANSFORMATIVE - Could reshape AI infrastructure market"
    elif result.speedup >= 5:
        efficiency_level = "SIGNIFICANT"
        market_impact = "HIGH - Strong competitive advantage for datacenter deployments"
    elif result.speedup >= 2:
        efficiency_level = "MODERATE"
        market_impact = "MEDIUM - Valuable for cost-sensitive applications"
    else:
        efficiency_level = "LIMITED"
        market_impact = "LOW - Needs further optimization"

    print(f"\nEfficiency Level: {efficiency_level}")
    print(f"Market Impact:    {market_impact}")

    print(f"\nKey Selling Points:")
    print(f"  1. {result.speedup:.1f}x faster inference")
    print(f"  2. {result.memory_reduction:.1f}% less memory")
    print(f"  3. {result.energy_savings:.1f}% energy savings")
    print(f"  4. Production-ready implementation")

    print("\n" + "=" * 70)
    print("✓ Comparison complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("  1. Review results in comparison_results/grcm_vs_transformer.json")
    print("  2. Use these metrics in licensing materials")
    print("  3. Run on real workloads for validation")
    print("  4. Benchmark on GPU for full performance analysis")


if __name__ == "__main__":
    main()
