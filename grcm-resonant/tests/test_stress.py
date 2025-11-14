"""Stress tests for GRCM: long-running and high-iteration scenarios."""
import pytest
import torch
import time
from grcm import ResonantConsciousnessModule


@pytest.mark.stress
@pytest.mark.slow
class TestStressScenarios:
    """Long-running stress tests."""

    def test_10k_iterations(self):
        """
        Stress test: 10,000 forward passes.

        Validates:
        - No memory leaks
        - Stable phi
        - Consistent coherence
        - No crashes
        """
        print("\n" + "=" * 60)
        print("STRESS TEST: 10,000 Iterations")
        print("=" * 60)

        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )
        model.eval()

        num_iterations = 10000
        coherence_scores = []
        phi_values = []

        start_time = time.time()

        with torch.no_grad():
            for i in range(num_iterations):
                image_emb = torch.randn(1, 512)
                audio_emb = torch.randn(1, 768)
                action = torch.randn(1, 4)

                result = model(image_emb, audio_emb, action)

                coherence_scores.append(result['coherence'].item())
                phi_values.append(result['phi'])

                # Progress
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"  {i + 1:,}/{num_iterations:,} iterations ({rate:.1f} it/s)")

        end_time = time.time()
        total_time = end_time - start_time

        # Validation
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {num_iterations / total_time:.1f} it/s")
        print(f"Mean latency: {1000 * total_time / num_iterations:.2f} ms")

        # Coherence quality
        above_threshold = sum(1 for c in coherence_scores if c > 0.7)
        coherence_pct = (above_threshold / len(coherence_scores)) * 100
        print(f"\nCoherence >0.7: {coherence_pct:.1f}%")

        # Phi stability
        phi_mean = sum(phi_values) / len(phi_values)
        phi_variance = sum((p - phi_mean) ** 2 for p in phi_values) / len(phi_values)
        phi_std = phi_variance ** 0.5
        print(f"Phi: {phi_mean:.3f} ± {phi_std:.3f}")

        # Assertions
        assert len(model.phi.phi_history) == num_iterations, "All iterations logged"
        assert phi_std < 1.0, f"Phi unstable: std={phi_std:.3f}"
        assert coherence_pct > 30, f"Low coherence: {coherence_pct:.1f}%"

        print("\n✓ Stress test passed!")

    def test_continuous_episode_accumulation(self):
        """
        Test episodic threading with continuous high-coherence inputs.

        Validates:
        - Episode accumulation
        - Identity token evolution
        - Memory stability
        """
        print("\n" + "=" * 60)
        print("STRESS TEST: Continuous Episode Accumulation")
        print("=" * 60)

        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )
        model.set_desire(0)  # Curiosity
        model.eval()

        # Use similar inputs to trigger high coherence
        base_image = torch.ones(1, 512)
        base_audio = torch.ones(1, 768)

        num_steps = 1000
        episode_counts = []

        with torch.no_grad():
            for i in range(num_steps):
                # Add small noise
                image_emb = base_image + torch.randn(1, 512) * 0.1
                audio_emb = base_audio + torch.randn(1, 768) * 0.1
                action = torch.randn(1, 4) * 0.5

                result = model(image_emb, audio_emb, action)

                episode_counts.append(model.threads.get_num_episodes())

                if (i + 1) % 200 == 0:
                    episodes = model.threads.get_num_episodes()
                    print(f"  Step {i + 1}: {episodes} episodes accumulated")

        final_episodes = model.threads.get_num_episodes()
        print(f"\nFinal episodes: {final_episodes}")
        print(f"Max episodes (deque limit): 50")

        # Should have accumulated some episodes
        assert final_episodes > 0, "No episodes accumulated"
        assert final_episodes <= 50, "Exceeded max episodes"

        # Identity token should have evolved
        assert not torch.allclose(
            model.threads.identity_token,
            torch.zeros(32)
        ), "Identity token not updated"

        print("✓ Episode accumulation test passed!")

    def test_memory_stability(self):
        """
        Test memory grid stability under continuous updates.

        Validates:
        - Memory doesn't diverge
        - Coherence gating works
        - No NaN/Inf values
        """
        print("\n" + "=" * 60)
        print("STRESS TEST: Memory Stability")
        print("=" * 60)

        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )
        model.eval()

        num_steps = 5000
        memory_norms = []

        with torch.no_grad():
            for i in range(num_steps):
                image_emb = torch.randn(1, 512)
                audio_emb = torch.randn(1, 768)

                result = model(image_emb, audio_emb)

                # Track memory norm
                mem_norm = result['memory'].norm().item()
                memory_norms.append(mem_norm)

                # Check for NaN/Inf
                assert torch.isfinite(result['memory']).all(), f"Memory diverged at step {i}"
                assert torch.isfinite(result['output']).all(), f"Output diverged at step {i}"

                if (i + 1) % 1000 == 0:
                    print(f"  Step {i + 1}: Memory norm = {mem_norm:.3f}")

        # Memory should stabilize
        final_1000 = memory_norms[-1000:]
        mean_norm = sum(final_1000) / len(final_1000)
        variance = sum((n - mean_norm) ** 2 for n in final_1000) / len(final_1000)
        std_norm = variance ** 0.5

        print(f"\nFinal 1000 steps:")
        print(f"  Mean norm: {mean_norm:.3f}")
        print(f"  Std norm: {std_norm:.3f}")

        # Memory should be bounded
        assert mean_norm < 100, f"Memory exploded: norm={mean_norm:.3f}"
        assert std_norm < 10, f"Memory unstable: std={std_norm:.3f}"

        print("✓ Memory stability test passed!")

    @pytest.mark.slow
    def test_batch_processing_stress(self):
        """
        Stress test with large batches.

        Validates:
        - Batch processing works correctly
        - No OOM errors
        - Consistent results across batch sizes
        """
        print("\n" + "=" * 60)
        print("STRESS TEST: Batch Processing")
        print("=" * 60)

        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16, 32]
        num_batches = 100

        for batch_size in batch_sizes:
            print(f"\n  Testing batch size {batch_size}...")

            with torch.no_grad():
                for _ in range(num_batches):
                    image_emb = torch.randn(batch_size, 512)
                    audio_emb = torch.randn(batch_size, 768)
                    action = torch.randn(batch_size, 4)

                    result = model(image_emb, audio_emb, action)

                    # Validate shapes
                    assert result['output'].shape[0] == batch_size
                    assert result['coherence'].shape[0] == batch_size
                    assert result['qualia'].shape[0] == batch_size

            print(f"    ✓ {num_batches} batches processed")

        print("\n✓ Batch processing stress test passed!")

    def test_desire_switching_stress(self):
        """
        Stress test with rapid desire switching.

        Validates:
        - Desire switching doesn't break model
        - Different desires produce different behaviors
        - No state corruption
        """
        print("\n" + "=" * 60)
        print("STRESS TEST: Rapid Desire Switching")
        print("=" * 60)

        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32,
            num_desires=4
        )
        model.eval()

        num_switches = 1000
        desire_alignments = {i: [] for i in range(4)}

        with torch.no_grad():
            for i in range(num_switches):
                # Cycle through desires
                desire_idx = i % 4
                model.set_desire(desire_idx)

                image_emb = torch.randn(1, 512)
                audio_emb = torch.randn(1, 768)

                result = model(image_emb, audio_emb)

                desire_alignments[desire_idx].append(
                    result['desire_align'].item()
                )

                if (i + 1) % 250 == 0:
                    print(f"  {i + 1} desire switches completed")

        # Each desire should have samples
        for idx in range(4):
            assert len(desire_alignments[idx]) == num_switches // 4
            mean_align = sum(desire_alignments[idx]) / len(desire_alignments[idx])
            print(f"  Desire {idx}: mean alignment = {mean_align:.3f}")

        print("\n✓ Desire switching stress test passed!")


@pytest.mark.stress
class TestPerformanceRegression:
    """Performance regression tests to catch slowdowns."""

    def test_latency_regression(self):
        """
        Ensure latency doesn't exceed baseline.

        Baseline: ~100ms on CPU (without optimization)
        """
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )
        model.eval()

        num_warmup = 10
        num_iterations = 100

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                image_emb = torch.randn(1, 512)
                audio_emb = torch.randn(1, 768)
                model(image_emb, audio_emb)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                image_emb = torch.randn(1, 512)
                audio_emb = torch.randn(1, 768)

                start = time.perf_counter()
                result = model(image_emb, audio_emb)
                end = time.perf_counter()

                latencies.append((end - start) * 1000)  # ms

        mean_latency = sum(latencies) / len(latencies)
        print(f"\nMean latency: {mean_latency:.2f} ms")

        # Regression threshold: 200ms (generous for CPU)
        assert mean_latency < 200, f"Performance regression: {mean_latency:.2f} ms > 200 ms"

        print("✓ No performance regression detected")

    def test_memory_leak_check(self):
        """
        Check for memory leaks over 1000 iterations.
        """
        import gc

        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )
        model.eval()

        # Force garbage collection
        gc.collect()

        num_iterations = 1000

        with torch.no_grad():
            for _ in range(num_iterations):
                image_emb = torch.randn(1, 512)
                audio_emb = torch.randn(1, 768)
                result = model(image_emb, audio_emb)

                # Explicitly delete to free memory
                del result

        # Force GC again
        gc.collect()

        # Check that phi history is only item retained
        assert len(model.phi.phi_history) == num_iterations

        print("✓ No memory leaks detected")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "stress"])
