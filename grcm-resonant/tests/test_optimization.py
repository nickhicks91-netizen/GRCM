"""Tests for optimization utilities."""
import pytest
import torch
import tempfile
import os
from grcm import ResonantConsciousnessModule

try:
    from grcm.optimization import OptimizedGRCM, export_to_onnx, test_onnx_inference
    from grcm.benchmark import GRCMBenchmark, BenchmarkResult
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False


@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Optimization modules not available")
class TestOptimizedGRCM:
    """Tests for OptimizedGRCM wrapper."""

    def test_basic_optimization(self):
        """Test basic OptimizedGRCM creation."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        optimized = OptimizedGRCM(model, mode='default', quantize=False)

        assert optimized is not None
        info = optimized.get_optimization_info()
        assert 'compiled' in info
        assert 'device' in info

    def test_optimized_forward(self):
        """Test forward pass with optimization."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        optimized = OptimizedGRCM(model, mode='default', quantize=False)

        image_emb = torch.randn(1, 512)
        audio_emb = torch.randn(1, 768)
        action = torch.tensor([[0.1, 0.2, 0.0, 0.0]])

        result = optimized(image_emb, audio_emb, action)

        # Check all outputs present
        assert 'output' in result
        assert 'coherence' in result
        assert 'phi' in result
        assert 'qualia' in result

    @pytest.mark.skipif(torch.cuda.is_available(), reason="Quantization only on CPU")
    def test_quantization(self):
        """Test INT8 quantization (CPU only)."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        # Apply quantization
        optimized = OptimizedGRCM(model, mode='default', quantize=True)

        # Test inference
        image_emb = torch.randn(1, 512)
        audio_emb = torch.randn(1, 768)

        result = optimized(image_emb, audio_emb)

        assert result['coherence'].shape == (1, 1)


@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Optimization modules not available")
class TestONNXExport:
    """Tests for ONNX export."""

    def test_export_basic(self):
        """Test basic ONNX export."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            export_to_onnx(
                model,
                output_path=onnx_path,
                opset_version=18,
                dynamic_batch=True,
                simplify=False
            )

            # Check file exists and has reasonable size
            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 1000  # At least 1KB

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)

    @pytest.mark.skipif(True, reason="ONNXRuntime may not be installed")
    def test_onnx_inference(self):
        """Test ONNX inference (requires onnxruntime)."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            export_to_onnx(model, output_path=onnx_path)
            test_onnx_inference(onnx_path, num_samples=2)

        except ImportError:
            pytest.skip("onnxruntime not installed")
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Optimization modules not available")
class TestBenchmark:
    """Tests for benchmarking utilities."""

    def test_benchmark_creation(self):
        """Test GRCMBenchmark creation."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        benchmark = GRCMBenchmark(model, warmup_iterations=2)
        assert benchmark is not None

    def test_latency_benchmark(self):
        """Test latency benchmarking."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        benchmark = GRCMBenchmark(model, warmup_iterations=2)

        result = benchmark.benchmark_latency(
            batch_size=1,
            num_iterations=10,
            name="Test",
            optimization="none"
        )

        assert isinstance(result, BenchmarkResult)
        assert result.mean_latency_ms > 0
        assert result.throughput_samples_per_sec > 0
        assert result.num_iterations == 10

    def test_coherence_quality(self):
        """Test coherence quality benchmarking."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        benchmark = GRCMBenchmark(model)

        metrics = benchmark.benchmark_coherence_quality(num_samples=20)

        assert 'mean_coherence' in metrics
        assert 'std_coherence' in metrics
        assert 'percentage_above_threshold' in metrics
        assert 0 <= metrics['mean_coherence'] <= 1

    def test_phi_stability(self):
        """Test phi stability benchmarking."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        benchmark = GRCMBenchmark(model)

        metrics = benchmark.benchmark_phi_stability(num_iterations=20)

        assert 'mean_phi' in metrics
        assert 'std_phi' in metrics
        assert 'stable' in metrics
        assert isinstance(metrics['stable'], bool)

    def test_memory_usage(self):
        """Test memory usage estimation."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        benchmark = GRCMBenchmark(model)

        metrics = benchmark.memory_usage()

        assert 'total_params' in metrics
        assert 'param_memory_mb' in metrics
        assert metrics['total_params'] > 0
        assert metrics['param_memory_mb'] > 0

    def test_batch_sizes(self):
        """Test benchmarking across batch sizes."""
        model = ResonantConsciousnessModule(
            input_dim=15,
            freq_dim=8,
            memory_size=32
        )

        benchmark = GRCMBenchmark(model, warmup_iterations=2)

        results = benchmark.benchmark_batch_sizes(
            batch_sizes=[1, 2],
            num_iterations=5,
            optimization="test"
        )

        assert len(results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in results)
