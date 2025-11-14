"""Prometheus metrics exporter for GRCM."""
import warnings
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("Prometheus client not installed. Install with: pip install prometheus-client")

import time


class GRCMMetricsExporter:
    """
    Prometheus metrics exporter for GRCM.

    Exports:
        - Request counters (total, errors, halts)
        - Latency histograms
        - Model metrics (phi, coherence, qualia)
        - System metrics (memory usage, throughput)
    """

    def __init__(self, registry: Optional['CollectorRegistry'] = None):
        """Initialize metrics exporter."""
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus client not installed")

        self.registry = registry or CollectorRegistry()

        # Request metrics
        self.request_total = Counter(
            'grcm_requests_total',
            'Total number of requests',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.request_errors = Counter(
            'grcm_errors_total',
            'Total number of errors',
            ['error_type'],
            registry=self.registry
        )

        self.ethical_halts = Counter(
            'grcm_ethical_halts_total',
            'Total number of ethical halts (dissonance)',
            registry=self.registry
        )

        # Latency metrics
        self.request_duration = Histogram(
            'grcm_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )

        # Model quality metrics
        self.phi_value = Gauge(
            'grcm_phi',
            'Current Phi (integrated information) value',
            registry=self.registry
        )

        self.phi_mean = Gauge(
            'grcm_phi_mean',
            'Mean Phi over recent window',
            registry=self.registry
        )

        self.phi_std = Gauge(
            'grcm_phi_std',
            'Phi standard deviation (stability)',
            registry=self.registry
        )

        self.coherence_mean = Gauge(
            'grcm_coherence_mean',
            'Mean coherence score',
            registry=self.registry
        )

        self.coherence_below_threshold = Gauge(
            'grcm_coherence_below_threshold_percentage',
            'Percentage of samples with coherence < 0.7',
            registry=self.registry
        )

        # Qualia metrics
        self.qualia_calm = Gauge(
            'grcm_qualia_calm',
            'Qualia calm state probability',
            registry=self.registry
        )

        self.qualia_alert = Gauge(
            'grcm_qualia_alert',
            'Qualia alert state probability',
            registry=self.registry
        )

        self.qualia_curious = Gauge(
            'grcm_qualia_curious',
            'Qualia curious state probability',
            registry=self.registry
        )

        self.qualia_conflicted = Gauge(
            'grcm_qualia_conflicted',
            'Qualia conflicted state probability (dissonance)',
            registry=self.registry
        )

        # Desire metrics
        self.desire_align_mean = Gauge(
            'grcm_desire_align_mean',
            'Mean desire alignment',
            registry=self.registry
        )

        self.current_desire = Gauge(
            'grcm_current_desire_idx',
            'Current desire state index',
            registry=self.registry
        )

        # Memory metrics
        self.memory_norm = Gauge(
            'grcm_memory_norm',
            'Memory state norm (magnitude)',
            registry=self.registry
        )

        self.episode_count = Gauge(
            'grcm_episode_count',
            'Number of accumulated episodes',
            registry=self.registry
        )

        # System metrics
        self.throughput = Summary(
            'grcm_throughput_samples_per_second',
            'Throughput in samples per second',
            registry=self.registry
        )

    def record_request(self, method: str, endpoint: str):
        """Record a request."""
        self.request_total.labels(method=method, endpoint=endpoint).inc()

    def record_error(self, error_type: str):
        """Record an error."""
        self.request_errors.labels(error_type=error_type).inc()

    def record_ethical_halt(self):
        """Record an ethical halt."""
        self.ethical_halts.inc()

    def record_latency(self, endpoint: str, duration: float):
        """Record request latency."""
        self.request_duration.labels(endpoint=endpoint).observe(duration)

    def update_model_metrics(self, result: dict):
        """
        Update all model metrics from a forward pass result.

        Args:
            result: Dictionary from GRCM forward pass
        """
        # Phi
        self.phi_value.set(result['phi'])

        # Coherence
        coherence = result['coherence'].mean().item()
        self.coherence_mean.set(coherence)

        # Qualia
        qualia = result['qualia'][0].cpu().numpy()
        self.qualia_calm.set(float(qualia[0]))
        self.qualia_alert.set(float(qualia[1]))
        self.qualia_curious.set(float(qualia[2]))
        self.qualia_conflicted.set(float(qualia[3]))

        # Desire alignment
        desire_align = result['desire_align'].mean().item()
        self.desire_align_mean.set(desire_align)

        # Memory
        memory_norm = result['memory'].norm().item()
        self.memory_norm.set(memory_norm)

        # Ethical halt
        if result['halt']:
            self.record_ethical_halt()

    def update_statistics(self, model):
        """
        Update statistical metrics from model state.

        Args:
            model: ResonantConsciousnessModule instance
        """
        metrics = model.get_metrics()

        self.phi_mean.set(metrics['phi_mean'])
        self.phi_std.set(metrics['phi_std'])
        self.current_desire.set(metrics['current_desire'])
        self.episode_count.set(metrics['num_episodes'])

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)


# Global metrics instance
_metrics_exporter: Optional[GRCMMetricsExporter] = None


def get_metrics_exporter() -> GRCMMetricsExporter:
    """Get or create global metrics exporter."""
    global _metrics_exporter
    if _metrics_exporter is None:
        _metrics_exporter = GRCMMetricsExporter()
    return _metrics_exporter


if __name__ == "__main__":
    print("GRCM Prometheus Metrics Exporter")
    print("=" * 50)

    if PROMETHEUS_AVAILABLE:
        print("\n✓ Prometheus client available")
        print("\nUsage:")
        print("  from grcm.prometheus_metrics import get_metrics_exporter")
        print("  exporter = get_metrics_exporter()")
        print("  exporter.update_model_metrics(result)")
    else:
        print("\n✗ Prometheus client not installed")
        print("  Install with: pip install prometheus-client")
