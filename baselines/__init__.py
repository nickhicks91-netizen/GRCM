"""
Baseline Models for Comparison
Standard transformer and attention baselines for benchmarking GRCM efficiency
"""
from .transformer_baseline import TransformerBaseline, BaselineConfig

__all__ = ['TransformerBaseline', 'BaselineConfig']
