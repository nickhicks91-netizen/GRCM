"""Configuration management for GRCM."""
import yaml
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


@dataclass
class GRCMConfig:
    """Configuration for ResonantConsciousnessModule.

    All formulas and thresholds are configurable via YAML.
    """
    # Architecture
    input_dim: int = 15
    freq_dim: int = 8
    memory_size: int = 32
    clip_dim: int = 512
    wav_dim: int = 768
    prop_dim: int = 16
    qualia_dim: int = 4
    num_desires: int = 4

    # Attention
    base_bandwidth: float = 0.5
    min_bandwidth: float = 0.1
    max_bandwidth: float = 1.0
    coherence_threshold: float = 0.7

    # Desire
    desire_bandwidth_scale: float = 0.2
    desire_alignment_threshold: float = 0.5

    # Episodic
    max_episodes: int = 50
    episode_threshold: float = 0.7
    arc_bias_scale: float = 0.1

    # Phi
    phi_threshold: float = 1.5  # "Aware" state threshold

    # Qualia
    dissonance_threshold: float = 0.6  # Ethical halt trigger

    # Body
    body_mass: float = 1.0
    body_dt: float = 0.1

    # Training
    echo_lr: float = 0.01
    echo_epochs: int = 5
    echo_phi_weight: float = 0.01

    @classmethod
    def from_yaml(cls, path: str) -> 'GRCMConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @staticmethod
    def get_default_config() -> 'GRCMConfig':
        """Get default configuration."""
        return GRCMConfig()


def load_config(path: Optional[str] = None) -> GRCMConfig:
    """
    Load configuration from YAML file or return defaults.

    Args:
        path: Path to YAML config file. If None, uses defaults.

    Returns:
        GRCMConfig instance
    """
    if path is None:
        return GRCMConfig.get_default_config()
    return GRCMConfig.from_yaml(path)
