"""
Unit tests for models
"""
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    EchoZeroHybrid,
    PurposeGate,
    RetentionReservoir,
    TokenEnhancedAttention,
    RTF,
    EchoCore,
)


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 128


@pytest.fixture
def dim():
    return 64


def test_purpose_gate(device, batch_size, seq_len, dim):
    """Test PurposeGate module"""
    model = PurposeGate(dim=dim).to(device)
    x = torch.randn(batch_size, seq_len, dim).to(device)

    output, stats = model(x, return_stats=True)

    assert output.shape == x.shape
    assert 'retention_rate' in stats
    assert 0 <= stats['retention_rate'] <= 1


def test_retention_reservoir(device, batch_size, seq_len, dim):
    """Test RetentionReservoir module"""
    reservoir_size = 128
    model = RetentionReservoir(dim=dim, reservoir_size=reservoir_size).to(device)
    x = torch.randn(batch_size, seq_len, dim).to(device)

    output, state, stats = model(x)

    assert output.shape == x.shape
    assert state.shape == (batch_size, reservoir_size)
    assert 'state_norm' in stats


def test_tea(device, batch_size, seq_len, dim):
    """Test TokenEnhancedAttention module"""
    model = TokenEnhancedAttention(dim=dim, num_heads=4).to(device)
    x = torch.randn(batch_size, seq_len, dim).to(device)

    output, attn = model(x)

    assert output.shape == x.shape
    assert attn.shape[0] == batch_size


def test_rtf(device, batch_size, seq_len, dim):
    """Test RTF module"""
    model = RTF(dim=dim, depth=2, num_heads=4).to(device)
    x = torch.randn(batch_size, seq_len, dim).to(device)

    output, states = model(x)

    assert output.shape == x.shape
    assert len(states) == 2


def test_echo_core(device, batch_size, seq_len, dim):
    """Test EchoCore module"""
    reservoir_dim = 256
    model = EchoCore(input_dim=dim, reservoir_dim=reservoir_dim).to(device)
    x = torch.randn(batch_size, seq_len, dim).to(device)

    output, state, stats = model(x)

    assert output.shape == x.shape
    assert state.shape == (batch_size, reservoir_dim)
    assert 'spectral_radius' in stats


def test_echozero_hybrid_forward(device, batch_size, seq_len):
    """Test EchoZeroHybrid forward pass"""
    model = EchoZeroHybrid(
        dim=64,
        num_classes=2,
        reservoir_dim=128,
        reservoir_size=64,
    ).to(device)

    x = torch.randn(batch_size, seq_len).to(device)

    output = model(x)

    assert 'logits' in output
    assert 'features' in output
    assert 'retained_mass' in output
    assert 'states' in output

    assert output['logits'].shape == (batch_size, 2)
    assert output['features'].shape == (batch_size, 64)
    assert isinstance(output['retained_mass'], float)


def test_echozero_hybrid_with_states(device, batch_size, seq_len):
    """Test EchoZeroHybrid with recurrent states"""
    model = EchoZeroHybrid(dim=64, num_classes=2).to(device)

    x = torch.randn(batch_size, seq_len).to(device)

    # First pass
    output1 = model(x)
    states1 = output1['states']

    # Second pass with states
    output2 = model(
        x,
        echo_state=states1['echo'],
        retention_state=states1['retention'],
        rtf_states=states1['rtf'],
    )

    assert output2['logits'].shape == (batch_size, 2)


def test_echozero_hybrid_ablation(device, batch_size, seq_len):
    """Test EchoZeroHybrid with different component configurations"""
    configs = [
        {'use_purpose_gate': False},
        {'use_echo_core': False},
        {'use_retention': False},
        {'use_tea': False},
        {'use_rtf': False},
    ]

    x = torch.randn(batch_size, seq_len).to(device)

    for config in configs:
        model = EchoZeroHybrid(dim=64, num_classes=2, **config).to(device)
        output = model(x)
        assert output['logits'].shape == (batch_size, 2)
