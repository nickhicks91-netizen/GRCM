"""
Unit tests for data loading
"""
import pytest
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DEAPDataset, SyntheticDataset


def test_deap_dataset():
    """Test DEAP dataset"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = DEAPDataset(
            data_dir=tmpdir,
            split='train',
            seq_len=256,
            num_channels=1,
            use_synthetic=True,
        )

        assert len(dataset) == 1000

        signal, label = dataset[0]
        assert signal.shape == (256,)
        assert label in [0, 1]


def test_deap_dataset_multichannel():
    """Test DEAP dataset with multiple channels"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = DEAPDataset(
            data_dir=tmpdir,
            split='train',
            seq_len=256,
            num_channels=32,  # Multi-channel EEG
            use_synthetic=True,
        )

        signal, label = dataset[0]
        assert signal.shape == (256, 32)


def test_synthetic_dataset():
    """Test synthetic dataset"""
    dataset = SyntheticDataset(
        num_samples=100,
        seq_len=128,
        num_channels=1,
        num_classes=2,
    )

    assert len(dataset) == 100

    signal, label = dataset[0]
    assert signal.shape == (128, 1)
    assert label in [0, 1]


def test_dataset_reproducibility():
    """Test that dataset is reproducible with same index"""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset1 = DEAPDataset(tmpdir, split='train', use_synthetic=True)
        dataset2 = DEAPDataset(tmpdir, split='train', use_synthetic=True)

        signal1, label1 = dataset1[42]
        signal2, label2 = dataset2[42]

        assert (signal1 == signal2).all()
        assert label1 == label2
