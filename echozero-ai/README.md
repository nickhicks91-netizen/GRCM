# EchoZero-Hybrid

**Purpose-Gated Retention Reservoir Architecture for Efficient Neural Processing**

## Overview

EchoZero-Hybrid combines:
- **Purpose Gates**: Adaptive information filtering and routing
- **Retention Reservoirs**: Efficient memory mechanisms with controlled capacity
- **TEA (Token-Enhanced Attention)**: Advanced attention mechanisms
- **RTF (Retention Transformer Features)**: Hybrid retention-transformer architecture
- **Echo Core**: Echo state network foundations

## Installation

```bash
pip install -e .
```

## Quick Start

### Training

```bash
python scripts/train.py
```

### Evaluation

```bash
python scripts/evaluate.py model_path=experiments/best_model.pt
```

### Benchmarking

```bash
python scripts/benchmark.py
```

### Export

```bash
python scripts/export.py
```

## Architecture

The EchoZero-Hybrid model processes sequences through:

1. **Purpose Gate**: Filters input based on learned importance
2. **Echo Core**: Processes through echo state reservoir
3. **Retention Reservoir**: Maintains compressed memory state
4. **TEA Module**: Applies token-enhanced attention
5. **RTF Module**: Retention-transformer hybrid processing

## Configuration

Edit `configs/train.yaml` for training parameters and `configs/data/deap.yaml` for dataset configuration.

## Testing

```bash
pytest
```

## Docker

```bash
docker build -t echozero-ai .
docker run -it echozero-ai
```

## License

MIT
