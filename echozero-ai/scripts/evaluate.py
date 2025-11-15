"""
Evaluation script for EchoZero-Hybrid
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EchoZeroHybrid
from src.data import create_dataloaders
from src.utils import MetricsCalculator


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Evaluate trained model"""
    # Model path from command line or default
    model_path = cfg.get('model_path', 'experiments/best_model.pt')

    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = EchoZeroHybrid(**cfg.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Val loss at save: {checkpoint.get('val_loss', 'unknown')}")

    # Create dataloaders
    _, _, test_loader = create_dataloaders(cfg)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate
    metrics = MetricsCalculator(num_classes=cfg.model.num_classes)

    all_retention_rates = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, return_all_stats=False)

            # Update metrics
            metrics.update(outputs['logits'], labels)
            all_retention_rates.append(outputs['retained_mass'])

    # Compute final metrics
    final_metrics = metrics.compute()

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_metrics['f1']:.4f}")
    print(f"Avg Retention Rate: {sum(all_retention_rates) / len(all_retention_rates):.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
