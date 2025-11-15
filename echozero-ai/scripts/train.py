"""
Training script for EchoZero-Hybrid
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EchoZeroHybrid
from src.data import create_dataloaders
from src.losses import EchoZeroLoss
from src.utils import Logger, MetricsCalculator


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config: str):
    """Get torch device"""
    if device_config == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_config)


def train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch, cfg):
    """Train for one epoch"""
    model.train()
    metrics = MetricsCalculator(num_classes=cfg.model.num_classes)

    total_loss = 0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs, return_all_stats=False)

        # Compute loss
        loss, loss_dict = criterion(
            outputs['logits'],
            labels,
            outputs['retained_mass'],
            outputs['features'],
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if cfg.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)

        optimizer.step()

        # Update metrics
        metrics.update(outputs['logits'], labels)
        total_loss += loss.item()

        # Logging
        if batch_idx % cfg.logging.log_every == 0:
            step = epoch * num_batches + batch_idx
            logger.log_scalar('train/loss', loss.item(), step)
            logger.log_scalar('train/cls_loss', loss_dict['classification'], step)
            logger.log_scalar('train/retention', outputs['retained_mass'], step)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    train_metrics = metrics.compute_and_reset()

    return avg_loss, train_metrics


def validate(model, val_loader, criterion, device, cfg):
    """Validate model"""
    model.eval()
    metrics = MetricsCalculator(num_classes=cfg.model.num_classes)

    total_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs, return_all_stats=False)

            # Compute loss
            loss, _ = criterion(
                outputs['logits'],
                labels,
                outputs['retained_mass'],
                outputs['features'],
            )

            # Update metrics
            metrics.update(outputs['logits'], labels)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    val_metrics = metrics.compute_and_reset()

    return avg_loss, val_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Main training function"""
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Get device
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    # Create logger
    logger = Logger(
        log_dir=cfg.logging.log_dir,
        use_tensorboard=cfg.logging.use_tensorboard,
    )
    logger.info("Starting training...")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = EchoZeroHybrid(**cfg.model)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss
    criterion = EchoZeroLoss(
        num_classes=cfg.model.num_classes,
        **cfg.loss,
    )

    # Create optimizer
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # Learning rate scheduler
    if cfg.train.lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=cfg.train.lr_min,
        )
    else:
        scheduler = None

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg.train.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{cfg.train.epochs}")

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch, cfg
        )

        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, cfg)
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Log to tensorboard
        logger.log_scalar('epoch/train_loss', train_loss, epoch)
        logger.log_scalar('epoch/val_loss', val_loss, epoch)
        logger.log_scalar('epoch/train_acc', train_metrics['accuracy'], epoch)
        logger.log_scalar('epoch/val_acc', val_metrics['accuracy'], epoch)
        logger.log_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        if epoch % cfg.train.save_every == 0:
            checkpoint_path = Path(cfg.logging.log_dir) / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': OmegaConf.to_container(cfg),
            }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if cfg.train.save_best:
                best_path = Path(cfg.logging.log_dir) / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'config': OmegaConf.to_container(cfg),
                }, best_path)
                logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if cfg.train.early_stopping and patience_counter >= cfg.train.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

    logger.info("Training complete!")
    logger.close()


if __name__ == '__main__':
    main()
