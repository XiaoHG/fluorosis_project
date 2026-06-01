"""
Utility functions: seed setting, early stopping, config loading, class weights.
"""
import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """Set random seed for reproducibility across CPU and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(yaml_path: str):
    """Load YAML config file. Returns dict or empty dict if file missing."""
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def compute_class_weights(labels, num_classes: int):
    """Compute inverse-frequency class weights for imbalanced datasets.
    Args:
        labels: tensor of integer class labels
        num_classes: total number of classes
    Returns:
        weight tensor of shape (num_classes,), normalized to sum to num_classes
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1)
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return weights


class EarlyStopping:
    """Early stopping with patience and best model restoration.
    Monitors a metric (higher is better or lower is better).
    Saves best model checkpoint in memory and restores on early stop.
    """

    def __init__(self, patience: int = 15, mode: str = 'max', min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, metric: float, epoch: int, model: nn.Module) -> bool:
        """Check if should continue training. Returns True if early stop triggered."""
        if self.mode == 'max':
            score = metric
        else:
            score = -metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self._save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model)
        return False

    def _save_checkpoint(self, model: nn.Module):
        """Save model state dict in memory (CPU copy)."""
        self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def restore(self, model: nn.Module):
        """Restore best model weights to the model."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
