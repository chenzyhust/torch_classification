import torch
import torch.nn as nn
import torch.nn.functional as F

class MixLoss:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, preds, labels):
        label_a, label_b , lam = labels
        return lam * self.criterion(preds, label_a) + (1 - lam) * self.criterion(preds, label_b)

class RicapLoss:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
    
    def __call__(self, preds, labels):
        labels_list, weights = labels
        return sum([
            weight * self.criterion(preds, labels)
            for labels, weight in zip(labels_list, weights)
        ])

class DualCutoutLoss:
    def __init__(self, alpha):
        self.alpha = alpha
        self.loss_func = nn.CrossEntropyLoss()

    def __call__(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        predictions1, predictions2 = predictions[:, 0], predictions[:, 1]
        return (self.loss_func(predictions1, targets) + self.loss_func(
            predictions2, targets)) * 0.5 + self.alpha * F.mse_loss(
                predictions1, predictions2)