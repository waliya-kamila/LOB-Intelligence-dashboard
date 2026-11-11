import torch
import torch.nn as nn


class MultiHorizonLoss(nn.Module):

    def __init__(self, class_weights):
        super().__init__()

        if class_weights.shape != (5, 3):
            raise ValueError("Expected class_weights shape (5, 3)")

        self.register_buffer(
            "class_weights",
            class_weights.float()
        )

    def forward(self, predictions, targets):

        if predictions.ndim != 3 or predictions.shape[1:] != (5, 3):
            raise ValueError("Predictions must have shape (B, 5, 3)")

        if targets.ndim != 2 or targets.shape[1] != 5:
            raise ValueError("Targets must have shape (B, 5)")

        if predictions.shape[0] != targets.shape[0]:
            raise ValueError("Batch sizes do not match")

        if targets.dtype != torch.long:
            raise ValueError("Targets must have dtype torch.long")

        if targets.numel() > 0:
            if targets.min() < 0 or targets.max() > 2:
                raise ValueError("Targets must contain labels 0, 1, 2")

        horizon_losses = []

        for horizon in range(5):

            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights[horizon]
            )

            loss = criterion(
                predictions[:, horizon, :],
                targets[:, horizon]
            )

            horizon_losses.append(loss)

        return torch.stack(horizon_losses).mean()