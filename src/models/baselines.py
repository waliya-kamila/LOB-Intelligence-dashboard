import torch
import torch.nn as nn


class CNNOnly(nn.Module):

    def __init__(
        self,
        input_features=144,
        channels=64,
        num_horizons=5,
        num_classes=3,
        dropout=0.2
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_horizons * num_classes)
        )

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.mean(dim=2)

        logits = self.classifier(x)

        return logits.reshape(x.shape[0], 5, 3), None


class GRUOnly(nn.Module):

    def __init__(
        self,
        input_features=144,
        hidden=64,
        num_horizons=5,
        num_classes=3,
        dropout=0.2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_horizons * num_classes)
        )

    def forward(self, x):

        x, _ = self.gru(x)
        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits.reshape(x.shape[0], 5, 3), None


class CNNGRUWithoutAttention(nn.Module):

    def __init__(
        self,
        input_features=144,
        channels=64,
        hidden=64,
        num_horizons=5,
        num_classes=3,
        dropout=0.2
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_horizons * num_classes)
        )

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.gru(x)
        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits.reshape(x.shape[0], 5, 3), None
