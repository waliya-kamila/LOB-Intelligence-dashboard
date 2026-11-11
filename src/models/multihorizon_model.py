import torch
import torch.nn as nn


class TemporalCNNGRUAttention(nn.Module):

    def __init__(
        self,
        input_features=144,
        cnn_channels=64,
        gru_hidden=64,
        num_horizons=5,
        num_classes=3,
        dropout=0.2
    ):
        super().__init__()

        self.input_features = input_features
        self.num_horizons = num_horizons
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, cnn_channels, 5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, 3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU()
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True
        )

        gru_output_size = gru_hidden * 2

        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_horizons * num_classes)
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError("Input must have shape (B, T, F)")

        if x.shape[-1] != self.input_features:
            raise ValueError("Incorrect number of input features")

        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.gru(x)

        attention_scores = self.attention(x).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(
            x * attention_weights.unsqueeze(-1),
            dim=1
        )

        context = self.dropout(context)
        logits = self.classifier(context)

        logits = logits.reshape(
            x.shape[0],
            self.num_horizons,
            self.num_classes
        )

        return logits, attention_weights
