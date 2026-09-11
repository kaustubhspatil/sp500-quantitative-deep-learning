"""Model architectures for the capstone."""

import torch
import torch.nn as nn


class StockPredictor(nn.Module):
    """
    Feedforward network for cross-sectional next-day return prediction.
    Each hidden block: Linear -> BatchNorm -> ReLU -> Dropout.
    """

    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FinancialLSTM(nn.Module):
    """
    Two-layer LSTM over a (batch, seq_len, features) window; the hidden state
    of the final timestep feeds a small FC head.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        return self.head(out[:, -1, :])  # last timestep only


class FinancialCNN1D(nn.Module):
    """
    1D CNN over the same windows. Conv1d expects (batch, channels, length),
    so the forward pass permutes from the dataset's (batch, length, features).
    """

    def __init__(self, input_channels, dropout=0.3):
        super().__init__()
        def block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2),
            )
        self.features = nn.Sequential(
            block(input_channels, 32, 7), block(32, 64, 5), block(64, 128, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, len, feat) -> (batch, feat, len)
        return self.head(self.pool(self.features(x)))


def build_resnet18(mode="feature_extraction", num_classes=2):
    """
    ImageNet-pretrained ResNet-18 for candlestick chart classification.

    mode="feature_extraction": freeze everything, train only the new FC head.
    mode="fine_tune":          also unfreeze the last two residual stages.
    """
    from torchvision import models

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    if mode == "fine_tune":
        for stage in (model.layer3, model.layer4):
            for p in stage.parameters():
                p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head - trainable
    return model


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)
