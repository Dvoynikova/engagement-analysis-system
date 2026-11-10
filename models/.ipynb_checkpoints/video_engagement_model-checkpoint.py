import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование."""

    def __init__(self, d_model, max_len=200):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer-блок, соответствующий обученной архитектуре."""

    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.5):
        super().__init__()

        self.attn1 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        ff_dim = dim_feedforward or d_model * 4

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out1, _ = self.attn1(x, x, x)
        x = self.norm1(x + attn_out1)

        attn_out2, _ = self.attn2(x, x, x)
        x = self.norm2(x + attn_out2)

        # Feed-forward слой не используется в исходной модели
        return x


class VideoEngagementModel(nn.Module):
    """Классификатор вовлечённости по последовательности видеопризнаков."""

    def __init__(self, feat_dim=768, n_classes=3, T=125,
                 nhead=8, dropout=0.5):
        super().__init__()

        self.pos_enc = PositionalEncoding(feat_dim, max_len=T)

        self.transformer1 = TransformerBlock(
            feat_dim, nhead, dropout=dropout
        )
        self.transformer2 = TransformerBlock(
            feat_dim, nhead, dropout=dropout
        )

        self.conv1 = nn.Conv1d(
            feat_dim, feat_dim, kernel_size=1
        )

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.Hardtanh(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self.attn_pool = nn.Linear(feat_dim, 1)

    def forward(self, x):
        # x: (B, T, 768)
        x = self.pos_enc(x)
        x = self.transformer1(x)
        x = self.transformer2(x)

        # Conv1d ожидает (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.mean(dim=2)

        # Выход — три логита: Low, Medium, High
        return self.fc(x)


CLASS_NAMES = ["Low", "Medium", "High"]
FEATURE_DIM = 768
MAX_LEN = 125


def load_engagement_model(weights_path=None, device=None):
    """Загружает модель и сохранённые обученные веса."""

    if device is None:
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"

    device = torch.device(device)

    if weights_path is None:
        weights_path = (
            Path(__file__).resolve().parent
            / "weights"
            / "VideoEngagementModel_best.pt"
        )
    else:
        weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Не найден файл весов модели: {weights_path}"
        )

    model = VideoEngagementModel(
        feat_dim=FEATURE_DIM,
        n_classes=len(CLASS_NAMES),
        T=MAX_LEN,
        nhead=8,
        dropout=0.5,
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


def predict_engagement(model, features, device):
    """
    Оценивает вовлечённость по последовательности признаков.

    features: массив формы (T, 768).
    Возвращает предсказанный класс и вероятности классов.
    """

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Ожидается массив формы (T, {FEATURE_DIM}), "
            f"получена форма {features.shape}"
        )

    # Приводим последовательность к длине 125
    if len(features) > MAX_LEN:
        features = features[:MAX_LEN]
    elif len(features) < MAX_LEN:
        padding = np.zeros(
            (MAX_LEN - len(features), FEATURE_DIM),
            dtype=np.float32,
        )
        features = np.concatenate([features, padding], axis=0)

    # Нормализацию не выполняем: при обучении она не использовалась
    x = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probabilities).item())

    return {
        "class": CLASS_NAMES[predicted_idx],
        "class_idx": predicted_idx,
        "probabilities": {
            name: float(probabilities[i].item())
            for i, name in enumerate(CLASS_NAMES)
        },
    }