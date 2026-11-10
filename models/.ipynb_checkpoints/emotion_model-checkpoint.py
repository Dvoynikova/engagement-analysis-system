from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.mwatt_modal_fusion import MWAttCrossModalTransformer


MAX_LEN = 125
FEATURE_DIM = 768

ACTIVATION_LABELS = ["Low", "Medium", "High"]
VALENCE_LABELS = ["Negative", "Neutral", "Positive"]


def load_emotion_model(
    d_model,
    init_type,
    M_mode,
    checkpoint_path,
    device
):
    model = MWAttCrossModalTransformer(
        video_dim=768,
        audio_dim=768,
        d_model=d_model,
        nhead=1,
        num_layers=2,
        n_classes=3,
        T=125,
        dropout=0.3,
        learnable_M=True,
        init_type=init_type,
        M_mode=M_mode
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    # Сопоставляем названия слоёв обученной модели
    # с названиями слоёв модели в приложении
    key_mapping = {
        "embedding.0.weight": "head.0.weight",
        "embedding.0.bias": "head.0.bias",
        "classifier.1.weight": "head.3.weight",
        "classifier.1.bias": "head.3.bias"
    }

    model_state = model.state_dict()

    for old_key, new_key in key_mapping.items():
        if old_key in checkpoint:
            if checkpoint[old_key].shape != model_state[new_key].shape:
                raise ValueError(
                    f"Несовпадение размерностей {old_key} и {new_key}: "
                    f"{checkpoint[old_key].shape} != {model_state[new_key].shape}"
                )

            checkpoint[new_key] = checkpoint.pop(old_key)

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    return model


def load_emotion_models(device="cuda"):
    device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    weights_dir = Path("models/weights")

    activation_model = load_emotion_model(
        checkpoint_path=weights_dir / "MWAttCrossModalTransformer_Arousal_best.pt",
        d_model=512,
        init_type="uniform",
        M_mode="mul",
        device=device
    )

    valence_model = load_emotion_model(
        checkpoint_path=weights_dir / "MWAttCrossModalTransformer_Valence_best.pt",
        d_model=512,
        init_type="random",
        M_mode="none",
        device=device
    )

    return activation_model, valence_model, device


@torch.no_grad()
def predict_emotion(
    model,
    video_features,
    audio_features,
    device,
    labels
):
    """
    video_features: (125, 768)
    audio_features: (1000, 768)
    """

    video = torch.as_tensor(
        video_features,
        dtype=torch.float32,
        device=device
    )

    audio = torch.as_tensor(
        audio_features,
        dtype=torch.float32,
        device=device
    )

    # L2-нормализация каждой временной позиции
    video = F.normalize(video, p=2, dim=-1)
    audio = F.normalize(audio, p=2, dim=-1)

    # Интерполяция аудио: 1000 -> 125
    audio = audio.transpose(0, 1).unsqueeze(0)

    audio = F.interpolate(
        audio,
        size=MAX_LEN,
        mode="linear",
        align_corners=False
    )

    audio = audio.squeeze(0).transpose(0, 1)

    # Ограничение длины видеопоследовательности
    if video.size(0) < MAX_LEN:
        pad = torch.zeros(
            MAX_LEN - video.size(0),
            FEATURE_DIM,
            device=device
        )
        video = torch.cat([video, pad], dim=0)
    else:
        video = video[:MAX_LEN]

    # Маска: все 125 временных позиций валидны
    mask = torch.ones(
        1,
        MAX_LEN,
        dtype=torch.float32,
        device=device
    )

    video = video.unsqueeze(0)
    audio = audio.unsqueeze(0)

    logits = model(video, audio, mask)
    probabilities = torch.softmax(logits, dim=-1)[0]

    class_idx = int(torch.argmax(probabilities).item())
    predicted_class = labels[class_idx]

    probabilities_dict = {
        label: float(probabilities[i].item())
        for i, label in enumerate(labels)
    }

    return {
        "class": predicted_class,
        "class_idx": class_idx,
        "probabilities": probabilities_dict
    }