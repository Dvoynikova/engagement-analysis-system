from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from models.mwatt_modal_fusion import MWAttCrossModalTransformer


CLASS_NAMES = ["Low", "Medium", "High"]
VIDEO_DIM = 7
AUDIO_DIM = 3
MAX_LEN = 125


def load_multimodal_model(weights_path=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if weights_path is None:
        weights_path = (
            Path(__file__).parent
            / "weights"
            / "MWAttCrossModalTransformer_Eng_best.pt"
        )

    model = MWAttCrossModalTransformer(
        video_dim=VIDEO_DIM,
        audio_dim=AUDIO_DIM,
        d_model=512,
        nhead=1,
        num_layers=2,
        n_classes=3,
        T=MAX_LEN,
        dropout=0.3,
        learnable_M=True,
        init_type="random",
        M_mode="mul"
    ).to(device)

    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


def predict_engagement(model, video_features, audio_features, device):
    video_features = np.asarray(video_features, dtype=np.float32)
    audio_features = np.asarray(audio_features, dtype=np.float32)

    if video_features.ndim != 2 or video_features.shape[1] != VIDEO_DIM:
        raise ValueError(
            f"Ожидаются видеопризнаки формы (N, {VIDEO_DIM}), "
            f"получено {video_features.shape}"
        )

    if audio_features.ndim != 2 or audio_features.shape[1] != AUDIO_DIM:
        raise ValueError(
            f"Ожидаются аудиопризнаки формы (N, {AUDIO_DIM}), "
            f"получено {audio_features.shape}"
        )

    if len(video_features) == 0 or len(audio_features) == 0:
        raise ValueError("Последовательности признаков не должны быть пустыми")

    # Аудиопризнаки приводим к длине видеопоследовательности.
    audio_tensor = torch.from_numpy(audio_features).T.unsqueeze(0)
    audio_tensor = F.interpolate(
        audio_tensor,
        size=len(video_features),
        mode="linear",
        align_corners=False
    )
    audio_features = audio_tensor.squeeze(0).T.numpy()

    # Ограничиваем длину последовательностей.
    video_features = video_features[-MAX_LEN:]
    audio_features = audio_features[-MAX_LEN:]

    # Нормализация, как в обучающем Dataset.
    video_features = F.normalize(
        torch.from_numpy(video_features),
        p=2,
        dim=1
    )
    audio_features = F.normalize(
        torch.from_numpy(audio_features),
        p=2,
        dim=1
    )

    # Дополнение последовательностей до MAX_LEN.
    seq_len = len(video_features)

    video_x = torch.zeros(MAX_LEN, VIDEO_DIM)
    audio_x = torch.zeros(MAX_LEN, AUDIO_DIM)

    video_x[:seq_len] = video_features
    audio_x[:seq_len] = audio_features

    mask = torch.zeros(MAX_LEN)
    mask[:seq_len] = 1

    video_x = video_x.unsqueeze(0).to(device)
    audio_x = audio_x.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(video_x, audio_x, mask)
        probabilities = torch.softmax(logits, dim=1)[0]

    class_idx = int(torch.argmax(probabilities).item())

    return {
        "class": CLASS_NAMES[class_idx],
        "class_idx": class_idx,
        "probabilities": {
            name: float(probabilities[i].item())
            for i, name in enumerate(CLASS_NAMES)
        }
    }