import cv2
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from pathlib import Path

class ViTAffectNetExtractor:
    def __init__(self, model_name=None, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if model_name is None:
            model_name = (
                Path(__file__).resolve().parent.parent
                / "models" / "pretrained" / "vit-face-affectnet"
            )

        self.processor = ViTImageProcessor.from_pretrained(
            model_name, local_files_only=True
        )
        self.model = ViTForImageClassification.from_pretrained(
            model_name, local_files_only=True
        )

        self.model.config.output_hidden_states = True
        self.model.to(self.device)
        self.model.eval()

        self.feature_dim = self.model.config.hidden_size

    @torch.no_grad()
    def extract(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # OpenCV использует BGR, ViT обучался на RGB
        image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = Image.fromarray(image)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)
        features = outputs.hidden_states[-1][:, 0]

        return features.squeeze(0).cpu().numpy().astype(np.float32)