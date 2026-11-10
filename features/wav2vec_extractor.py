import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path

class Wav2VecExtractor:
    def __init__(
        self,
        model_name=None,
        sample_rate=16000,
        duration=10,
        target_length=1000,
        device=None
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.sample_rate = sample_rate
        self.duration = duration
        self.max_samples = sample_rate * duration
        self.target_length = target_length

        if model_name is None:
            model_name = (
                Path(__file__).resolve().parent.parent
                / "models" / "pretrained" / "wav2vec2-base"
            )

        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name, local_files_only=True
        )
        self.model = Wav2Vec2Model.from_pretrained(
            model_name, local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()

        self.feature_dim = self.model.config.hidden_size

    def _prepare_audio(self, audio):
        audio = np.asarray(audio).reshape(-1)
    
        if np.issubdtype(audio.dtype, np.integer) or (
            audio.size > 0 and np.max(np.abs(audio)) > 1.0
        ):
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
    
        if len(audio) < self.max_samples:
            audio = np.pad(
                audio,
                (0, self.max_samples - len(audio))
            )
        else:
            audio = audio[:self.max_samples]
    
        return audio

    @torch.no_grad()
    def extract(self, audio):
        audio = self._prepare_audio(audio)

        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 1e-4:
            return np.zeros(
                (self.target_length, self.feature_dim),
                dtype=np.float32
            )

        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)
        outputs = self.model(input_values)

        features = outputs.last_hidden_state

        # (1, T, 768) -> (1, 768, T)
        features = features.transpose(1, 2)

        features = F.interpolate(
            features,
            size=self.target_length,
            mode="linear",
            align_corners=False
        )

        # (1, 768, 1000) -> (1000, 768)
        features = features.squeeze(0).transpose(0, 1)

        return features.cpu().numpy().astype(np.float32)

    def extract_window(self, audio_data, start_sec=0):
        sample_rate = self.sample_rate
        start_sample = int(start_sec * sample_rate)
        end_sample = start_sample + self.max_samples
    
        audio_window = audio_data[start_sample:end_sample]
    
        return self.extract(audio_window)