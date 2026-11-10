
import subprocess
import numpy as np
import librosa
from ffmpeg_utils import get_ffmpeg_path

class ProsodyFeatureExtractor:
    SR = 16000
    WINDOW_DURATION = 10
    MAX_AUDIO_SAMPLES = SR * WINDOW_DURATION
    HOP_LENGTH = 160
    N_FEATURES = 3
    T_AUDIO = MAX_AUDIO_SAMPLES // HOP_LENGTH

    def load_audio(self, video_path):
        """
        Извлекает аудиодорожку из видео.
        Способ декодирования соответствует обучающему скрипту.
        """
        command = [
            get_ffmpeg_path(),
            "-i", str(video_path),
            "-f", "wav",
            "-ar", str(self.SR),
            "-ac", "1",
            "-"
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )

        audio_bytes = process.stdout

        if not audio_bytes:
            raise ValueError("Не удалось извлечь аудиодорожку")

        # Сохраняем тот же способ чтения WAV-байтов, что использовался
        # при подготовке обучающих признаков.
        audio = (
            np.frombuffer(audio_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        return audio

    def extract_features(self, audio):
        """
        Извлекает признаки просодии для одного окна длительностью 10 секунд.

        Возвращает массив формы (1000, 3):
        pitch, RMS, voiced.
        """
        audio = np.asarray(audio, dtype=np.float32)

        # Дополнение или обрезка до 10 секунд.
        if len(audio) < self.MAX_AUDIO_SAMPLES:
            audio = np.pad(
                audio,
                (0, self.MAX_AUDIO_SAMPLES - len(audio))
            )
        else:
            audio = audio[:self.MAX_AUDIO_SAMPLES]

        # Проверка общей громкости сигнала.
        global_rms = np.sqrt(np.mean(audio ** 2))

        if global_rms < 1e-4:
            return np.zeros(
                (self.T_AUDIO, self.N_FEATURES),
                dtype=np.float32
            )

        # RMS.
        rms = librosa.feature.rms(
            y=audio,
            hop_length=self.HOP_LENGTH
        )[0]

        # Частота основного тона.
        pitch, _, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=400,
            sr=self.SR,
            hop_length=self.HOP_LENGTH
        )

        pitch = np.nan_to_num(pitch)
        voiced = (pitch > 0).astype(np.float32)

        # Формируем последовательность признаков.
        min_length = min(len(pitch), len(rms), len(voiced))

        features = np.stack([
            pitch[:min_length],
            rms[:min_length],
            voiced[:min_length]
        ], axis=1)

        # Приведение к 1000 временным шагам.
        if len(features) < self.T_AUDIO:
            padding = np.zeros(
                (self.T_AUDIO - len(features), self.N_FEATURES),
                dtype=np.float32
            )
            features = np.concatenate([features, padding], axis=0)
        else:
            features = features[:self.T_AUDIO]

        return features.astype(np.float32)

    def extract_window(self, audio, start_sec=0):
        """
        Извлекает признаки из 10-секундного окна аудиодорожки.
        """
        start_sample = int(start_sec * self.SR)
        end_sample = start_sample + self.MAX_AUDIO_SAMPLES

        audio_window = audio[start_sample:end_sample]

        return self.extract_features(audio_window)

    def extract_from_video(self, video_path):
        """
        Извлекает признаки для первых 10 секунд видео.
        """
        audio = self.load_audio(video_path)
        return self.extract_window(audio, start_sec=0)
