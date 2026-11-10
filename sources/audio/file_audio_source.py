
import subprocess
import numpy as np

from sources.audio.base_audio_source import BaseAudioSource
from ffmpeg_utils import get_ffmpeg_path

class FileAudioSource(BaseAudioSource):

    SR = 16000

    @property
    def sample_rate(self):
        return self.SR

    def load(self, source):

        command = [
            get_ffmpeg_path(),
            "-i", str(source),
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
            raise ValueError(
                "Не удалось извлечь аудиодорожку"
            )

        audio = (
            np.frombuffer(audio_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        return audio

    def close(self):
        pass
