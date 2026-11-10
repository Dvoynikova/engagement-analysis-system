import soundcard as sc
import numpy as np
from PySide6.QtCore import QThread, Signal


class SystemAudioCapture(QThread):
    audio_chunk = Signal(object)
    error = Signal(str)

    def __init__(self, sample_rate=16000, chunk_duration=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.running = True

    def run(self):
        try:
            speaker = sc.default_speaker()
            loopback = sc.get_microphone(
                id=str(speaker.name),
                include_loopback=True
            )

            frames_per_chunk = int(
                self.sample_rate * self.chunk_duration
            )

            with loopback.recorder(
                samplerate=self.sample_rate
            ) as recorder:

                while self.running:
                    audio = recorder.record(
                        numframes=frames_per_chunk
                    )

                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)

                    self.audio_chunk.emit(
                        np.asarray(audio, dtype=np.float32)
                    )

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False
        self.wait(2000)