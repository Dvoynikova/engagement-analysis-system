
from sources.audio.file_audio_source import FileAudioSource


class AudioManager:

    def __init__(self):
        self.source = None
        self.audio_data = None

    def load_file(self, file_path):
        self.clear()

        self.source = FileAudioSource()
        self.audio_data = self.source.load(file_path)

        print(
            f"Аудио загружено: "
            f"{self.duration:.2f} сек."
        )

        return self.audio_data

    @property
    def sample_rate(self):
        if self.source is None:
            return None

        return self.source.sample_rate

    @property
    def duration(self):
        if self.audio_data is None:
            return 0.0

        return len(self.audio_data) / self.sample_rate

    def get_audio_data(self):
        return self.audio_data

    def clear(self):
        if self.source is not None:
            self.source.close()

        self.source = None
        self.audio_data = None
