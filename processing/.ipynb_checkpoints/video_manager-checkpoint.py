
from sources.video.file_video_source import FileVideoSource


class VideoManager:

    def __init__(self):
        self.source = None

    def open_file(self, file_path):
        self.release()

        self.source = FileVideoSource()
        self.source.open(file_path)

    def read(self):
        if self.source is None:
            return False, None

        return self.source.read()

    @property
    def fps(self):
        if self.source is None:
            return 25.0

        return self.source.fps

    def is_opened(self):
        return (
            self.source is not None
            and self.source.is_opened()
        )

    def release(self):
        if self.source is not None:
            self.source.release()
            self.source = None
