
import cv2
from sources.video.base_video_source import BaseVideoSource


class FileVideoSource(BaseVideoSource):

    def __init__(self):
        self.capture = None

    def open(self, source):
        self.release()
        self.capture = cv2.VideoCapture(str(source))

        if not self.capture.isOpened():
            self.release()
            raise RuntimeError(
                f"Не удалось открыть видео: {source}"
            )

    def read(self):
        if self.capture is None:
            return False, None

        return self.capture.read()

    def is_opened(self):
        return (
            self.capture is not None
            and self.capture.isOpened()
        )

    @property
    def fps(self):
        if self.capture is None:
            return 25.0

        value = self.capture.get(cv2.CAP_PROP_FPS)
        return value if value > 0 else 25.0

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
