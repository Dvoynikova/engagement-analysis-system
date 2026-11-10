
from abc import ABC, abstractmethod


class BaseVideoSource(ABC):

    @abstractmethod
    def open(self, source):
        """Открывает источник видео."""
        pass

    @abstractmethod
    def read(self):
        """Возвращает следующий кадр."""
        pass

    @abstractmethod
    def is_opened(self):
        """Проверяет, открыт ли источник."""
        pass

    @abstractmethod
    def release(self):
        """Освобождает ресурсы."""
        pass

    @property
    @abstractmethod
    def fps(self):
        """Возвращает частоту кадров."""
        pass
