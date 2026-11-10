
from abc import ABC, abstractmethod
import numpy as np


class BaseAudioSource(ABC):

    @abstractmethod
    def load(self, source) -> np.ndarray:
        """Загружает аудиоданные."""
        pass

    @abstractmethod
    def close(self):
        """Освобождает ресурсы источника."""
        pass

    @property
    @abstractmethod
    def sample_rate(self):
        """Возвращает частоту дискретизации."""
        pass
