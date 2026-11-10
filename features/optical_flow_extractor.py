
import cv2
import numpy as np


class OpticalFlowFeatureExtractor:
    FEATURE_DIM = 7
    IMAGE_SIZE = (128, 128)

    def __init__(self):
        # Предыдущий кадр для каждого участника
        self.previous_gray = {}

    @staticmethod
    def compute_flow_features(flow):
        vx = flow[..., 0]
        vy = flow[..., 1]

        magnitude = np.sqrt(vx**2 + vy**2)

        mean_mag = magnitude.mean()
        std_mag = magnitude.std()

        mean_vx = vx.mean()
        mean_vy = vy.mean()

        std_vx = vx.std()
        std_vy = vy.std()

        # Энтропия направления движения
        angles = np.arctan2(vy, vx)
        hist, _ = np.histogram(
            angles,
            bins=8,
            range=(-np.pi, np.pi),
            density=True
        )
        hist = hist + 1e-8
        entropy = -np.sum(hist * np.log(hist))

        return np.array([
            mean_mag,
            std_mag,
            mean_vx,
            mean_vy,
            std_vx,
            std_vy,
            entropy
        ], dtype=np.float32)

    def extract(self, face_crop, person_id):
        """
        Извлекает признаки оптического потока для одного участника.

        face_crop: изображение лица из исходного кадра (BGR).
        person_id: идентификатор отслеживаемого участника.
        """

        if face_crop is None or face_crop.size == 0:
            self.reset(person_id)
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # Перевод в оттенки серого
        if face_crop.ndim == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop.copy()

        # Приведение к размеру, использовавшемуся при обучении
        gray = cv2.resize(gray, self.IMAGE_SIZE)

        # Для первого кадра вычислить поток невозможно
        if person_id not in self.previous_gray:
            self.previous_gray[person_id] = gray
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        previous_gray = self.previous_gray[person_id]

        # Расчёт оптического потока Farneback
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Сохраняем текущий кадр для следующего вызова
        self.previous_gray[person_id] = gray

        return self.compute_flow_features(flow)

    def reset(self, person_id):
        """Сбрасывает историю кадров одного участника."""
        self.previous_gray.pop(person_id, None)

    def reset_all(self):
        """Сбрасывает историю кадров всех участников."""
        self.previous_gray.clear()
