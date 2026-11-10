from pathlib import Path

import onnxruntime as ort
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self, det_size=(640, 640)):
        # Корневая папка проекта
        project_root = Path(__file__).resolve().parent.parent

        # Проверяем доступность CUDA в ONNX Runtime
        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in available_providers:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        # Загружаем локальную модель InsightFace
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            root=str(project_root),
            providers=providers
        )

        self.face_app.prepare(
            ctx_id=ctx_id,
            det_size=det_size
        )

    def detect_faces(self, frame, min_score=0.5):
        faces = self.face_app.get(frame)

        results = []

        for face in faces:
            if face.det_score < min_score:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "score": float(face.det_score)
            })

        return results