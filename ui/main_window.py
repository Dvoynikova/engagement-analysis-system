import cv2
import matplotlib
matplotlib.use("Agg")

import numpy as np
import time
import torch

from collections import defaultdict, deque
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QComboBox
)

from detection.face_detector import FaceDetector
from processing.video_manager import VideoManager
from processing.audio_manager import AudioManager
from processing.feature_manager import FeatureManager
from processing.screen_capture import ScreenCapture
from processing.system_audio_capture import SystemAudioCapture

from features.optical_flow_extractor import OpticalFlowFeatureExtractor
from features.prosody_extractor import ProsodyFeatureExtractor
from features.vit_affectnet_extractor import ViTAffectNetExtractor
from features.wav2vec_extractor import Wav2VecExtractor

from models.multimodal_engagement_model import (
    load_multimodal_model,
    predict_engagement,
    MAX_LEN
)

from models.emotion_model import (
    load_emotion_models,
    predict_emotion,
    ACTIVATION_LABELS,
    VALENCE_LABELS
)

from analysis.dynamics_exporter import (
    prepare_analysis_rows,
    export_csv,
    export_plot
)


device = "cuda" if torch.cuda.is_available() else "cpu"

class ExportWorker(QThread):
    completed = Signal(str, str)
    failed = Signal(str)

    def __init__(self, records, png_path, csv_path):
        super().__init__()
        self.records = records
        self.png_path = png_path
        self.csv_path = csv_path

    def run(self):
        try:
            individual_rows, group_rows = prepare_analysis_rows(
                self.records
            )
    
            export_csv(
                individual_rows,
                group_rows,
                self.csv_path
            )
    
            export_plot(
                individual_rows,
                group_rows,
                self.png_path
            )
    
            self.completed.emit(self.png_path, self.csv_path)
    
        except Exception as e:
            self.failed.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        
        super().__init__()
    
        # Настройки главного окна
        self.setWindowTitle("Система анализа вовлеченности")
        self.resize(1000, 700)
        
        # Детектор лиц
        self.face_detector = FaceDetector()

        # Извлечение видеопризнаков — оптический поток
        self.flow_extractor = OpticalFlowFeatureExtractor()
        
        # Извлечение аудиопризнаков — просодические характеристики
        self.prosody_extractor = ProsodyFeatureExtractor()

        # Извлечение признаков для распознавания эмоций
        self.vit_extractor = ViTAffectNetExtractor(device=device)
        self.wav2vec_extractor = Wav2VecExtractor(device=device)

        self.video_manager = VideoManager()
        self.audio_manager = AudioManager()

        # Захват экрана
        self.screen_capture = ScreenCapture()
        
        # Поток захвата системного звука
        self.system_audio_capture = None
        
        # Источник данных: file или screen
        self.source_mode = "file"

        # Буфер последних 10 секунд аудио
        self.LIVE_AUDIO_SR = 16000
        self.LIVE_AUDIO_DURATION = 10
        self.LIVE_AUDIO_MAX_SAMPLES = (
            self.LIVE_AUDIO_SR * self.LIVE_AUDIO_DURATION
        )
        
        self.live_audio_chunks = deque()
        self.live_audio_samples = 0
        
        # Время начала анализа экрана
        self.screen_started_at = None

        # Время начала анализа экрана
        self.screen_started_at = None

        self.feature_manager = FeatureManager(
            engagement_video_extractor=self.flow_extractor,
            engagement_audio_extractor=self.prosody_extractor,
            emotion_video_extractor=self.vit_extractor,
            emotion_audio_extractor=self.wav2vec_extractor
        )
                
        # Мультимодальная модель оценки вовлечённости
        self.engagement_model, self.model_device = load_multimodal_model(
            device=device
        )
        # Модели распознавания активации и валентности
        (
            self.activation_model,
            self.valence_model,
            self.emotion_device
        ) = load_emotion_models(device=device)
        
        # Аудиоданные загруженного видео
        self.audio_data = None
        

        # Последние оценки вовлечённости участников
        self.person_engagement = {}
        self.person_emotions = {}

        self.person_prediction_history = defaultdict(
            lambda: {
                "engagement": deque(maxlen=6),
                "activation": deque(maxlen=6),
                "valence": deque(maxlen=6)
            }
        )
        # Полная история предсказаний для последующего экспорта
        self.analysis_history = []
        self.export_worker = None
        # Параметры обработки видео
        self.FRAME_SKIP = 2
        self.DETECTION_INTERVAL = 10
        # Частота обновления предсказания
        self.PREDICTION_INTERVAL = 10
        self.VIT_FRAME_SKIP_SCREEN = 5
        
        # Длина временного окна в секундах
        self.WINDOW_DURATION = 10.0
        
        # Счетчик кадров исходного видео
        self.frame_idx = 0
        
        # Последние обнаруженные лица
        self.current_faces = []
        
        # ID для следующего нового участника
        self.next_person_id = 1

        # Кнопка загрузки видео
        self.load_button = QPushButton("Загрузить видео")
        self.load_button.clicked.connect(self.load_video)

        # Выбор монитора
        self.monitor_combo = QComboBox()
        
        for i, monitor in enumerate(
            self.screen_capture.get_monitors(),
            start=1
        ):
            self.monitor_combo.addItem(
                f"Монитор {i} ({monitor['width']}x{monitor['height']})",
                i
            )
        
        # Кнопка запуска анализа экрана
        self.screen_button = QPushButton("Начать анализ экрана")
        self.screen_button.clicked.connect(
            self.start_screen_analysis
        )
        
        # Кнопка остановки анализа экрана
        self.stop_screen_button = QPushButton("Остановить анализ")
        self.stop_screen_button.clicked.connect(
            self.stop_screen_analysis
        )
        self.stop_screen_button.setEnabled(False)

        # Кнопка экспорта анализа
        self.export_button = QPushButton("Экспортировать анализ")
        self.export_button.clicked.connect(self.export_analysis)
        self.export_button.setEnabled(False)

        # Область для отображения видео
        self.video_label = QLabel("Здесь будет отображаться видео")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 500)

        # Компоновка элементов
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.monitor_combo)
        layout.addWidget(self.screen_button)
        layout.addWidget(self.stop_screen_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.video_label)

        # Центральный виджет
        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)

        # Таймер для показа кадров
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # История кадров каждого участника
        self.person_history = {}

        self.last_emotion_video_feature = {}
        
        # Частота кадров исходного видео
        self.video_fps = 25

        self.profile_stats = defaultdict(float)
        self.profile_window_start = time.perf_counter()

    def load_video(self):

        dialog = QFileDialog(self)
        dialog.setWindowTitle("Выберите видео")
        dialog.setNameFilter("Видео (*.mp4 *.avi *.mov *.mkv)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    
        if dialog.exec():
            file_path = dialog.selectedFiles()[0]
        else:
            return
    
        # Если пользователь выбрал файл
        if file_path:
            self.timer.stop()
        
            # Переключаем источник данных на видеофайл
            self.source_mode = "file"
            self.screen_started_at = None
            
            # Если другое видео уже было открыто,
            # освобождаем предыдущий объект
            self.video_manager.release()
            self.audio_manager.clear()
            self.feature_manager.reset()

            # Открываем выбранное видео
            self.video_manager.open_file(file_path)

            # Проверяем, удалось ли открыть видео
            if not self.video_manager.is_opened():
                return

            # Загружаем аудиодорожку видео
            try:
                self.audio_data = self.audio_manager.load_file(file_path)
            except Exception as e:
                self.audio_data = None
                print(f"Ошибка загрузки аудио: {e}")

            # Сбрасываем данные предыдущего видео
            self.frame_idx = 0
            self.current_faces = []
            self.next_person_id = 1
            
            self.person_history = {}
            self.person_engagement = {}
            self.person_emotions = {}
            self.person_prediction_history.clear()
            
            self.analysis_history.clear()
            self.export_button.setEnabled(False)
            
            self.flow_extractor.reset_all()
            
            # Получаем FPS исходного видео
            self.video_fps = self.video_manager.fps

            # Запускаем таймер.
            # Интервал задается в миллисекундах.
            interval = int(1000 / self.video_fps)
            self.timer.start(interval)
    #функция сопоставления лиц        
    def calculate_iou(self, box1, box2):
        """
        Вычисляет IoU двух прямоугольников.
    
        box = (x1, y1, x2, y2)
        """
    
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
    
        # Координаты пересечения
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)
    
        # Если пересечения нет
        if x2 <= x1 or y2 <= y1:
            return 0.0
    
        intersection = (x2 - x1) * (y2 - y1)
    
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
        union = area1 + area2 - intersection
    
        if union <= 0:
            return 0.0
    
        return intersection / union

    def remove_old_frames(self, current_timestamp):
        """Удаляет кадры, которые вышли за пределы временного окна."""
    
        window_start = current_timestamp - self.WINDOW_DURATION
    
        for person_id in list(self.person_history.keys()):
            history = self.person_history[person_id]
    
            # Оставляем только кадры внутри временного окна
            self.person_history[person_id] = [
                item
                for item in history
                if item["timestamp"] >= window_start
            ]

    #Функция присвоения ID
    def update_face_tracking(self, detected_faces):
        """Сопоставляет обнаруженные лица с участниками предыдущего кадра."""
    
        if not self.current_faces:
            for face in detected_faces:
                face["person_id"] = self.next_person_id
                self.next_person_id += 1
            return detected_faces
    
        # Если в обоих кадрах обнаружено только одно лицо,
        # сохраняем его ID независимо от перемещения по экрану.
        if len(self.current_faces) == 1 and len(detected_faces) == 1:
            detected_faces[0]["person_id"] = self.current_faces[0]["person_id"]
            return detected_faces
    
        candidates = []
    
        for new_idx, new_face in enumerate(detected_faces):
            nx1, ny1, nx2, ny2 = new_face["bbox"]
            new_cx = (nx1 + nx2) / 2
            new_cy = (ny1 + ny2) / 2
            new_area = max(1, (nx2 - nx1) * (ny2 - ny1))
    
            for old_idx, old_face in enumerate(self.current_faces):
                ox1, oy1, ox2, oy2 = old_face["bbox"]
                old_cx = (ox1 + ox2) / 2
                old_cy = (oy1 + oy2) / 2
                old_area = max(1, (ox2 - ox1) * (oy2 - oy1))
    
                iou = self.calculate_iou(
                    new_face["bbox"],
                    old_face["bbox"]
                )
    
                distance = np.sqrt(
                    (new_cx - old_cx) ** 2 +
                    (new_cy - old_cy) ** 2
                )
    
                # Нормируем расстояние на размер старой рамки.
                normalized_distance = distance / np.sqrt(old_area)
    
                # Проверяем, насколько изменился размер рамки.
                size_ratio = new_area / old_area
    
                # Отбрасываем явно неподходящие совпадения.
                if not (0.25 <= size_ratio <= 4.0):
                    continue
    
                if iou < 0.1 and normalized_distance > 2.5:
                    continue
    
                # Чем меньше значение, тем лучше совпадение.
                match_score = normalized_distance + 0.25 * (1 - iou)
    
                candidates.append(
                    (match_score, new_idx, old_idx)
                )
    
        candidates.sort(key=lambda x: x[0])
    
        assigned_new = set()
        assigned_old = set()
    
        for _, new_idx, old_idx in candidates:
            if new_idx in assigned_new or old_idx in assigned_old:
                continue
    
            detected_faces[new_idx]["person_id"] = (
                self.current_faces[old_idx]["person_id"]
            )
    
            assigned_new.add(new_idx)
            assigned_old.add(old_idx)
    
        # Присваиваем новые ID лицам, которым не нашли соответствие.
        for new_idx, face in enumerate(detected_faces):
            if new_idx not in assigned_new:
                face["person_id"] = self.next_person_id
                self.next_person_id += 1
    
        return detected_faces
    
    def update_frame(self):
        frame_started = time.perf_counter()
    
        # Получаем следующий кадр
        capture_started = time.perf_counter()
    
        if self.source_mode == "screen":
            frame = self.screen_capture.capture(
                self.monitor_combo.currentData(),
                scale=0.25
            )
            current_timestamp = time.monotonic() - self.screen_started_at
        else:
            ret, frame = self.video_manager.read()
    
            if not ret:
                self.timer.stop()
                self.video_manager.release()
                return
    
            current_timestamp = self.frame_idx / self.video_fps

        self.profile_stats["capture"] += time.perf_counter() - capture_started
        # Удаляем кадры, вышедшие за пределы временного окна
        t = time.perf_counter()
        self.remove_old_frames(current_timestamp)
        self.profile_stats["remove_old_frames"] += time.perf_counter() - t

        # Для экрана обрабатываем каждый кадр,
        # для видео сохраняем прежний пропуск каждого второго кадра
        frame_skip = (
            1 if self.source_mode == "screen"
            else self.FRAME_SKIP
        )
        
        t = time.perf_counter()
        skip_frame = self.frame_idx % frame_skip != 0
        self.profile_stats["skip_check"] += time.perf_counter() - t
        
        if skip_frame:
            self.frame_idx += 1
            return
    
        # -----------------------------------------
        # FRAME_SKIP
        # -----------------------------------------
        self.profile_stats["before_features"] += (
            time.perf_counter() - frame_started
        )
    
        # -----------------------------------------
        # DETECTION_INTERVAL
        # -----------------------------------------
    
        # Детектируем лица только периодически
        detection_started = time.perf_counter()
        t = time.perf_counter()
        if (
            self.current_faces == []
            or self.frame_idx
            % (self.FRAME_SKIP * self.DETECTION_INTERVAL) == 0
        ):
            detected_faces = self.face_detector.detect_faces(frame)
            
            t = time.perf_counter()
            self.current_faces = self.update_face_tracking(detected_faces)
            self.profile_stats["tracking"] += time.perf_counter() - t

        self.profile_stats["face_detection_and_tracking"] += time.perf_counter() - t
        
        self.profile_stats["face_detection"] += (
            time.perf_counter() - detection_started
        )
    
        # -----------------------------------------
        # Используем текущие bbox
        # -----------------------------------------
        # Кэш аудиопризнаков для текущего кадра
        audio_features_for_frame = None
        emotion_audio_features_for_frame = None
        audio_window_for_frame = None

        t = time.perf_counter()
        self.feature_manager.begin_frame()
        self.profile_stats["begin_frame"] += time.perf_counter() - t
        
        face_loop_started = time.perf_counter()
        for face in self.current_faces:
    
            x1, y1, x2, y2 = face["bbox"]
    
            person_id = face["person_id"]
            score = face["score"]
            #-------------------------
            #сохранение истории кадров
            # ID участника
            person_id = face["person_id"]
            
            # Ограничиваем координаты размерами кадра
            height, width = frame.shape[:2]
            
            x1_crop = max(0, min(x1, width))
            y1_crop = max(0, min(y1, height))
            x2_crop = max(0, min(x2, width))
            y2_crop = max(0, min(y2, height))

            crop_started = time.perf_counter()
            # Вырезаем лицо из текущего кадра
            face_crop = frame[
                y1_crop:y2_crop,
                x1_crop:x2_crop
            ]
            self.profile_stats["face_crop"] += time.perf_counter() - crop_started
            
            # Проверяем, что кроп не пустой
            if face_crop.size > 0:

            # Вырезаем лицо из текущего кадра

                # Извлекаем видеопризнаки ViT для распознавания эмоций
                t = time.perf_counter()

                if self.source_mode == "screen":
                    if self.frame_idx % (
                        self.FRAME_SKIP * self.VIT_FRAME_SKIP_SCREEN
                    ) == 0:
                        emotion_video_feature = self.feature_manager.extract_emotion_video(face_crop)
                        self.last_emotion_video_feature[person_id] = emotion_video_feature
                    else:
                        emotion_video_feature = self.last_emotion_video_feature.get(person_id)
                else:
                    emotion_video_feature = self.feature_manager.extract_emotion_video(face_crop)
                
                self.profile_stats["vit"] += time.perf_counter() - t
            
                # Извлекаем признаки оптического потока
                # Передаём исходный кроп лица в формате BGR
                t = time.perf_counter()

                video_feature = self.feature_manager.extract_engagement_video(
                    face_crop,
                    person_id
                )
                
                self.profile_stats["optical_flow"] += time.perf_counter() - t
            
                # Создаём историю нового участника
                if person_id not in self.person_history:
                    self.person_history[person_id] = []
            
                # Сохраняем информацию о кадре
                self.person_history[person_id].append({
                    "frame_idx": self.frame_idx,
                    "timestamp": current_timestamp,
                    "bbox": (x1, y1, x2, y2),
                    "feature": video_feature,
                    "emotion_feature": emotion_video_feature,
                    "det_score": face["score"]
                })

                # Пересчитываем вовлечённость примерно раз в секунду
                t = time.perf_counter()
                history = self.get_person_window(person_id)
                
                self.profile_stats["get_window"] += time.perf_counter() - t
            
                processed_frame_idx = (
                    self.frame_idx
                    if self.source_mode == "screen"
                    else self.frame_idx // self.FRAME_SKIP
                )
                    
                if self.source_mode == "screen":
                    audio_ready = (
                        self.live_audio_samples >= self.LIVE_AUDIO_MAX_SAMPLES
                    )
                else:
                    audio_ready = self.audio_data is not None
                
                should_predict = (
                    len(history) >= MAX_LEN
                    and audio_ready
                    and (
                        person_id not in self.person_engagement
                        or processed_frame_idx % self.PREDICTION_INTERVAL == 0
                    )
                )
                    
                if should_predict:

                    analysis_record = {
                        "timestamp_sec": float(current_timestamp),
                        "person_id": person_id,
                        "engagement_probs": None,
                        "activation_probs": None,
                        "valence_probs": None
                    }
                    
                    # Видеопризнаки участника за последние 10 секунд
                    video_features = np.stack([
                        item["feature"]
                        for item in history[-MAX_LEN:]
                    ]).astype(np.float32)

                    # Видеопризнаки ViT за последние 10 секунд
                    emotion_video_features = np.stack([
                        item["emotion_feature"]
                        for item in history[-MAX_LEN:]
                    ]).astype(np.float32)


                    # Подготовка аудиоданных для обеих моделей
                    if self.source_mode == "screen":
                        # Последние 10 секунд системного звука
                        audio_input = self.get_live_audio_window()
                        audio_start = 0
                    else:
                        # Аудиодорожка загруженного видео
                        audio_input = self.audio_data
                        audio_start = max(0, current_timestamp - self.WINDOW_DURATION)

                    try:
                        # Извлекаем аудиопризнаки только один раз
                        # для текущего кадра
                        if audio_features_for_frame is None:

                            if self.source_mode == "screen":
                        
                                if audio_window_for_frame is None:
                                    audio_window_for_frame = self.get_live_audio_window()
                        
                                audio_input = audio_window_for_frame
                                audio_start = 0
                        
                            else:
                        
                                audio_input = self.audio_data
                        
                                audio_start = max(
                                    0,
                                    current_timestamp - self.WINDOW_DURATION
                                )
                        
                            audio_features_for_frame = (
                                self.feature_manager.extract_engagement_audio(
                                    audio_input,
                                    start_sec=audio_start
                                )
                            )
                            
                        
                        # Используем общую последовательность
                        # аудиопризнаков для всех участников
                        audio_features = audio_features_for_frame
                    
                        # Мультимодальное предсказание
                        result = predict_engagement(
                            self.engagement_model,
                            video_features,
                            audio_features,
                            self.model_device
                        )
                    
                        self.person_engagement[person_id] = result
                        self.person_prediction_history[person_id]["engagement"].append(
                            result["probabilities"]
                        )
                        analysis_record["engagement_probs"] = self.get_smoothed_probabilities(
                            person_id, "engagement"
                        )
                    
                    except Exception as e:
                        print(
                            f"Ошибка предсказания вовлечённости "
                            f"участника {person_id}: {e}"
                        )

                    # -----------------------------------------
                    # Распознавание активации и валентности
                    # -----------------------------------------
                    
                    try:
                        # Извлекаем Wav2Vec2-признаки один раз
                        # для текущего кадра и используем для всех лиц
                        if emotion_audio_features_for_frame is None:

                            emotion_audio_features_for_frame = (
                                self.feature_manager.extract_emotion_audio(
                                    audio_input,
                                    start_sec=audio_start
                                )
                            )
                    
                        emotion_audio_features = emotion_audio_features_for_frame
                    
                        # Активация
                        activation_result = predict_emotion(
                            self.activation_model,
                            emotion_video_features,
                            emotion_audio_features,
                            self.emotion_device,
                            ACTIVATION_LABELS
                        )
                    
                        # Валентность
                        valence_result = predict_emotion(
                            self.valence_model,
                            emotion_video_features,
                            emotion_audio_features,
                            self.emotion_device,
                            VALENCE_LABELS
                        )
                    
                        self.person_emotions[person_id] = {
                            "activation": activation_result,
                            "valence": valence_result
                        }

                        self.person_prediction_history[person_id]["activation"].append(
                            activation_result["probabilities"]
                        )
                        
                        self.person_prediction_history[person_id]["valence"].append(
                            valence_result["probabilities"]
                        )
                        analysis_record["activation_probs"] = self.get_smoothed_probabilities(
                            person_id, "activation"
                        )
                        
                        analysis_record["valence_probs"] = self.get_smoothed_probabilities(
                            person_id, "valence"
                        )
                    
                    except Exception as e:
                        print(
                            f"Ошибка распознавания эмоций "
                            f"участника {person_id}: {e}" )
                    # Сохраняем результаты текущего предсказания
                    if any(
                        analysis_record[key] is not None
                        for key in [
                            "engagement_probs",
                            "activation_probs",
                            "valence_probs"
                        ]
                    ):
                        self.analysis_history.append(analysis_record)
                        self.export_button.setEnabled(True)
                        
            else:
                self.feature_manager.reset_engagement_video(person_id)
            #-------------------------
    
            # Рисуем рамку
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            
            # Параметры подписей
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_height = 20
            
            # Подпись слева от рамки лица
            def draw_left_label(text, y, color, font_scale=0.5):
                (text_width, text_height), _ = cv2.getTextSize(
                    text, font, font_scale, 1
                )
            
                text_x = max(5, x1 - text_width - 10)
                text_y = max(text_height + 5, y)
            
                cv2.putText(
                    frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    1,
                    cv2.LINE_AA
                )
            
            # Имя участника
            draw_left_label(
                f"Person {person_id}",
                y1 - 3 * line_height,
                (0, 255, 0),
                0.45
            )
            
            # Вовлечённость
            if person_id in self.person_engagement:
                result = self.person_engagement[person_id]
                engagement = result["class"]
                probability = result["probabilities"][engagement]
                engagement_text = f"Engagement: {engagement} ({probability:.0%})"
            else:
                n_frames = len(self.get_person_window(person_id))
                engagement_text = f"Engagement: collecting" #{n_frames}/{MAX_LEN}
            
            draw_left_label(
                engagement_text,
                y1 - 2 * line_height,
                (0, 255, 255)
            )
            
            # Активация и валентность
            if person_id in self.person_emotions:
                emotion_result = self.person_emotions[person_id]
                activation = emotion_result["activation"]["class"]
                valence = emotion_result["valence"]["class"]
            
                activation_text = f"Activation: {activation}"
                valence_text = f"Valence: {valence}"
            else:
                activation_text = "Activation: collecting"
                valence_text = "Valence: collecting"
            
            draw_left_label(
                activation_text,
                y1 - line_height,
                (0, 255, 255)
            )
            
            draw_left_label(
                valence_text,
                y1 - 10,
                (0, 255, 255)
            )
            '''
            # Уверенность детекции
            cv2.putText(
                frame,
                f"Face: {score:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            '''
        self.profile_stats["face_loop"] += (
            time.perf_counter() - face_loop_started
        )
        
        # -----------------------------------------
        # BGR → RGB
        # -----------------------------------------
        render_started = time.perf_counter()
        
        frame = self.draw_group_summary(frame)


        
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        
    
        height, width, channels = frame.shape
    
        image = QImage(
            frame.data,
            width,
            height,
            width * channels,
            QImage.Format_RGB888
        )
    
        pixmap = QPixmap.fromImage(image)
    
        pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    
        self.video_label.setPixmap(pixmap)
        self.profile_stats["render"] += (
            time.perf_counter() - render_started
        )
        # Переходим к следующему кадру
        self.frame_idx += 1

    def get_person_window(self, person_id):
        """Возвращает кадры текущего временного окна участника."""
    
        return self.person_history.get(person_id, [])

    def closeEvent(self, event):

        self.timer.stop()
    
        if self.system_audio_capture is not None:
            self.system_audio_capture.stop()
            self.system_audio_capture = None
    
        self.video_manager.release()
        self.audio_manager.clear()
        self.feature_manager.reset()
    
        self.screen_capture.release()

        if (
            self.export_worker is not None
            and self.export_worker.isRunning()
        ):
            self.export_worker.wait()
    
        event.accept()

    def get_group_average(self, task):
        labels = {
            "engagement": ["Low", "Medium", "High"],
            "activation": ["Low", "Medium", "High"],
            "valence": ["Negative", "Neutral", "Positive"]
        }[task]
    
        participant_probs = []
    
        for person_id, history in self.person_prediction_history.items():
            predictions = history[task]
    
            if not predictions:
                continue
    
            mean_probs = np.mean(
                [list(prediction.values()) for prediction in predictions],
                axis=0
            )
    
            participant_probs.append(mean_probs)
    
        if not participant_probs:
            return "collecting"
    
        group_probs = np.mean(participant_probs, axis=0)
    
        return labels[np.argmax(group_probs)]

    def get_smoothed_probabilities(self, person_id, task):
        labels = {
            "engagement": ["Low", "Medium", "High"],
            "activation": ["Low", "Medium", "High"],
            "valence": ["Negative", "Neutral", "Positive"]
        }[task]
    
        predictions = self.person_prediction_history[person_id][task]
    
        if not predictions:
            return None
    
        return {
            label: float(np.mean([
                prediction[label] for prediction in predictions
            ]))
            for label in labels
        }


    def draw_group_summary(self, frame):
        engagement = self.get_group_average("engagement")
        activation = self.get_group_average("activation")
        valence = self.get_group_average("valence")
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        padding = 12
        line_height = 25
    
        title = "GROUP DYNAMICS"
    
        rows = [
            f"Engagement: {engagement}",
            f"Activation: {activation}",
            f"Valence: {valence}"
        ]
    
        all_text = [title] + rows
    
        max_width = max(
            cv2.getTextSize(text, font, 0.55, 1)[0][0]
            for text in all_text
        )
    
        panel_width = max_width + 2 * padding
        panel_height = len(all_text) * line_height + 2 * padding
    
        frame_h, frame_w = frame.shape[:2]
    
        x = 15
        y = max(15, frame_h - panel_height - 15)
    
        overlay = frame.copy()
    
        cv2.rectangle(
            overlay,
            (x, y),
            (min(x + panel_width, frame_w - 1),
             min(y + panel_height, frame_h - 1)),
            (25, 25, 25),
            -1
        )
    
        frame = cv2.addWeighted(
            overlay, 0.75, frame, 0.25, 0
        )
    
        cv2.putText(
            frame,
            title,
            (x + padding, y + padding + 15),
            font,
            0.6,
            (255, 255, 255),
            1
        )
    
        colors = [
            (0, 255, 255),
            (0, 255, 255),
            (0, 255, 255)
        ]
    
        for i, text in enumerate(rows):
            cv2.putText(
                frame,
                text,
                (x + padding, y + padding + (i + 1) * line_height + 15),
                font,
                0.55,
                colors[i],
                1
            )
    
        return frame

    def export_analysis(self):

        if not self.analysis_history:

            QMessageBox.warning(
                self,
                "Экспорт анализа",
                "Нет данных для экспорта."
            )
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить анализ групповой динамики",
            "group_dynamics.png",
            "PNG (*.png)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if not file_path:
            return
    
        png_path = Path(file_path)
    
        if png_path.suffix.lower() != ".png":
            png_path = png_path.with_suffix(".png")
    
        csv_path = png_path.with_suffix(".csv")
    
        # Создаём снимок истории на момент запуска экспорта
        records = list(self.analysis_history)
        self.export_button.setEnabled(False)
    
        self.export_worker = ExportWorker(
            records,
            str(png_path),
            str(csv_path)
        )
    
        self.export_worker.completed.connect(
            self.on_export_completed
        )
    
        self.export_worker.failed.connect(
            self.on_export_failed
        )
    
        self.export_worker.finished.connect(
            self.on_export_finished
        )
        self.export_worker.start()

    def on_export_completed(self, png_path, csv_path):
        QMessageBox.information(
            self,
            "Экспорт завершён",
            f"Файлы сохранены:\n\n{png_path}\n{csv_path}"
        )
    
    def on_export_failed(self, error):
        QMessageBox.critical(
            self,
            "Ошибка экспорта",
            f"Не удалось сохранить анализ:\n{error}"
        )
    
    def on_export_finished(self):
        self.export_worker = None
        self.export_button.setEnabled(
            bool(self.analysis_history)
        )

    def handle_audio_chunk(self, audio):
        """Добавляет поступивший аудиофрагмент в буфер."""
    
        audio = np.asarray(audio, dtype=np.float32).flatten()
    
        self.live_audio_chunks.append(audio)
        self.live_audio_samples += len(audio)
    
        # Оставляем только последние 10 секунд
        while self.live_audio_samples > self.LIVE_AUDIO_MAX_SAMPLES:
            excess = (
                self.live_audio_samples
                - self.LIVE_AUDIO_MAX_SAMPLES
            )
    
            first_chunk = self.live_audio_chunks[0]
    
            if len(first_chunk) <= excess:
                self.live_audio_chunks.popleft()
                self.live_audio_samples -= len(first_chunk)
            else:
                self.live_audio_chunks[0] = first_chunk[excess:]
                self.live_audio_samples -= excess
    
    
    def get_live_audio_window(self):
        """Возвращает последние 10 секунд аудио."""
    
        if self.live_audio_samples < self.LIVE_AUDIO_MAX_SAMPLES:
            return None
    
        audio = np.concatenate(list(self.live_audio_chunks))
    
        return audio[-self.LIVE_AUDIO_MAX_SAMPLES:]

    def start_screen_analysis(self):
        self.timer.stop()
        self.video_manager.release()
        self.audio_manager.clear()
        self.feature_manager.reset()
        self.flow_extractor.reset_all()
    
        # Сбрасываем историю предыдущего анализа
        self.frame_idx = 0
        self.current_faces = []
        self.next_person_id = 1
    
        self.person_history = {}
        self.person_engagement = {}
        self.person_emotions = {}
        self.person_prediction_history.clear()
        self.analysis_history.clear()
        self.last_emotion_video_feature.clear()
    
        self.live_audio_chunks.clear()
        self.live_audio_samples = 0
    
        self.audio_data = None
    
        # Переключаем источник данных
        self.source_mode = "screen"
        self.screen_started_at = time.monotonic()
    
        self.video_fps = 25

        # Статистика производительности
        self.profile_stats = defaultdict(float)
        self.profile_window_start = time.perf_counter()
    
        # Запускаем захват системного звука
        self.system_audio_capture = SystemAudioCapture(
            sample_rate=self.LIVE_AUDIO_SR
        )
    
        self.system_audio_capture.audio_chunk.connect(
            self.handle_audio_chunk
        )
    
        self.system_audio_capture.error.connect(
            lambda message: print(
                f"Ошибка захвата звука: {message}"
            )
        )
    
        self.system_audio_capture.start()
    
        # Запускаем обработку экрана
        self.timer.start(40)
    
        self.screen_button.setEnabled(False)
        self.stop_screen_button.setEnabled(True)
        self.load_button.setEnabled(False)
        self.monitor_combo.setEnabled(False)
        self.export_button.setEnabled(False)
    
    
    def stop_screen_analysis(self):
        self.timer.stop()
    
        if self.system_audio_capture is not None:
            self.system_audio_capture.stop()
            self.system_audio_capture = None
    
        self.screen_button.setEnabled(True)
        self.stop_screen_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.monitor_combo.setEnabled(True)
    
        self.export_button.setEnabled(
            bool(self.analysis_history)
        )
