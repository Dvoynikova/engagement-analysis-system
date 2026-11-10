class FeatureManager:

    def __init__(
        self,
        engagement_video_extractor,
        engagement_audio_extractor,
        emotion_video_extractor=None,
        emotion_audio_extractor=None
    ):
        # Экстракторы вовлечённости
        self.engagement_video_extractor = engagement_video_extractor
        self.engagement_audio_extractor = engagement_audio_extractor

        # Экстракторы эмоций
        self.emotion_video_extractor = emotion_video_extractor
        self.emotion_audio_extractor = emotion_audio_extractor

        # Кэш аудиопризнаков вовлечённости
        self.audio_cache = {}

        # Кэш аудиопризнаков эмоций
        self.emotion_audio_cache = {}

    def begin_frame(self):
        """
        Вызывается в начале обработки нового кадра.
        Очищает кэши аудиопризнаков.
        """
        self.audio_cache.clear()
        self.emotion_audio_cache.clear()

    # --------------------------------------------------
    # ВОВЛЕЧЁННОСТЬ
    # --------------------------------------------------

    def extract_engagement_video(self, face_crop, person_id):
        """Извлекает видеопризнаки конкретного участника."""
        return self.engagement_video_extractor.extract(
            face_crop,
            person_id
        )

    def extract_engagement_audio(self, audio_data, start_sec=0):
        """
        Извлекает аудиопризнаки вовлечённости.
        Результат кэшируется для текущего кадра.
        """
        sample_rate = self.engagement_audio_extractor.SR
        cache_key = int(round(start_sec * sample_rate))

        if cache_key not in self.audio_cache:
            features = self.engagement_audio_extractor.extract_window(
                audio_data,
                start_sec=start_sec
            )
            self.audio_cache[cache_key] = features

        return self.audio_cache[cache_key]

    def reset_engagement_video(self, person_id):
        """Сбрасывает историю видеопризнаков участника."""
        self.engagement_video_extractor.reset(person_id)

    # --------------------------------------------------
    # ЭМОЦИИ: АКТИВАЦИЯ И ВАЛЕНТНОСТЬ
    # --------------------------------------------------

    def extract_emotion_video(self, face_crop):
        """
        Извлекает ViT-признаки лица.
        Face crop передаётся из MainWindow после детекции лица.
        """
        if self.emotion_video_extractor is None:
            raise RuntimeError(
                "Emotion video extractor is not initialized"
            )

        return self.emotion_video_extractor.extract(face_crop)

    def extract_emotion_audio(self, audio_data, start_sec=0):
        """
        Извлекает Wav2Vec2-признаки общей аудиодорожки.
        Результат кэшируется для текущего кадра.
        """
        if self.emotion_audio_extractor is None:
            raise RuntimeError(
                "Emotion audio extractor is not initialized"
            )

        sample_rate = self.emotion_audio_extractor.sample_rate
        cache_key = int(round(start_sec * sample_rate))

        if cache_key not in self.emotion_audio_cache:
            features = self.emotion_audio_extractor.extract_window(
                audio_data,
                start_sec=start_sec
            )
            self.emotion_audio_cache[cache_key] = features

        return self.emotion_audio_cache[cache_key]

    # --------------------------------------------------
    # СБРОС СОСТОЯНИЯ
    # --------------------------------------------------

    def reset(self):
        """Очищает временные кэши."""
        self.audio_cache.clear()
        self.emotion_audio_cache.clear()