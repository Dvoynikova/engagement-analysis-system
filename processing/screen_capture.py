
import cv2
import mss
import numpy as np


class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()

    def get_monitors(self):
        # Нулевой элемент — объединённая область всех мониторов
        return self.sct.monitors[1:]

    def capture(self, monitor_index=1, scale=0.5):
        screenshot = self.sct.grab(
            self.sct.monitors[monitor_index]
        )
    
        frame = np.asarray(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
        if scale != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
    
        return frame

    def release(self):
        self.sct.close()
