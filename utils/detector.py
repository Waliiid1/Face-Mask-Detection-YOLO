from ultralytics import YOLO
import os


class YOLOModel:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = YOLO(model_path)

    def predict(self, image):
        results = self.model(image, conf=0.25)
        return results[0]