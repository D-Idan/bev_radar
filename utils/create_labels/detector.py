"""Car detection module using YOLOv8."""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict

import logging
from ultralytics.utils import LOGGER

# Suppress ultralytics logging
LOGGER.setLevel(logging.WARNING)


class CarDetector:
    def __init__(self, model_name: str, confidence: float = 0.5, iou: float = 0.45):
        """
        Initialize the car detector.

        Args:
            model_name: Name of the YOLO model to use
            confidence: Confidence threshold for detections
            iou: IOU threshold for NMS
        """
        self.model = YOLO(model_name, verbose=False)
        self.confidence = confidence
        self.iou = iou

    def detect(self, image_path: str, classes: List[int]) -> List[Dict]:
        """
        Detect cars in an image.

        Args:
            image_path: Path to the image
            classes: List of class IDs to detect

        Returns:
            List of detection dictionaries with bbox and confidence
        """
        results = self.model(
            image_path,
            conf=self.confidence,
            iou=self.iou,
            classes=classes,
            verbose = False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls
                    })

        return detections

    def get_image_size(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions."""
        img = cv2.imread(image_path)
        return img.shape[1], img.shape[0]  # width, height