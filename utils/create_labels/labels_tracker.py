"""Object tracking module for assigning consistent IDs."""
import os

from tqdm import tqdm
from ultralytics import YOLO
import numpy as np
from typing import Dict, List, Tuple
import cv2
from pathlib import Path

class CarTracker:
    def __init__(self, model_name: str, tracker_config: str, confidence: float = 0.5):
        """
        Initialize the tracker.

        Args:
            model_name: YOLO model name
            tracker_config: Tracker configuration file
            confidence: Detection confidence threshold
        """
        model_name_path = Path(__file__).parent / model_name
        model_name = model_name_path if model_name_path.exists() else model_name
        self.model = YOLO(model_name)
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.track_history = {}

    def track_batch(self, image_paths: List[str], classes: List[int]) -> Dict[str, List[Dict]]:
        """
        Track objects across multiple images.
        """
        results_dict = {}

        # Process images in sequence for tracking
        for img_path in tqdm(image_paths, desc="Label Images and Tracking objects", unit="image"):

            # Verify image exists and is readable
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                print(f"ERROR: Cannot read image: {img_path}")
                print(f"  - File exists: {os.path.exists(img_path)}")
                print(f"  - File size: {os.path.getsize(img_path) if os.path.exists(img_path) else 'N/A'}")
                continue  # Skip this image

            try:
                results = self.model.track(
                    img_path,
                    persist=True,
                    tracker=self.tracker_config,
                    conf=self.confidence,
                    classes=classes
                )

                detections = []
                for r in results:
                    boxes = r.boxes
                    if boxes is not None and hasattr(boxes, 'id') and boxes.id is not None:
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            track_id = int(boxes.id[i].item())
                            conf = box.conf[0].item()

                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'id': track_id,
                                'confidence': conf
                            })

                results_dict[img_path] = detections

            except Exception as e:
                print(f"ERROR processing {img_path}: {str(e)}")
                results_dict[img_path] = []  # Empty list for failed images

        return results_dict