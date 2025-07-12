# yolo_service/core/model.py
# Wrapper for loading YOLO model inference on batches of frames.

from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict

class YOLOModel:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize the YOLO Model.

        Args:
            model_path (str, optional): Path to pre-trained YOLO model. Defaults to "yolov8n.pt".

        Raises:
            ValueError: If selected device is not available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(model_path)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA selected but not available.")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError(f"MPS selected but not available.")
        
    def infer_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """Perform object dection on a batch of frames.

        Args:
            frames (List[np.ndarray]): List of frames as numpy arrays (RGB format).

        Returns:
            List[Dict]: List of detections per frame, each with 'detections' key containing dicts (class, confidence, bbox).
        """
        results = self.model(frames, device=self.device, verbose=False)
        detections = []
        for result in results:
            dets = []
            for box in result.boxes:
                dets.append({
                    "class": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0]
                })
            detections.append({"detections": dets})
            