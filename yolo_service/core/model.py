# yolo_service/core/model.py
# Wrapper for loading YOLO model inference on batches of frames.

from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict
import os
import cv2

class YOLOModel:
    def __init__(self, model_path: str = "yolov8n.pt", task: str = "detect"):
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
        results = self.model(frames, device=self.device, verbose=True)
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
        return detections
        
    def segment_batch(self, frames: List[np.ndarray], output_path_dir: str, frame_index_start: int = 0) -> List[Dict]:
        """
        Perform segmentation and save binary masks as PNGs.

        Args:
            frames (List[np.ndarray]): List of RGB frames as numpy arrays.
            output_path (str): Path to the output JSON file. Masks will be saved in a sibling directory.
            frame_index_start (int): Index offset for global frame indexing.

        Returns:
            List[Dict]: List of segmentation results per frame, with relative PNG mask paths.
        """
        # Derive mask directory next to the output JSON file
        masks_dir = os.path.join(output_path_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        results = self.model(frames, device=self.device, verbose=False)
        segmentations = []

        for frame_idx, result in enumerate(results):
            global_frame_idx = frame_index_start + frame_idx
            segms = []

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # shape: (N, H, W)
                boxes = result.boxes

                for i in range(len(masks)):
                    mask = (masks[i] > 0.5).astype(np.uint8) * 255
                    print(np.count_nonzero(mask))
                    mask_filename = f"frame_{global_frame_idx}_obj_{i}.png"
                    mask_full_path = os.path.join(masks_dir, mask_filename)
                    cv2.imwrite(mask_full_path, mask)

                    cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
                    conf = float(boxes.conf[i].item()) if boxes.conf is not None else -1

                    segms.append({
                        "mask_path": os.path.relpath(mask_full_path, start=os.path.dirname(output_path_dir)),
                        "class": cls_id,
                        "confidence": conf
                    })

            segmentations.append({
                "frame_index": global_frame_idx,
                "segmentations": segms
            })

        return segmentations