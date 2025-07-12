# yolo_service/core/video.py
# Extracts video frames and batches for inference.

import cv2
import numpy as np
from typing import List, Generator

def extract_frames(video_path: str, batch_size: int = 32) -> Generator[List[np.ndarray], None, None]:
    """Extracts frames from a video file in batches.

    Args:
        video_path (str): Path to input video file.
        batch_size (int, optional): Number of frames per batch, defaults to 32.

    Yields:
        Generator[List[np.ndarry], None, None]: Batches of frames as numpy arrays in RGB format.

    Raises:
        ValueError
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video from: {video_path}.")
    
    batch = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if len(batch) == batch_size:
            yield batch
            batch = []
        
    if batch:
        yield batch
    
    cap.release()

    


    