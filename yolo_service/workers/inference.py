# yolo_service/workers/inference.py

import json
from yolo_service.core.model import YOLOModel
from yolo_service.core.video import extract_frames
from redis import Redis
from rq import get_current_job


def run_inference(input_path: str, output_path: str, model_path: str = "yolov8n.pt"):
    """Runs inference using job queue, similar to yolo_service/cli.py
    
    Args:
        input_path (str): Path to video to infer upon

        output_path (str): Path to output file to dump detections

        model_path (str, Optional): Path to YOLO model, defaults to yolov8n.pt
    """
    job = get_current_job()
    if model_path:
        model = YOLOModel(model_path)
    else:
        model = YOLOModel()
    all_detections = []
    for batch in extract_frames(input_path):
        detections = model.infer_batch(batch)
        all_detections.extend(detections)
    with open(output_path, "w") as f:
        json.dump({"frames": all_detections}, f, indent=4)

    redis = Redis(host='redis', port=6379)
    redis.set(job.id, output_path)