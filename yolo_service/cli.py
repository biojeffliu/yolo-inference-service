# yolo_service/cli.py
# CLI entry point for YOLO inference 

import json
import argparse
from yolo_service.core.model import YOLOModel
from yolo_service.core.video import extract_frames
import os

def detect_images(model: YOLOModel, video_path: str, output_dir: str, verbose: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    all_detections = []
    batch_count = 0
    total_frames = 0
    output_json_path = os.path.join(output_dir, "segmentations.json")
    for batch in extract_frames(video_path):
        batch_size = len(batch)
        if verbose:
            print(f"Dumping detections to {output_json_path}...")
            batch_count += 1
            total_frames += batch_size
        detections = model.infer_batch(batch)
        all_detections.extend(detections)

    
    if verbose:
        print(f"Dumping detections to {output_json_path}...")
    with open(output_json_path, "w") as f:
        json.dump({"frames": all_detections}, f, indent=4)

    print(f"Detections saved to {output_json_path}")

def segment_images(model: YOLOModel, video_path: str, output_dir: str, verbose: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    all_segmentations = []
    batch_count = 0
    total_frames = 0
    global_frame_idx = 0
    for batch in extract_frames(video_path):
        batch_size = len(batch)
        if verbose:
            print(f"Dumping segmentations to {output_dir}...")
            batch_count += 1
            total_frames += batch_size
        segmentations = model.segment_batch(batch, output_dir, global_frame_idx)
        global_frame_idx += batch_size
        all_segmentations.extend(segmentations)

    output_json_path = os.path.join(output_dir, "segmentations.json")
    if verbose:
        print(f"Dumping segmentations to {output_json_path}...")
    with open(output_json_path, "w") as f:
        json.dump({"frames": all_segmentations}, f, indent=4)

    print(f"Segmentations saved to {output_dir}")    

def main():
    """CLI to run inference using YOLO model."""
    parser = argparse.ArgumentParser(prog="yolo_service", description="YOLO Video Inference CLI")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("output_dir", type=str, help="Path to the output json file for detections")
    parser.add_argument("--model-path", type=str, help="Path to YOLO model, default yolov8n.pt")
    parser.add_argument("--task", type=str, choices=["detect", "segment"], default="detect", help="Task type for YOLO model, default detect")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enables verbosity on output messages")
    args = parser.parse_args()

    if args.verbose:
        print(f"Initializing YOLO model with task {args.task}...")
    if args.model_path:
        model = YOLOModel(model_path=args.model_path, task=args.task)
    else:
        model = YOLOModel(task=args.task)
    if args.verbose:
        print(f"Model successfully loaded onto device: {model.device}")

    if args.task == "detect":
        detect_images(model, args.video_path, args.output_dir, verbose=args.verbose)
    elif args.task == "segment":
        segment_images(model, args.video_path, args.output_dir, verbose=args.verbose)

if __name__ == "__main__":
    main()
