# yolo_service/cli.py
# CLI entry point for YOLO inference 

import json
import argparse
from yolo_service.core.model import YOLOModel
from yolo_service.core.video import extract_frames

def main():
    """CLI to run inference using YOLO model."""
    parser = argparse.ArgumentParser(prog="yolo_service", description="YOLO Video Inference CLI")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("output_json", type=str, help="Path to the output json file for detections")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enables verbosity on output messages")
    args = parser.parse_args()

    if args.verbose:
        print(f"Initializing YOLO model...")
    model = YOLOModel()
    if args.verbose:
        print(f"Model successfully loaded onto device: {model.device}")

    all_detections = []
    batch_count = 0
    total_frames = 0
    for batch in extract_frames(args.video_path):
        batch_size = len(batch)
        if args.verbose:
            print(f"Inferring on batch {batch_count} of size {batch_size}")
            batch_count += 1
            total_frames += batch_size
        detections = model.infer_batch(batch)
        all_detections.extend(detections)

    if args.verbose:
        print(f"Dumping detections to {args.output_json}...")
    with open(args.output_json, "w") as f:
        json.dump({"frames": all_detections}, f, indent=4)

    print(f"Detections saved to {args.output_json}")

if __name__ == "__main__":
    main()
