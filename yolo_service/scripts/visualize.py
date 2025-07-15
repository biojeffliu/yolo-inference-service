# yolo_service/scripts/visualize.py
# Takes detections and overlays them back onto video

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import json

colors_rgb = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Red
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (0, 255, 255),   # Cyan
    (255, 192, 203), # Pink
    (165, 42, 42),   # Brown
    (0, 0, 0)        # Black
]

def main():
    parser = argparse.ArgumentParser(prog="Detection visualization script.",
                                     description="Visualizes detections by overlaying bounding boxes with labels and probabilities.")
    parser.add_argument("labels_path", type=str, help="Path to list of labels txt file")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("detections_path", type=str, help="Path to YOLO outputted video detections")
    parser.add_argument("output_video", type=str, help="Path to output video with overlays")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Minimum confidence threshold for detections")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enables verbosity on output messages")
    args = parser.parse_args()

    verbose = args.verbose

    labels = open(args.labels_path).read().strip().split('\n')

    with open(args.detections_path, "r") as f:
        data = json.load(f)
    frame_dets = data["frames"]
    if verbose:
        print(f"Loaded {len(frame_dets)} frame detections from {args.detections_path}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video from: {args.video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(frame_dets) != total_frames:
        print(f"[WARN] JSON frame count doesn't match video, may skip overlays")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    if verbose:
        print(f"Output video: {args.output_video} (FPS: {fps}, Width x Height: {width} x {height})")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(frame_dets):
            dets = frame_dets[frame_idx]["detections"]
            for det in dets:
                if det["confidence"] >= args.min_conf:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    class_id = det["class"]
                    label = f"{args.labels_path}_{class_id} {det['confidence']:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors_rgb[class_id % len(colors_rgb)], 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, colors_rgb[class_id % len(colors_rgb)], 2)
        out.write(frame)
        if verbose and frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")
        frame_idx += 1
    cap.release()
    out.release()
    print(f"Annotated video saved to {args.output_video}")


if __name__ == "__main__":
    main()