# yolo_service/api/main.py
import grpc
from concurrent import futures
import os
import json
from redis import Redis
from rq import Queue
from yolo_service.api import inference_pb2_grpc, inference_pb2
from yolo_service.workers.inference import run_inference

class InferenceServiceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.redis = Redis(host='redis', port=6379)
        self.q = Queue(connection=self.redis)
    
    def InferVideo(self, request, context):
        video_path = f"tmp/{request.filename}"
        with open(video_path, "wb") as f:
            f.write(request.video_data)

        output_path = f"tmp/detections_{request.filename}.json"
        job = self.q.enqueue(run_inference, video_path, output_path)
        return inference_pb2.JobResponse(job_id=job.id)
    
    def GetResult(self, request, context):
        result_path = self.redis.get(request.job_id)
        if not result_path:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Job not found or pending")
            return inference_pb2.InferenceResult()
        result_path = result_path.decode('utf-8')
        with open(result_path, "r") as f:
            data = json.load(f)

        result = inference_pb2.InferenceResult()
        for frame in data["frames"]:
            frame_det = inference_pb2.FrameDetections()
            for det in frame["detections"]:
                detection = inference_pb2.Detection(
                    class_id=det["class"],
                    confidence=det["confidence"],
                    bbox=det["bbox"]
                )
                frame_det.detections.append(detection)
            result.frames.append(frame_det)
        os.remove(result_path)
        return result



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()