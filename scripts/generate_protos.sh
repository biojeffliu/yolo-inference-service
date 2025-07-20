#!/bin/zsh
python3 -m grpc_tools.protoc -I yolo_service/api --python_out=yolo_service/api --pyi_out=yolo_service/api --grpc_python_out=yolo_service/api yolo_service/api/inference.proto