FROM --platform=linux/arm64 python:3.11-slim
#docker build --platform=linux/arm64 -t <image-name> .

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip \
    wget curl libopencv-dev redis-server gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
        # https://download.pytorch.org/whl/cu121
    /opt/venv/bin/pip install --no-cache-dir \
        opencv-python-headless ultralytics==8.2.0 \
        pytest==8.2.2 ruff==0.5.0 black==24.4.2 \
        redis==5.0.7 rq==1.16.2 grpcio==1.64.1 grpcio-tools==1.64.1


ENV PATH="/opt/venv/bin:$PATH"

COPY yolo_service /app/yolo_service
COPY scripts /app/scripts

RUN chmod +x /app/scripts/generate_protos.sh && \
    /app/scripts/generate_protos.sh

EXPOSE 50051

CMD ["python", "yolo_service/api/main.py"]