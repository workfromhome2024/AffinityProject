FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    mediapipe==0.10.14 \
    opencv-python \
    numpy \
    pin \
    celery \
    redis

COPY . .

RUN mkdir -p /app/shared_media

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--queues=anyteleop", "--pool=solo"]
