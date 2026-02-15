FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    opencv-python \
    numpy \
    scipy \
    cython \
    pycocotools \
    onnxruntime \
    celery \
    redis \
    && pip install --no-cache-dir --no-deps git+https://github.com/jeffffffli/HybrIK.git \
    && pip install --no-cache-dir git+https://github.com/YanjieZe/GMR.git

COPY . .

RUN mkdir -p /app/shared_media

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--queues=anyteleop", "--pool=solo"]
