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
    redis

# HybrIK: clone into image, relax opencv pin, editable install (keeps source in place)
RUN git clone https://github.com/jeffffffli/HybrIK.git /opt/hybrik \
    && sed -i "s/'opencv-python==4.1.2.30'/'opencv-python'/g" /opt/hybrik/setup.py \
    && pip install --no-cache-dir -e /opt/hybrik

# GMR: General Motion Retargeting
RUN pip install --no-cache-dir git+https://github.com/YanjieZe/GMR.git

COPY . .

RUN mkdir -p /app/shared_media

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--queues=anyteleop", "--pool=solo"]
