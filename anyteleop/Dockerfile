FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (ffmpeg needed by GVHMR video I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    libgl1 libglib2.0-0 git build-essential ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.3.0 with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121


# SMPL body model package
RUN pip install --no-cache-dir "smplx[torch]"

# Python dependencies
RUN pip install --no-cache-dir \
    opencv-python numpy scipy cython pycocotools celery redis ultralytics

# GVHMR (hmr4d package) — video-level human mesh recovery
RUN git clone https://github.com/lithiumice/GVHMR.git /opt/gvhmr \
    && pip install --no-cache-dir -e /opt/gvhmr

# GMR: General Motion Retargeting (sparse checkout — only unitree_g1 assets)
RUN git clone --no-checkout --filter=blob:none https://github.com/YanjieZe/GMR.git /opt/gmr \
    && cd /opt/gmr \
    && git sparse-checkout set --no-cone '/*' '!/assets/*' '/assets/unitree_g1/**' \
    && git checkout \
    && pip install --no-cache-dir -e /opt/gmr

COPY . .

RUN mkdir -p /app/shared_media

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info", "--queues=anyteleop", "--pool=solo"]
