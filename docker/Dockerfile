# Docker Hub's PyTorch image mirrored for reliable China-region pulls.
FROM docker.m.daocloud.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY docker/requirements-smoke.txt /tmp/requirements-smoke.txt
RUN git clone --depth 1 --branch v6.2 https://github.com/ultralytics/yolov5.git yolov5 && \
    pip install --no-cache-dir -r yolov5/requirements.txt && \
    pip install --no-cache-dir "numpy<2" -r /tmp/requirements-smoke.txt && \
    sed -i 's/np\\.int/int/g' yolov5/utils/dataloaders.py

WORKDIR /workspace
COPY docker/requirements-smoke.txt /workspace/docker/requirements-smoke.txt
COPY scripts /workspace/scripts

CMD ["bash"]
