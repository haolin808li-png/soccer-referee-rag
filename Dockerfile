FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/referee-ai
COPY pyproject.toml ./
COPY src ./src
RUN python3 -m pip install --upgrade pip && python3 -m pip install -e '.[video,train,rag]'
COPY . .
CMD ["bash"]
