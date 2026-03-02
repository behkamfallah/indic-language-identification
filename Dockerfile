# CUDA 11.8 runtime + cuDNN on Ubuntu 22.04 (compatible with driver 12.5)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System deps: Python, audio/libs, git
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        libsox-dev sox \
        git \
        && rm -rf /var/lib/apt/lists/*

# Ensure python/pip commands point to Python 3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Workdir inside the container
WORKDIR /app

# Copy dependency list and install (uses cu118 wheels)
COPY requirements_cuda.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_cuda.txt

# Copy the project code
COPY . .

CMD ["/bin/bash"]
