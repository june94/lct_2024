FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop ffmpeg libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
#RUN apt-get update && apt-get install -y gcc zip curl ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \

RUN pip install ultralytics==8.2.31 sahi==0.11.16
RUN pip install streamlit==1.35.0

# working opencv-python                 4.5.5.64

#TODO optional install onnx-runtime and tensorrt 

# add code path to PYTHONPATH
ENV PYTHONPATH /lct_2024:$PYTHONPATH

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser

WORKDIR /home/appuser
ENV PATH="/home/appuser/.local/bin:${PATH}""""