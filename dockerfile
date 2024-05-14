FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive 

COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y python3-pip python3-dev 
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -r requirements.txt