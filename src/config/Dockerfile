# tensorflow image will pull down the required nvidia cuda image for you as well
FROM tensorflow/tensorflow:latest-gpu-py3
RUN apt-get update
RUN apt-get install -y git wget rsync python-gdcm tmux
ADD requirements.txt /app/
RUN pip install -r /app/requirements.txt
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN pip install opencv-python
# open port for tensorboard
EXPOSE 6006