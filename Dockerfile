FROM nvidia/cuda:9.0-base-ubuntu16.04
FROM nvidia/cuda:9.0-runtime-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

COPY /. /.

RUN	apt-get update && \
    apt-get install emacs24 -y && \
    apt-get install lsof -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y python3.6 && \
    apt-get install -y python3.6-dev && \
    apt-get install -y python3-pip && \
    apt-get install -y python3.6-venv && \
    apt-get install -y git-core && \
    python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install wheel && \
    pip install -r requirements.txt

EXPOSE 8000
