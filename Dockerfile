FROM openvino/ubuntu18_dev:latest
COPY requirements.txt /tmp/
USER root
WORKDIR /work

RUN apt-get update && apt-get install -y vim fish git gcc libmariadb-dev tmux

RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

SHELL [ "fish" ]