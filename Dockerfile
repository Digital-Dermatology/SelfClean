FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y libtcmalloc-minimal4
RUN apt-get install -y sox
RUN apt-get install -y git

RUN pip install --upgrade pip

WORKDIR /workspace

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt --upgrade --no-cache-dir

COPY . /workspace/
RUN git config --global --add safe.directory '*'

ENV PYTHONPATH "$PYTHONPATH:./"
