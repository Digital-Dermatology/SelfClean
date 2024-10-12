FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y apt-transport-https
RUN apt-get install -y libtcmalloc-minimal4
RUN apt-get install -y libomp-dev
RUN apt-get install -y sox
RUN apt-get install -y git
RUN apt-get install -y gcc g++ python3-dev python-dev
RUN apt-get clean

RUN pip install --upgrade pip

WORKDIR /workspace

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
COPY requirements.extras.txt /assets/requirements.extras.txt
RUN pip install -r /assets/requirements.txt --upgrade --no-cache-dir
RUN pip install -r /assets/requirements.extras.txt --upgrade --no-cache-dir

COPY . /workspace/
RUN git config --global --add safe.directory '*'

ENV PYTHONPATH "$PYTHONPATH:./"
