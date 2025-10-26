FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /workspace/

RUN pip install --upgrade pip && pip install wandb