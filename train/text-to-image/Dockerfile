# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

WORKDIR /opt/ml/code

COPY requirements.txt /opt/ml/code/
RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY sdxl-lora-sagemaker.py /opt/ml/code/

ENV SAGEMAKER_PROGRAM sdxl-lora-sagemaker.py
