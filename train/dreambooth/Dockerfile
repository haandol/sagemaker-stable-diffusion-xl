FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker

WORKDIR /opt/ml/code

COPY requirements.txt /opt/ml/code/
RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

COPY train_dreambooth.py /opt/ml/code/

ENV SAGEMAKER_PROGRAM train_dreambooth.py
