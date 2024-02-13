# Sagemaker SDXL Lora

# Setup

Create sagemaker notebook instance with following configuration

- instance type: >ml.g5.2xlarge

Add following policies to the sagemaker notebook role

- IAM role: `AmazonEC2ContainerRegistryFullAccess`, `AWSCodeCommitFullAccess`

# Clone the repo

clone this repo (https://github.com/haandol/sagemaker-stable-diffusion-xl) to the notebook instance

# Build docker image

build docker image and push it to ECR.

```bash
cd sagemaker-stable-diffusion-xl/train/text-to-image
./build_and_push.sh
```

# Train

## Text-to-Image Lora

open [text-to-image/sdxl-lora.ipynb](/notebook/train/text-to-image/sdxl-lora.ipynb) on `conda_python3` kernel.

TBU

## ControlNet

open [controlnet/sdxl.ipynb](/notebook/train/controlnet/sdxl.ipynb) on `conda_python3` kernel.

TBU

# Inference

## Text-to-Image Lora

open [test/text-to-image/sdxl-lora.ipynb](/notebook/test/text-to-image/sdxl-lora.ipynb) on `conda_python3` kernel.

TBU

## ControlNet

open [test/controlnet/sdxl.ipynb](/notebook/test/controlnet/sdxl.ipynb) on `conda_python3` kernel.

TBU
