# Sagemaker SDXL Lora

# Prerequisite

Run sagemaker notebook instance with `conda_python3` kernel.

And clone this repository to the notebook instance.

The sagemaker notebook instance role should have `AmazonSageMakerFullAccess` and `AmazonEC2ContainerRegistryFullAccess` policy.

# Train

## Build Docker Image

Run [build_and_push.sh](/train/text-to-image/src/build_and_push.sh) on sagemaker notebook to build and push docker image to ECR.

## Text-to-Image Lora

open [train/text-to-image/sdxl-lora.ipynb](/train/text-to-image/sdxl-lora.ipynb) on `conda_python3` kernel.

# Test Trained Model

## Text-to-Image Lora

open [train/text-to-image/test/sdxl-lora.ipynb](/train/text-to-image/test/sdxl-lora.ipynb) on `conda_python3` kernel.

# Merge Trained LoRA Model

## Merge LoRA weigth to base model

After train the model, merge the trained model with the base model to fine-tune further.

open [train/text-to-image/merge/merge-lora-to-base.ipynb](/train/text-to-image/merge/merge-lora-to-base.ipynb) on `conda_python3` kernel.

## Upload to S3

open [train/text-to-image/merge/test-and-upload.ipynb](/train/text-to-image/merge/test-and-upload.ipynb) on `conda_python3` kernel.

After merge and upload, you can use the model for fine-tune furthre by setting s3 location to `train_model_uri` at [train/text-to-image/sdxl-lora.ipynb](/train/text-to-image/sdxl-lora.ipynb).
