#!/bin/bash

# The name of our algorithm
algorithm_name=sdxl-text-to-image-lora-training-gpu

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-ap-northeast-2}

commit_tag=$(git rev-parse --short=10 HEAD)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${commit_tag}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${region}.amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

docker push ${fullname}

echo "Pushed image: ${fullname}"