#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image."""

import sys
import json
import tarfile
import argparse
import functools
import gc
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

MODEL_UNTARRED_PATH = "/tmp"


def get_validation_prompts(args):
    """Read the dataset info file and return the instance prompt and the (optional) class prompt, validation_prompt"""
    try:
        with open(os.path.join(args.train, "dataset_info.json"), "r") as f:
            dataset_info = json.loads(f.read())
        validation_prompts = dataset_info.get("validation_prompts")
        return validation_prompts
    except Exception as e:
        logger.error(
            f"Could not find/read/parse dataset_info.json file, exception: '{e}'. "
            f"To train the model, please include "
            "dataset_info file with json file of the format {'validation_prompts': [<<validation_prompt1>>, <<validation_prompt2>>, ...]}"
        )
        raise


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def str2bool(x: str) -> bool:
    return x.lower() in ('true', '1')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL"),
        help="Directory with the pre-trained model.",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING"),
        help="Directory with the training data.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        type=str2bool,
        default=True,
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        type=str2bool,
        default=True,
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str2bool,
        default=False,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--force_snr_gamma",
        type=str2bool,
        default=False,
        help=(
            "When using SNR gamma with rescaled betas for zero terminal SNR, a divide-by-zero error can cause NaN"
            " condition when computing the SNR with a sigma value of zero. This parameter overrides the check,"
            " allowing the use of SNR gamma with a terminal SNR model. Use with caution, and closely monitor results."
        ),
    )
    parser.add_argument(
        "--use_ema",
        type=str2bool,
        default=False,
        help="Whether to use EMA model.",
    )
    parser.add_argument(
        "--allow_tf32",
        type=str2bool,
        default=False,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        type=str2bool,
        default=False,
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        type=str2bool,
        default=True,
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0,
        help="The scale of noise offset.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=("Run fine-tuning validation every X epochs. The validation process consists of running the prompt"),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "text2image-fine-tune"),
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return {"model_input": model_input.cpu()}


def main(args):
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        transformers.set_seed(args.seed)

    accelerator_project_config = ProjectConfiguration(
        logging_dir=args.model_dir,
        total_limit=args.checkpoints_total_limit
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        MODEL_UNTARRED_PATH, subfolder="tokenizer", revision=None, use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        MODEL_UNTARRED_PATH, subfolder="tokenizer_2", revision=None, use_fast=False
    )
    
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        MODEL_UNTARRED_PATH, revision=None,
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        MODEL_UNTARRED_PATH, revision=None, subfolder="text_encoder_2",
    )

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        MODEL_UNTARRED_PATH, subfolder="scheduler",
    )
    # Check for terminal SNR in combination with SNR Gamma
    if (
        args.snr_gamma
        and not args.force_snr_gamma
        and (
            hasattr(noise_scheduler.config, "rescale_betas_zero_snr") and noise_scheduler.config.rescale_betas_zero_snr
        )
    ):
        raise ValueError(
            f"The selected noise scheduler for the model {MODEL_UNTARRED_PATH} uses rescaled betas for zero SNR.\n"
            "When this configuration is present, the parameter --snr_gamma may not be used without parameter --force_snr_gamma.\n"
            "This is due to a mathematical incompatibility between our current SNR gamma implementation, and a sigma value of zero."
        )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        MODEL_UNTARRED_PATH, subfolder="text_encoder", revision=None
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        MODEL_UNTARRED_PATH, subfolder="text_encoder_2", revision=None
    )
    vae_path = (
        MODEL_UNTARRED_PATH
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=None
    )
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_UNTARRED_PATH, subfolder="unet", revision=None
    )

    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            MODEL_UNTARRED_PATH, subfolder="unet", revision=None,
        )
        ema_unet = EMAModel(
            ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for unet.")
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        logger.info("Using AdamW8bit optimizer for training.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        logger.info("Using AdamW optimizer for training.")
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    if not Path(args.train).exists():
        raise ValueError(
            f"Instance {args.train} images root doesn't exists.")

    dataset = load_dataset(
        "imagefolder",
        data_dir=args.train,
        cache_dir=args.cache_dir,
    )
    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # 6. Get the column names for input/target.
    image_column = 'image'
    caption_column = 'text'

    # Preprocessing the datasets.
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # Preprocessing the datasets.
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            if args.random_flip and random.random() < 0.5:
                # flip
                x1 = image.width - x1
                image = train_flip(image)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory. We will pre-compute the VAE encodings too.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
        caption_column=caption_column,
    )
    compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        new_fingerprint_for_vae = Hasher.hash("vae")
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)
        train_dataset = train_dataset.map(
            compute_vae_encodings_fn,
            batched=True,
            batch_size=args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
            new_fingerprint=new_fingerprint_for_vae,
        )

    del text_encoders, tokenizers, vae
    gc.collect()
    torch.cuda.empty_cache()

    def collate_fn(examples):
        model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples])
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
        pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, init_kwargs={"wandb": {"allow_val_change": True}})
        if is_wandb_available():
            wandb.config.update(dict(vars(args)), allow_val_change=True)

    # Train!
    total_batch_size = args.batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Sample noise that we'll add to the latents
                model_input = batch["model_input"].to(accelerator.device)
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                    )

                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                prompt_embeds = prompt_embeds
                model_pred = unet(
                    noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                ).sample

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join("/opt/ml/checkpoints", f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            validation_prompts = get_validation_prompts(args)

            if validation_prompts is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating images with prompt:"
                    f" {validation_prompts}."
                )
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                # create pipeline
                vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                    revision=None,
                )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    MODEL_UNTARRED_PATH,
                    vae=vae,
                    unet=accelerator.unwrap_model(unet),
                    revision=None,
                    torch_dtype=weight_dtype,
                )
                if args.prediction_type is not None:
                    scheduler_args = {"prediction_type": args.prediction_type}
                    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

                with torch.cuda.amp.autocast():
                    images = [
                        pipeline(prompt=validation_prompt, generator=generator, num_inference_steps=25).images[0]
                        for validation_prompt in validation_prompts
                    ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {validation_prompts[i]}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        # Serialize pipeline.
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=None,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL_UNTARRED_PATH,
            unet=unet,
            vae=vae,
            revision=None,
            torch_dtype=weight_dtype
        )
        if args.prediction_type is not None:
            scheduler_args = {"prediction_type": args.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(args.model_dir)

        # run inference
        images = []
        validation_prompts = get_validation_prompts(args)
        if validation_prompts:
            pipeline = pipeline.to(accelerator.device)
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
            with torch.cuda.amp.autocast():
                images = [
                    pipeline(validation_prompts[i], num_inference_steps=25, generator=generator).images[0]
                    for i in range(len(validation_prompts))
                ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {validation_prompts[i]}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

    accelerator.end_training()


def extract_pretrained_model_tarball(pretrained_tarball_dir: str, extracted_model_dir: str) -> None:
    """Extract the pre-trained model tarball to the extracted model directory."""
    input_model_path = next(Path(pretrained_tarball_dir).glob("*.tar.gz"))
    with tarfile.open(input_model_path, "r") as saved_model_tar:
        saved_model_tar.extractall(extracted_model_dir)


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Running training scripts with arguments: {args}.")

    extract_pretrained_model_tarball(args.pretrained_model, MODEL_UNTARRED_PATH)
    main(args)
