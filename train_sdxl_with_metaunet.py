#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fine-tuning script for Stable Diffusion XL for text2image with custom UNet."""

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import swanlab
from swanlab.integration.accelerate import SwanLabTracker

# 导入自定义的UNet
from meta_unet_right import UNet2DConditionModel

#from my_weight_net import ResNet32,VNet

from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#




import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


from cal_FID_v import SemanticDistanceEvaluator
from cal_FID import FIDEvaluator



def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    MetaBatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(MetaModule):
    def __init__(self, num_classes, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = MetaLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)












#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#





# 检查diffusers版本
# check_min_version("0.35.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    import torch_npu

    torch.npu.config.allow_internal_format = False

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

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./sdxl",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="./male_female",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
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
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
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
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
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
        action="store_true",
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
        action="store_true",
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://huggingface.co/papers/2303.09556.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="swanlab",
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError(f"`--proportion_empty_prompts` must be in the range [0, 1].")

    return args

def encode_prompt(batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch["text"]  # 使用我们生成的文本列

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
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
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

    # There might have slightly performance improvement
    # by changing model_input.cpu() to accelerator.gather(model_input)
    return {"model_input": model_input.cpu()}

def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights



def get_meta_lr(step):
    #分段衰减策略：初始2e-5，1万步后降为5e-6，2万步后降为1e-6
    if step < 10000:
        return 2e-5
    elif step < 20000:
        return 5e-6
    else:
        return 1e-6


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # 初始化SwanLabTracker
    swanlab_tracker = SwanLabTracker(
        "SDXL-Training",
        experiment_name="SDXL_Doctor_Headshots",
        description="Training SDXL on doctor headshots dataset",
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=swanlab_tracker,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    
    # 使用新的UNet类
    unet = UNet2DConditionModel()
    
    # 加载预训练权重
    if args.pretrained_model_name_or_path:
        # 加载原始UNet以获取权重
        from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
        original_unet = OriginalUNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        
        # 将权重转移到新的UNet
        unet.load_state_dict(original_unet.state_dict())
        del original_unet  # 删除原始UNet释放内存
        gc.collect()
    
    # 冻结vae和文本编码器
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    # 设置UNet为可训练
    unet.train()
    
    # For mixed precision training we cast all non-trainable weights to half-precision
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
        # 创建新的UNet实例
        ema_unet = UNet2DConditionModel()
        
        # 加载预训练权重
        if args.pretrained_model_name_or_path:
            from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
            original_unet = OriginalUNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
            )
            ema_unet.load_state_dict(original_unet.state_dict())
            del original_unet
            gc.collect()
        
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
    
    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
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
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
#####################################################################################
    # Optimizer creation
    params_to_optimize = unet.params()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    vnet = VNet(1, 100, 1).cuda()
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)
#####################################################################################

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            # 匹配所有图片格式（可根据需要添加其他格式）
            data_files["train"] = [
                os.path.join(args.train_data_dir, fname)
                for fname in os.listdir(args.train_data_dir)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/image_dataset#imagefolder

    # 添加文本列（根据文件名生成提示词）
    def generate_caption(examples):
        captions = []
        for image in examples[args.image_column]:
            # 获取图片路径
            file_path = image.filename
            # 提取文件名
            filename = os.path.basename(file_path)
            # 去掉扩展名
            filename_without_ext = os.path.splitext(filename)[0]
            # 根据文件名前缀生成提示词
            if "male" in filename_without_ext.lower():
                caption = "a headshot of a male doctor"
            elif "female" in filename_without_ext.lower():
                caption = "a headshot of a female doctor"
            else:
                # 默认提示词
                caption = "a professional headshot of a doctor"
            captions.append(caption)
        return {"text": captions}

    # 应用提示词生成函数
    dataset = dataset.map(
        generate_caption,
        batched=True,
        batch_size=100,
        num_proc=args.dataloader_num_workers,
        desc="Generating captions from filenames",
    )

    # Preprocessing the datasets.
    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode {interpolation=}.")
    train_resize = transforms.Resize(args.resolution, interpolation=interpolation)
    train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[args.image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
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
    )
    compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        new_fingerprint_for_vae = Hasher.hash((vae_path, args))
        train_dataset_with_embeddings = train_dataset.map(
            compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint
        )
        train_dataset_with_vae = train_dataset.map(
            compute_vae_encodings_fn,
            batched=True,
            batch_size=args.train_batch_size,
            new_fingerprint=new_fingerprint_for_vae,
        )
        precomputed_dataset = concatenate_datasets(
            [train_dataset_with_embeddings, train_dataset_with_vae.remove_columns(["image", "text"])], axis=1
        )
        precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)

    del compute_vae_encodings_fn, compute_embeddings_fn, text_encoder_one, text_encoder_two
    #del text_encoders, tokenizers, vae
    del text_encoders, tokenizers
    gc.collect()
    if is_torch_npu_available():
        torch_npu.npu.empty_cache()
    elif torch.cuda.is_available():
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
        precomputed_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
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
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )



#######################################################################################
    test_pic_dir = Path("test_pic")
    test_pic_dir.mkdir(parents=True, exist_ok=True)


    evaluator = SemanticDistanceEvaluator(
        benchmark_dir="./meta_data",
        vae_model=vae,
        semantic_space="inception",  # 或"vae"
        device="cuda"
    )


    #evaluator.precompute_benchmark_mean()
    evaluator.precompute_benchmark_stats()

    evaluator_FID = FIDEvaluator(benchmark_dir="./meta_data")
    evaluator_FID.precompute_benchmark_stats()




    vae2 = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae",
        torch_dtype=torch.float16  # 关键：显式半精度
    )


    if accelerator.is_local_main_process:
        epoch_progress_bar = tqdm(
            range(0, 1000),
            total=1000,
            desc="训练轮次",
            unit="epoch"
        )



    for epoch in range(0, 1000):
        train_loss = 0.0
    #-------------------------------------------------------------------------#
        #print(1)
        if epoch%5==0:
            #print("来一次")
    #这里创建了一个元mmeta_unet
            meta_unet = UNet2DConditionModel()

            meta_unet.load_state_dict(unet.state_dict())
            meta_unet = meta_unet.to(accelerator.device)  # 显式转移到训练设备
            meta_unet.train()

            #unet.to("cpu")
            #torch.cuda.empty_cache()

######
#现在要内循环训练meta_unet，目前以这个为准
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


#先对每个样本进行计算与meta样本中心的语义距离FID_v表示
            for step, batch in enumerate(train_dataloader):
                #print(step)
                if step>=1:
                    break
                with accelerator.accumulate(meta_unet):
                    # Sample noise that we'll add to the latents
                    model_input = batch["model_input"].to(accelerator.device)

#@@@@@@@@@
                    with torch.no_grad():  # 距离计算无需梯度
                    # 注意：model_input是VAE的潜变量，直接传入evaluator
                        distances = evaluator.compute_semantic_distance(model_input)  # 形状：[batch_size]
                        FID_v = distances.unsqueeze(1)  # 调整为[batch_size, 1]，匹配VNet输入格式
#@@@@@@@@@

                    noise = torch.randn_like(model_input)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                        )

                    bsz = model_input.shape[0]
                    if args.timestep_bias_strategy == "none":
                        # Sample a random timestep for each image without bias.
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                        )
                    else:
                        # Sample a random timestep for each image, potentially biased by the timestep weights.
                        # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                        weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
                            model_input.device
                        )
                        timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps).to(dtype=weight_dtype)

                    # time ids
                    def compute_time_ids(original_size, crops_coords_top_left):
                        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                        target_size = (args.resolution, args.resolution)
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=weight_dtype)
                        return add_time_ids

                    add_time_ids = torch.cat(
                        [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                    )

                    # Predict the noise residual
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                    
                    # 使用自定义UNet进行前向传播
                    model_pred = meta_unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]


                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                    # 计算损失
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    elif noise_scheduler.config.prediction_type == "sample":
                        # We set the target to latents here, but the model_pred will return the noise sample prediction.
                        target = model_input
                        # We will have to subtract the noise residual from the prediction to get the target sample.
                        model_pred = model_pred - noise
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        if noise_scheduler.config.prediction_type == "epsilon":
                            mse_loss_weights = mse_loss_weights / snr
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            mse_loss_weights = mse_loss_weights / (snr + 1)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    #accelerator.backward(loss)
                    #if accelerator.sync_gradients:
                    #    params_to_clip = meta_unet.params()
                    #    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    
                    v_lambda=vnet(FID_v)

                    #l_f_meta=v_lambda*FID_v
                    l_f_meta = torch.sum(v_lambda * loss) / len(FID_v)

                    meta_unet.zero_grad()

                    grads = torch.autograd.grad(l_f_meta, (meta_unet.params()), create_graph=True)


                    #optimizer.step()
                    #lr_scheduler.step()
                    #optimizer.zero_grad()
                    current_step = epoch * len(train_dataloader) + step
                    meta_lr = get_meta_lr(current_step)
                    meta_unet.update_params(lr_inner=meta_lr, source_params=grads)
                    del model_input, noise, noisy_model_input, add_time_ids, prompt_embeds, pooled_prompt_embeds
                    del model_pred, target, loss, avg_loss, v_lambda, l_f_meta, grads
                    # 强制清理未引用的 GPU 内存
                    torch.cuda.empty_cache()
######
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#上面是内循环更新了meta_unet，接下来要搞外循环更新vnet

            #应该从这开始
            original_unet = OriginalUNet2DConditionModel.from_pretrained(
                "./sdxl",
                subfolder="unet"
            )

            original_unet.load_state_dict(meta_unet.state_dict())
            original_unet = original_unet.to(dtype=torch.float16)

            pipe = StableDiffusionXLPipeline.from_pretrained(
                "./sdxl",
                unet=original_unet,
                vae=vae2,  # 使用训练中加载的VAE
                torch_dtype=torch.float16,  # 确保所有组件使用半精度
                safety_checker=None
            ).to("cuda")

            pipe.set_progress_bar_config(disable=True)


            #在这里使用cal_FID计算生成的一组图片与meta数据集的FID，然后利用FID进行梯度下降更新vnet参数
            num_gen_images = 20
            generated_images = []
            generator = torch.Generator(device="cuda").manual_seed(40)
            for _ in range(num_gen_images // args.train_batch_size):
                # 生成图像（使用更新后的meta_unet）
                #print(3)
                with torch.set_grad_enabled(True):
                    # 禁用随机采样的随机性，使用确定性推理
                    images = pipe(
                        prompt="a headshot of a doctor",
                        num_inference_steps=15,
                        generator=generator,
                        return_dict=False,
                        output_type="pt",
                        deterministic=True
                    )[0]
                    generated_images.append(images)
            # 2. 拼接生成图像为张量（形状：[num_gen_images, 3, H, W]）
            #generated_images = torch.cat(generated_images, dim=0).to(accelerator.device)
            #generated_tensor = torch.stack([transforms.ToTensor()(img) for img in generated_images]).to(accelerator.device)
            generated_tensor = torch.cat(generated_images, dim=0).to(accelerator.device)
            #fid_score = fid_evaluator.compute_fid(generated_images)
            #fid_loss = torch.tensor(fid_score, device=accelerator.device, requires_grad=True)
            fid_loss = evaluator_FID.compute_fid(generated_tensor)
            optimizer_vnet.zero_grad()
            accelerator.backward(fid_loss)
            accelerator.clip_grad_norm_(vnet.params(), args.max_grad_norm)
            optimizer_vnet.step()

            # 记录图像到SwanLab
            #accelerator.log({"generated_image": swanlab.Image(image, caption=prompt)}, step=global_step)
            #unet.to(accelerator.device)                    
            # 清理内存
            #del pipe,meta_unet,images
            #torch.cuda.empty_cache()
            del meta_unet, original_unet  # 新增删除original_unet和grads
            del pipe, images, generated_images, generated_tensor  # 清理生成图像相关张量
            gc.collect()  # 强制Python垃圾回收
            torch.cuda.empty_cache()  
                # 1. 拆解管道，释放子组件（关键步骤）
            if 'pipe' in locals():
                
                if hasattr(pipe, 'unet'):
                    del pipe.unet
                if hasattr(pipe, 'vae'):
                    del pipe.vae
                if hasattr(pipe, 'text_encoder'):
                    del pipe.text_encoder
                if hasattr(pipe, 'text_encoder_2'):
                    del pipe.text_encoder_2
                del pipe  # 最后删除管道本身

            
            if 'meta_unet' in locals():
                meta_unet.cpu()  # 移到CPU，切断GPU关联
                del meta_unet
            if 'original_unet' in locals():
                original_unet.cpu()
                del original_unet

            # 3. 清理生成图像张量（包括日志缓存）
            if 'generated_images' in locals():
                for img in generated_images:
                    del img  # 逐个删除列表元素
                del generated_images
            if 'generated_tensor' in locals():
                generated_tensor = generated_tensor.cpu()  # 移到CPU
                del generated_tensor
            if 'images' in locals():
                for img in images:
                    del img
                del images

            # 4. 强制清理GPU内存（带同步操作）
            torch.cuda.synchronize()  # 等待所有GPU操作完成
            torch.cuda.empty_cache()  # 释放未引用内存
            gc.collect()  # 回收CPU内存

            # 5. 重置分布式训练状态（针对accelerator）
            if 'accelerator' in locals():
                accelerator.clear()  # 清空accelerator的临时缓存
            # #######################################


#-----------------------------------------------------------------------#


        for step, batch in enumerate(train_dataloader):
            if step>10:
                break
            print(epoch)
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
                if args.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
                        model_input.device
                    )
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps).to(dtype=weight_dtype)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                
                # 使用自定义UNet进行前向传播
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]


                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                # 计算损失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

#######################################################################################

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)


        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                # create pipeline
                vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                    revision=args.revision,
                    variant=args.variant,
                )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    unet=accelerator.unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                if args.prediction_type is not None:
                    scheduler_args = {"prediction_type": args.prediction_type}
                    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = (
                    torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    if args.seed is not None
                    else None
                )
                pipeline_args = {"prompt": args.validation_prompt}

                with autocast_ctx:
                    images = [
                        pipeline(**pipeline_args, generator=generator, num_inference_steps=25).images[0]
                        for _ in range(args.num_validation_images)
                    ]

                # 使用SwanLab记录验证图片
                swanlab_images = [swanlab.Image(image, caption=args.validation_prompt) for image in images]
                accelerator.log({"validation": swanlab_images}, step=global_step)

                del pipeline
                if is_torch_npu_available():
                    torch_npu.npu.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

        if accelerator.is_local_main_process:
            epoch_progress_bar.update(1)
            # 可选：显示当前 epoch 信息
            epoch_progress_bar.set_postfix({"当前 epoch": epoch, "总 epoch": args.num_train_epochs})

        # 每个epoch结束时执行（放在epoch循环的最末尾）
        def clear_memory():
            # 1. 清理元学习/生成相关的临时模型和张量（根据你的代码变量名调整）
            for var in ['meta_unet', 'original_unet', 'pipe', 'generated_images', 'generated_tensor', 'images']:
                if var in locals():
                    del locals()[var]
            # 2. 清理主训练循环的中间变量
            for var in ['model_input', 'noise', 'noisy_model_input', 'add_time_ids', 'prompt_embeds', 'model_pred', 'target', 'loss']:
                if var in locals():
                    del locals()[var]
            # 3. 清理数据加载器缓存（避免数据集迭代器持有张量引用）
            if 'train_dataloader' in locals():
                train_dataloader.dataset = None  # 清空数据集引用
                train_dataloader = None  # 重置数据加载器（下一epoch重新初始化）
            
            # 4. 强制Python垃圾回收（清理无引用的CPU对象）
            gc.collect()
            
            # 5. 同步CUDA流（确保所有GPU操作完成，否则缓存无法释放）
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 关键：等待所有GPU任务结束
                torch.cuda.empty_cache()  # 释放未引用的GPU内存
                torch.cuda.reset_max_memory_allocated()  # 重置内存统计
                torch.cuda.reset_max_memory_cached()
            
            # 6. 重置accelerator状态（针对分布式训练）
            if 'accelerator' in locals():
                accelerator.free_memory()  # 释放accelerator管理的临时内存

        # 调用清理函数
        clear_memory()

    if accelerator.is_local_main_process:
        epoch_progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        vnet_save_path = os.path.join(args.output_dir, "vnet_final.pt")
        torch.save(vnet.state_dict(), vnet_save_path)
        logger.info(f"VNet已保存至: {vnet_save_path}")


        # Serialize pipeline.
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        if args.prediction_type is not None:
            scheduler_args = {"prediction_type": args.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline = pipeline.to(accelerator.device)
            generator = (
                torch.Generator(device=accelerator.device).manual_seed(args.seed)
                if args.seed is not None
                else None
            )

            with autocast_ctx:
                images = [
                    pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                    for _ in range(args.num_validation_images)
                ]

            # 记录最终生成的图片
            swanlab_images = [swanlab.Image(image, caption=args.validation_prompt) for image in images]
            accelerator.log({"test": swanlab_images}, step=global_step)

        if args.push_to_hub:
            save_model_card(
                repo_id=repo_id,
                images=images,
                validation_prompt=args.validation_prompt,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
