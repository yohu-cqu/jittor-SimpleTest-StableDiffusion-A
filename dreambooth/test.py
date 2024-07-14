import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import jittor as jt
import jittor.nn as nn
import argparse
import copy
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import transformers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from jittor import transform
from jittor.compatibility.optim import AdamW
from jittor.compatibility.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from JDiffusion import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers import DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
)

from examples.dreambooth.train import DreamBoothDataset, collate_fn, import_model_class_from_model_name_or_path, \
    encode_prompt

instance_data_dir = "/data/jittor2024/Problem2/JDiffusion/A/00/images"
instance_prompt = "style_00"
proxies = {
    # "http": "http://10.242.3.35:10811",
    # "https": "http://10.242.3.35:10811",
}
cached_path = "/data/jittor2024/Problem2/JDiffusion/cached_path"
pre_computed_encoder_hidden_states = None
pre_computed_class_prompt_encoder_hidden_states = None

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
    proxies=proxies,
    cache_dir=cached_path,
)

train_dataset = DreamBoothDataset(
    instance_data_root=instance_data_dir,
    instance_prompt=instance_prompt,
    class_data_root=None,
    class_prompt=None,
    class_num=100,
    tokenizer=tokenizer,
    size=512,
    center_crop=False,
    encoder_hidden_states=pre_computed_encoder_hidden_states,
    class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
    tokenizer_max_length=None,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples, False),
    num_workers=0,
)

text_encoder_cls = import_model_class_from_model_name_or_path("stabilityai/stable-diffusion-2-1", None)
text_encoder = text_encoder_cls.from_pretrained(
    "stabilityai/stable-diffusion-2-1", subfolder="text_encoder", revision=None, proxies=proxies,
    cache_dir=cached_path,
)

for step, batch in enumerate(train_dataloader):
    print(step, batch.keys())
    # Get the text embedding for conditioning
    encoder_hidden_states = encode_prompt(
        text_encoder,
        batch["input_ids"],
        batch["attention_mask"],
        text_encoder_use_attention_mask=False,
    )
    print(batch["input_ids"])

