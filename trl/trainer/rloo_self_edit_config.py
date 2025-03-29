# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class RLOOSelfEditConfig(TrainingArguments):
    r"""
    Configuration class for the [`RLOOSelfEditTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        json_file (`str`):
            Path to the JSON file with prompts and targets.
        model_name (`str`):
            HuggingFace model name for the base model.
        max_model_len (`int`, *optional*, defaults to `12800`):
            Maximum model length for the vLLM model.
        max_lora_rank (`int`, *optional*, defaults to `8`):
            Maximum LoRA rank for the vLLM model.
        num_generations (`int`, *optional*, defaults to `5`):
            Number of generations to sample per prompt in the inner loop.
        inner_temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature for the inner loop.
        inner_top_p (`float`, *optional*, defaults to `0.95`):
            Top-p sampling value for the inner loop.
        inner_max_tokens (`int`, *optional*, defaults to `256`):
            Maximum tokens to generate in the inner loop.
        outer_lora_rank (`int`, *optional*, defaults to `8`):
            LoRA rank for the outer adapter.
        outer_lora_alpha (`int`, *optional*, defaults to `16`):
            LoRA alpha for the outer adapter.
        outer_lora_dropout (`float`, *optional*, defaults to `0.1`):
            LoRA dropout for the outer adapter.
        total_steps (`int`, *optional*, defaults to `1000`):
            Total training steps.
    """

    json_file: str = field(
        metadata={"help": "Path to the JSON file with prompts and targets."}
    )
    model_name: str = field(
        metadata={"help": "HuggingFace model name for the base model."}
    )
    max_model_len: int = field(
        default=12800,
        metadata={"help": "Maximum model length for the vLLM model."}
    )
    max_lora_rank: int = field(
        default=8,
        metadata={"help": "Maximum LoRA rank for the vLLM model."}
    )
    num_generations: int = field(
        default=5,
        metadata={"help": "Number of generations to sample per prompt in the inner loop."}
    )
    inner_temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for the inner loop."}
    )
    inner_top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p sampling value for the inner loop."}
    )
    inner_max_tokens: int = field(
        default=256,
        metadata={"help": "Maximum tokens to generate in the inner loop."}
    )
    outer_lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank for the outer adapter."}
    )
    outer_lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha for the outer adapter."}
    )
    outer_lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout for the outer adapter."}
    )
    total_steps: int = field(
        default=1000,
        metadata={"help": "Total training steps."}
    )
