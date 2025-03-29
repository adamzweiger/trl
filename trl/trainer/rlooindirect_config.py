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
from typing import Optional, List, Dict, Any

from transformers import TrainingArguments


@dataclass
class RLOOIndirectConfig(TrainingArguments):
    r"""
    Configuration class for the [`RLOOIndirectTrainer`].

    This class inherits from [`~transformers.TrainingArguments`] and includes specific parameters for
    RLOO (REINFORCE Leave-One-Out) training using vLLM for generation with PEFT adapters and an indirect
    reward function specified via a Python file path.

    Parameters:
        > Parameters specific to RLOO Indirect training

        kl_coef (`float`, *optional*, defaults to `0.05`):
            Coefficient for the KL divergence penalty between the policy and the reference policy.
        rloo_k (`int`, *optional*, defaults to `2`):
            Number of online samples (generations) per prompt for REINFORCE Leave-One-Out (RLOO). Must be >= 2.
        normalize_reward (`bool`, *optional*, defaults to `False`):
            Whether to normalize the rewards obtained from the reward function before calculating advantages.
        reward_clip_range (`float`, *optional*, defaults to `10.0`):
            The range to clip rewards to if `normalize_reward` is `True`. Rewards are clipped to `[-reward_clip_range, reward_clip_range]`.
        normalize_advantage (`bool`, *optional*, defaults to `False`):
            Whether to normalize the calculated advantages.
        token_level_kl (`bool`, *optional*, defaults to `False`):
            Whether to apply the KL penalty at the token level (inside the reward calculation) or at the sequence level (added to the loss).
        reward_fn_path (`str`, *optional*, defaults to `None`):
            Path to the Python file containing the reward function. This file must define a function named `reward_fn` with the signature `reward_fn(llm: vllm.LLM, prompt_text: str, completions_text: List[str], target: Any, **kwargs) -> List[float]`.
            The function should return a list of k reward scores (floats, ideally between 0 and 1) corresponding to the k completions.
        reward_fn_kwargs (`Dict[str, Any]`, *optional*, defaults to `None`):
            Dictionary of keyword arguments to pass to the external reward function (`reward_fn`).

        > Parameters specific to PEFT (LoRA)

        lora_rank (`int`, *optional*, defaults to `8`):
            The rank of the LoRA update matrices.
        lora_alpha (`int`, *optional*, defaults to `16`):
            The alpha parameter for LoRA scaling (`scaling = lora_alpha / lora_rank`).
        lora_dropout (`float`, *optional*, defaults to `0.1`):
            The dropout probability for LoRA layers.
        lora_target_modules (`List[str]`, *optional*, defaults to `None`):
            List of module names or regex patterns to apply LoRA to. If `None`, PEFT will attempt to automatically infer target modules based on the model architecture.

        > Parameters inherited from TrainingArguments (relevant ones mentioned)

        per_device_train_batch_size (`int`, *optional*, defaults to `8`):
            Batch size per GPU/TPU core/CPU for training. Note that the effective batch size used for dataloading will be `per_device_train_batch_size // rloo_k`.
        gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of updates steps to accumulate gradients for before performing a backward/update pass.
        learning_rate (`float`, *optional*, defaults to `1.41e-5`):
             The initial learning rate for AdamW optimizer. RLOO typically uses smaller learning rates.
        num_train_epochs (`float`, *optional*, defaults to `3.0`):
            Total number of training epochs to perform.
        max_steps (`int`, *optional*, defaults to `-1`):
             If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
        logging_steps (`int` or `float`, *optional*, defaults to 500):
             Number of update steps between two logs if `logging_strategy="steps"`. Can be a float < 1 to log every fraction of an epoch.
        save_steps (`int` or `float`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Can be a float < 1 to save every fraction of an epoch.
        output_dir (`str`):
             The output directory where the model predictions and checkpoints will be written.
        seed (`int`, *optional*, defaults to `42`):
            Random seed that will be set at the beginning of training.

        > Parameters for vLLM Generation

        max_completion_length (`int`, *optional*, defaults to `256`):
            Maximum number of new tokens to generate for completions.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature for generation. Higher values make the output more random. Lower values make it more deterministic.
        top_p (`float`, *optional*, defaults to `1.0`):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
        top_k (`int` or `None`, *optional*, defaults to `None`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k is not applied.
        max_lora_rank (`int`, *optional*, defaults to `64`):
            The maximum LoRA rank to support in the vLLM engine. Should be >= `lora_rank`.

        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation (e.g., 'float16', 'bfloat16', 'auto').
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            Maximum sequence length supported by the vLLM engine. If `None`, uses the model's default.

        > Other Parameters

        exp_name (`str`, *optional*, defaults to `rloo_indirect`):
            Name for the experiment (used for run naming).
        num_sample_generations (`int`, *optional*, defaults to `10`):
             Number of times to generate sample completions during training for logging/evaluation. 0 to disable.
        missing_eos_penalty (`float`, *optional*, defaults to `None`):
            Penalty to apply to the score if the generated response does not contain the EOS token. If `None`, no penalty is applied.
        stop_token (`str`, *optional*, defaults to `"eos"`):
            If set to "eos", use the tokenizer's EOS token ID as the stop token for response truncation. Otherwise, this argument is ignored.
        stop_token_id (`int`, *optional*, defaults to `None`):
            Explicit token ID to use for truncating responses. Overrides `stop_token` if set.
    """

    # RLOO Specifics
    kl_coef: float = field(default=0.05, metadata={"help": "KL penalty coefficient."})
    rloo_k: int = field(
        default=2,
        metadata={"help": "REINFORCE Leave-One-Out (RLOO) number of online samples per prompt (must be >= 2)."},
    )
    normalize_reward: bool = field(default=False, metadata={"help": "Whether to normalize rewards."})
    reward_clip_range: float = field(default=10.0, metadata={"help": "Clip range for normalized rewards."})
    normalize_advantage: bool = field(default=False, metadata={"help": "Whether to normalize advantages."})
    token_level_kl: bool = field(
        default=False,
        metadata={"help": "Whether to use token-level KL penalty or sequence-level KL penalty."},
    )
    reward_fn_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the Python file defining the `reward_fn(llm, prompt_text, completions_text, target, **kwargs)` function."
        },
    )
    reward_fn_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=dict, metadata={"help": "Keyword arguments to pass to the reward function."}
    )
    num_ppo_epochs: int = field(
        default=1, metadata={"help": "Number of optimization epochs per batch of generated data (usually 1 for RLOO)."}
    ) # Keep name for consistency with original RLOO, but clarify meaning

    # PEFT Specifics
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout."})
    lora_target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Modules to apply LoRA to (e.g., ['q_proj', 'v_proj'] or None for auto)."}
    )

    # Generation Specifics (using vLLM)
    max_completion_length: int = field(default=256, metadata={"help": "Maximum number of tokens to generate."})
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature."})
    top_p: float = field(default=1.0, metadata={"help": "Nucleus sampling probability."})
    top_k: Optional[int] = field(default=None, metadata={"help": "Top-k sampling."})
    max_lora_rank: int = field(default=64, metadata={"help": "Max LoRA rank for vLLM engine."})

    # vLLM Engine Parameters
    use_vllm: bool = field(default=True, metadata={"help": "Force use_vllm=True for this trainer."})
    vllm_device: str = field(
        default="auto", metadata={"help": "Device for vLLM ('auto' or specific cuda device like 'cuda:1')."}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9, metadata={"help": "GPU memory utilization for vLLM."}
    )
    vllm_dtype: str = field(default="auto", metadata={"help": "Data type for vLLM ('auto', 'float16', 'bfloat16')."})
    vllm_max_model_len: Optional[int] = field(default=None, metadata={"help": "Max model length for vLLM."})

    # Other config
    exp_name: str = field(default="rloo_indirect", metadata={"help": "Experiment name."})
    num_sample_generations: int = field(
        default=10, metadata={"help": "Number of sample generation rounds during training (0 to disable)."}
    )
    missing_eos_penalty: Optional[float] = field(
        default=None, metadata={"help": "Penalty if EOS token is missing in response."}
    )
    stop_token: str = field(default="eos", metadata={"help": "Use 'eos' to truncate responses at EOS token."})
    stop_token_id: Optional[int] = field(default=None, metadata={"help": "Explicit stop token ID."})

    # Inherited defaults (can be overridden)
    learning_rate: float = field(default=1.41e-5, metadata={"help": "Learning rate."})
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Keep columns needed by reward_fn (e.g., 'target')."}
    ) # Keep needed columns

    def __post_init__(self):
        super().__post_init__() # Call parent's post_init
        if not self.use_vllm:
            raise ValueError("RLOOIndirectTrainer requires use_vllm=True.")
        if self.rloo_k < 2:
            raise ValueError("rloo_k must be >= 2 for REINFORCE Leave-One-Out.")
        if self.reward_fn_path is None or not os.path.isfile(self.reward_fn_path):
            raise FileNotFoundError(f"Reward function file not found at path: {self.reward_fn_path}")
        if self.lora_rank > self.max_lora_rank:
            raise ValueError(f"lora_rank ({self.lora_rank}) cannot be greater than max_lora_rank ({self.max_lora_rank})")

        # Adjust batch size for dataloader based on rloo_k
        if self.per_device_train_batch_size % self.rloo_k != 0:
             raise ValueError(f"per_device_train_batch_size ({self.per_device_train_batch_size}) must be divisible by rloo_k ({self.rloo_k})")
        self.local_dataloader_batch_size = self.per_device_train_batch_size // self.rloo_k

        # RLOO doesn't use PPO clipping
        # self.cliprange = 0.0 # Not used in RLOO loss

        # Determine dataset columns to keep - ensure 'prompt' and 'target' are kept
        # Assuming 'target' is needed by the custom reward function
        # Parent TrainingArguments removes unused columns based on model signature
        # We override this behavior slightly.
        if self.remove_unused_columns:
             self._saved_remove_unused_columns = True
             self.remove_unused_columns = False # Temporarily disable to inspect dataset
        else:
             self._saved_remove_unused_columns = False
