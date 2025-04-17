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
from typing import Optional, List, Dict, Any, Union

from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy, SchedulerType, HubStrategy # Import necessary enums/types


@dataclass
class RLOOIndirectConfig(TrainingArguments):
    r"""
    Configuration class for the PPO-style [`RLOOIndirectTrainer`].

    This class inherits from [`~transformers.TrainingArguments`] and includes specific parameters for
    RLOO (REINFORCE Leave-One-Out) training using a PPO-like optimization loop, vLLM for generation
    with PEFT adapters, and an indirect reward function (e.g., via ZMQ or external script).

    Parameters:
        model_name_or_path (`str`, *optional*, defaults to `None`):
            Path to pretrained model or model identifier from huggingface.co/models. Used for loading the base model.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow custom models defined on the Hub. Use with caution.
        torch_dtype (`str`, *optional*, defaults to `None`):
            Override the default `torch.dtype` and load the model with another dtype ('auto' for automatic).

        > Parameters specific to RLOO & PPO-Style Training

        kl_coef (`float`, *optional*, defaults to `0.05`):
            Coefficient for the KL divergence penalty between the policy and the reference policy (base model).
            Applied sequence-wise before advantage calculation in this implementation.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        rloo_k (`int`, *optional*, defaults to `2`):
            Number of online samples (generations) per prompt for REINFORCE Leave-One-Out (RLOO). Must be >= 2.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of optimization epochs to perform on the generated batch of data during the PPO optimization phase.
        num_mini_batches (`int`, *optional*, defaults to `1`):
            Number of mini-batches to split the full rollout batch into during the PPO optimization phase.
            `batch_size` must be divisible by `num_mini_batches * world_size`.
        batch_size (`int`, *optional*, defaults to `None`):
            Number of prompts processed in one full RLOO rollout *across all devices*. If `None`, it will be calculated
            as `per_device_train_batch_size * gradient_accumulation_steps * num_mini_batches * world_size`
            in the trainer initialization. This defines the size of the experience buffer generated before the PPO phase.
        total_episodes (`int`, *optional*, defaults to `None`):
            Total number of prompt-generation episodes to train on. If `None`, it's derived from `num_train_epochs`
            and the dataset size. This determines the total amount of generated data. `num_total_batches` (rollouts)
            is derived from `total_episodes / batch_size`.
        normalize_reward (`bool`, *optional*, defaults to `False`):
            Whether to normalize the rewards obtained from the reward function using running mean and std deviation *before* advantage calculation.
        reward_clip_range (`float`, *optional*, defaults to `10.0`):
            The range to clip rewards to *if* `normalize_reward` is `True`. Rewards are clipped to `[-reward_clip_range, reward_clip_range]`.
        normalize_advantage (`bool`, *optional*, defaults to `True`):
            Whether to normalize the calculated advantages using running mean and std deviation. Recommended for PPO stability.
        task_type (`str`, *optional*, defaults to `math`):
            Task type identifier ('math', 'fewshot', 'cpt') passed to the reward function.
        reward_fn_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Dictionary of keyword arguments to pass to the external reward function (e.g., via ZMQ context).
        missing_eos_penalty (`float`, *optional*, defaults to `None`):
            Penalty to apply to the reward score if the generated response does not contain the EOS token. If `None`, no penalty is applied.

        > Parameters specific to PEFT (LoRA)

        lora_rank (`int`, *optional*, defaults to `8`):
            The rank of the LoRA update matrices.
        lora_alpha (`int`, *optional*, defaults to `16`):
            The alpha parameter for LoRA scaling (`scaling = lora_alpha / lora_rank`).
        lora_dropout (`float`, *optional*, defaults to `0.1`):
            The dropout probability for LoRA layers.
        lora_adapter_name (`str`, *optional*, defaults to `"rloo_adapter"`):
            The name to use for the LoRA adapter being trained and saved/loaded by the trainer and vLLM.

        > Parameters for vLLM Generation

        max_response_length (`int`, *optional*, defaults to `256`):
            Maximum number of *new* tokens to generate for completions during the rollout phase.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature for generation. Higher values make the output more random.
        top_p (`float`, *optional*, defaults to `1.0`):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
        stop_token (`str`, *optional*, defaults to `"eos"`):
            If set to "eos", use the tokenizer's EOS token ID as the stop sequence for vLLM generation. Otherwise, this argument is ignored for the stop ID.
        stop_token_id (`int`, *optional*, defaults to `None`):
            Explicit token ID to use for truncating responses *after* generation. Overrides `stop_token` behavior for truncation if set.

        > Parameters for vLLM API Interaction

        vllm_api_url (`str`):
            **Required.** URL of the running vLLM OpenAI API compatible server (e.g., `http://localhost:8000`).
        adapter_save_dir (`str`, *optional*, defaults to `"./adapter_checkpoints"`):
            Directory to save temporary adapters between training steps for vLLM to load. Will be created if it doesn't exist.
        vllm_adapter_name (`str`, *optional*, defaults to `"dynamic_training_adapter"`):
            The fixed name used when dynamically loading/unloading the adapter via the vLLM API. Should match the LoRA adapter name used internally unless there's a specific reason.

        > Other Parameters

        exp_name (`str`, *optional*, defaults to `rloo_indirect_ppo`):
            Name for the experiment (used for run naming and potentially logging).
        log_policy_entropy (`bool`, *optional*, defaults to `True`):
            Whether to compute and log the policy entropy during the PPO optimization phase.

        > Parameters inherited from TrainingArguments (Key overrides and notes)

        per_device_train_batch_size (`int`, *optional*, defaults to `8`):
            Number of prompts processed per device *per gradient accumulation step* during the PPO optimization phase.
            This is the **micro-batch size** for the inner PPO loop.
        gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of micro-batch steps to accumulate gradients for before performing a backward/update pass during the PPO optimization phase.
        learning_rate (`float`, *optional*, defaults to `1e-5`):
            The initial learning rate for AdamW optimizer. RL often requires smaller LRs.
        num_train_epochs (`float`, *optional*, defaults to `1.0`):
            Total number of training epochs to perform, defined in terms of passing over the dataset *to generate prompts*. Used to calculate `total_episodes` if not set directly.
        max_steps (`int`, *optional*, defaults to `-1`):
            If set to a positive number, the total number of **PPO optimization steps** to perform (i.e., `global_step` in the PPO phase). Overrides `num_train_epochs`/`total_episodes`.
        logging_steps (`float`, *optional*, defaults to `10`):
            Log every X **PPO optimization steps**. Can be < 1 to log fractionally based on `max_steps`.
        save_steps (`float`, *optional*, defaults to `100`):
            Save checkpoint every X **PPO optimization steps**. Can be < 1 to save fractionally based on `max_steps`.
        eval_steps (`float`, *optional*, defaults to `100`):
            Run evaluation every X **PPO optimization steps**. Can be < 1 to evaluate fractionally based on `max_steps`. Set <= 0 to disable periodic eval.
        output_dir (`str`):
            **Required.** The output directory where the model adapters and checkpoints will be written.
        seed (`int`, *optional*, defaults to `42`):
            Random seed for reproducibility.
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            **Recommended: `False`**. Whether to remove columns not required by the model's forward pass. RL often needs extra columns ('prompt', 'target', metadata) for rollouts and rewards.
    """

    # --- Model & Tokenizer ---
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Enable trusting remote code for tokenizer/model"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Override default model dtype ('auto' derived from weights)."}
    )

    # --- RLOO & PPO Specifics ---
    kl_coef: float = field(
        default=0.05, metadata={"help": "Coefficient for the KL divergence penalty (applied sequence-wise)."}
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    rloo_k: int = field(
        default=2,
        metadata={"help": "RLOO number of online samples per prompt (must be >= 2)."},
    )
    num_ppo_epochs: int = field(
        default=4, metadata={"help": "Number of optimization epochs per rollout batch in PPO phase."}
    )
    num_mini_batches: int = field(
        default=1, metadata={"help": "Number of mini-batches to split rollout batch into for PPO updates."}
    )
    batch_size: Optional[int] = field(
        default=None, metadata={"help": "Total prompts per rollout across all devices. Calculated if None."}
    )
    total_episodes: Optional[int] = field(
        default=None, metadata={"help": "Total prompt-generation episodes to train on. Derived from num_train_epochs if None."}
    )
    normalize_reward: bool = field(
        default=False, metadata={"help": "Normalize rewards before advantage calculation."}
    )
    reward_clip_range: float = field(
        default=10.0, metadata={"help": "Clip range for normalized rewards ([-X, X])."}
    )
    normalize_advantage: bool = field(
        default=True, metadata={"help": "Normalize calculated advantages."} # Default True for PPO
    )
    task_type: str = field(
        default="math", metadata={"help": "Task type ('math', 'fewshot', 'cpt') passed to reward function."}
    )
    reward_fn_kwargs: Dict[str, Any] = field( # Use dict directly as default_factory is implied for mutable types
        default_factory=dict, metadata={"help": "Keyword arguments for the reward function."}
    )
    missing_eos_penalty: Optional[float] = field(
        default=None, metadata={"help": "Penalty if EOS token is missing. None to disable."}
    )

    # --- PEFT Specifics ---
    lora_rank: int = field(
        default=8, metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "LoRA alpha scaling."}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "LoRA dropout."}
    )
    lora_adapter_name: str = field(
        default="rloo_adapter",
        metadata={"help": "Name for the trained LoRA adapter."}
    )

    # --- Generation Specifics (vLLM) ---
    max_response_length: int = field( # Renamed from max_completion_length
        default=256, metadata={"help": "Maximum number of NEW tokens to generate during rollouts."}
    )
    temperature: float = field(
        default=0.7, metadata={"help": "Sampling temperature for generation."}
    )
    top_p: float = field(
        default=1.0, metadata={"help": "Nucleus sampling (p) threshold."}
    )
    stop_token: str = field(
        default="eos", metadata={"help": "If 'eos', use tokenizer's EOS token ID as stop sequence for vLLM. Ignored otherwise."}
    )
    stop_token_id: Optional[int] = field(
        default=None, metadata={"help": "Explicit stop token ID for post-generation truncation."}
    )

    # --- vLLM API Interaction ---
    # Make required fields non-optional in the signature
    vllm_api_url: str = field(
        default=None,
        metadata={"help": "Required: URL of the running vLLM OpenAI API server (e.g., http://localhost:8000)."}
    )
    adapter_save_dir: str = field(
        default="./adapter_checkpoints", metadata={"help": "Directory to save temporary adapters for vLLM."}
    )
    vllm_adapter_name: str = field(
        default="dynamic_training_adapter",
        metadata={"help": "Fixed name for dynamically loading/unloading adapter via vLLM API."}
    )

    # --- Other Config ---
    exp_name: str = field(
        default="rloo_indirect_ppo", metadata={"help": "Experiment name for logging."}
    )
    log_policy_entropy: bool = field(
        default=True, metadata={"help": "Compute and log policy entropy during PPO."}
    )

    # --- Inherited/Overridden TrainingArguments ---
    output_dir: str = field(
        default=None,
        metadata={"help": "Required: Output directory for checkpoints and adapters."}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Micro-batch size per device during PPO optimization."}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Steps to accumulate gradients during PPO before optimizer step."}
    )
    learning_rate: float = field(
        default=1e-5, metadata={"help": "Initial learning rate for AdamW."} # Adjusted default
    )
    num_train_epochs: float = field(
        default=1.0, metadata={"help": "Epochs defined by passing over the dataset for prompt generation."} # Adjusted default
    )
    max_steps: int = field(
        default=-1, metadata={"help": "Total number of PPO optimization steps. Overrides num_train_epochs/total_episodes."}
    )
    logging_steps: float = field(
         default=10, metadata={"help": "Log every X PPO optimization steps."}
    )
    save_steps: float = field(
        default=100, metadata={"help": "Save checkpoint every X PPO optimization steps."}
    )
    eval_steps: float = field(
        default=100, metadata={"help": "Evaluate every X PPO optimization steps."}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Recommended False: Keep extra columns for RL rollouts/rewards."}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed."}
    )


    def __post_init__(self):
        # Perform validation checks after initializing TrainingArguments
        if not hasattr(self, "output_dir") or self.output_dir is None:
             raise ValueError("`output_dir` must be specified.")
        if not hasattr(self, "vllm_api_url") or self.vllm_api_url is None:
             raise ValueError("`vllm_api_url` must be specified.")

        # Call parent's post_init AFTER checking required fields inherited/overridden
        super().__post_init__()

        if self.rloo_k < 2:
            raise ValueError("`rloo_k` must be >= 2 for REINFORCE Leave-One-Out.")
        if self.num_mini_batches <= 0:
            raise ValueError("`num_mini_batches` must be positive.")

        # Ensure adapter save directory exists (do this only on main process potentially?)
        # It's generally safe to call makedirs everywhere, it handles existing dirs.
        os.makedirs(self.adapter_save_dir, exist_ok=True)

        # Warnings and informational messages
        if self.remove_unused_columns:
            print(
                "Warning: `remove_unused_columns` is True. Set to False if reward function needs extra dataset columns."
            )
        if self.kl_coef == 0.0:
             print("Info: `kl_coef` is 0.0. No KL penalty will be applied during training.")
        else:
             print(f"Info: Sequence-level KL penalty will be applied with `kl_coef`={self.kl_coef}.")

        if self.batch_size is not None and self.batch_size <= 0:
             raise ValueError("If specified, `batch_size` (prompts per rollout) must be positive.")
        if self.total_episodes is not None and self.total_episodes <= 0:
             raise ValueError("If specified, `total_episodes` must be positive.")

        # Convert steps from float to int if they are >= 1
        if self.logging_steps >= 1: self.logging_steps = int(self.logging_steps)
        if self.save_steps >= 1: self.save_steps = int(self.save_steps)
        if self.eval_steps >= 1: self.eval_steps = int(self.eval_steps)
