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
        model_name_or_path (`str`, *optional*, defaults to `None`):
            Path to pretrained model or model identifier from huggingface.co/models.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow for custom models defined on the Hub in their own modeling files. This option should only
            be set to `True` for repositories you trust and in which you have read the code, as it will execute code
            present on the Hub on your local machine.
        torch_dtype (`str`, *optional*, defaults to `None`):
            Override the default `torch.dtype` and load the model with another dtype. If 'auto', the dtype will be automatically derived from the model's weights.

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
            The `llm` object passed is typically the reference model used for KL computation, not the active policy model.
        reward_fn_kwargs (`Dict[str, Any]`, *optional*, defaults to `None`):
            Dictionary of keyword arguments to pass to the external reward function (`reward_fn`).
        num_ppo_epochs (`int`, *optional*, defaults to `1`):
            Number of optimization epochs per batch of generated data. Typically 1 for RLOO as new data is generated in each step.
        should_self_edit (`bool`, *optional*, defaults to `False`):
            Whether to use self-editing in the reward function. If `True`, the internal reward function will be called rather than the specified reward_fn_path.

        > Parameters specific to PEFT (LoRA)

        lora_rank (`int`, *optional*, defaults to `8`):
            The rank of the LoRA update matrices.
        lora_alpha (`int`, *optional*, defaults to `16`):
            The alpha parameter for LoRA scaling (`scaling = lora_alpha / lora_rank`).
        lora_dropout (`float`, *optional*, defaults to `0.1`):
            The dropout probability for LoRA layers.
        lora_target_modules (`List[str]`, *optional*, defaults to `None`):
            List of module names or regex patterns to apply LoRA to. If `None`, PEFT will attempt to automatically infer target modules based on the model architecture.
        lora_adapter_name (`str`, *optional*, defaults to `outer_lora`):
            The name of the LoRA adapter to load. If `None`, the default adapter name will be used. This is only relevant if you are loading a pretrained LoRA adapter.

        > Parameters for vLLM Generation

        max_completion_length (`int`, *optional*, defaults to `256`):
            Maximum number of new tokens to generate for completions.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature for generation. Higher values make the output more random. Lower values make it more deterministic.
        top_p (`float`, *optional*, defaults to `1.0`):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.

        > Parameters for vLLM API Interaction

        vllm_api_url (`str`, *optional*, defaults to `None`):
            URL of the running vLLM OpenAI API server (e.g., http://localhost:8000). This is required for vLLM generation.
        adapter_save_dir (`str`, *optional*, defaults to `"./adapter_checkpoints"`):
            Directory to save temporary adapters for vLLM requests. This directory will be created if it does not exist.
        vllm_adapter_name (`str`, *optional*, defaults to `"dynamic_training_adapter"`):
            The fixed name used when dynamically loading the adapter via the vLLM API. This is required for vLLM generation.

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
        log_policy_entropy (`bool`, *optional*, defaults to `True`):
            Whether to log the policy entropy during training.


        > Parameters inherited from TrainingArguments (relevant ones mentioned)

        per_device_train_batch_size (`int`, *optional*, defaults to `8`):
            Batch size (number of prompts) per GPU/TPU core/CPU for training. The RLOO trainer will generate `rloo_k` completions for each prompt.
        gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of updates steps to accumulate gradients for before performing a backward/update pass.
        learning_rate (`float`, *optional*, defaults to `1.41e-5`):
             The initial learning rate for AdamW optimizer. RLOO typically uses smaller learning rates.
        num_train_epochs (`float`, *optional*, defaults to `3.0`):
            Total number of training epochs to perform.
        max_steps (`int`, *optional*, defaults to `-1`):
             If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
        logging_steps (`int` or `float`, *optional*, defaults to 10):
             Number of update steps between two logs if `logging_strategy="steps"`. Can be a float < 1 to log every fraction of an epoch. Adjusted default for RL.
        save_steps (`int` or `float`, *optional*, defaults to 100):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Can be a float < 1 to save every fraction of an epoch. Adjusted default for RL.
        output_dir (`str`):
             The output directory where the model predictions and checkpoints will be written.
        seed (`int`, *optional*, defaults to `42`):
            Random seed that will be set at the beginning of training.
        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to remove columns not required by the model's forward pass. It's recommended to set this to `False` for RL training, as extra columns (like 'target' or other metadata) might be needed by the reward function or for logging. Ensure your dataset includes at least a 'prompt' column.
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
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If 'auto' is passed, the "
                "dtype will be automatically derived from the model's weights."
            )
        },
    )

    # --- RLOO Specifics ---
    kl_coef: float = field(
        default=0.05, metadata={"help": "Coefficient for the KL divergence penalty."}
    )
    rloo_k: int = field(
        default=2,
        metadata={"help": "REINFORCE Leave-One-Out (RLOO) number of online samples per prompt (must be >= 2)."},
    )
    normalize_reward: bool = field(
        default=False, metadata={"help": "Whether to normalize rewards before advantage calculation."}
    )
    reward_clip_range: float = field(
        default=10.0, metadata={"help": "Clip range for normalized rewards ([-X, X])."}
    )
    normalize_advantage: bool = field(
        default=False, metadata={"help": "Whether to normalize calculated advantages."}
    )
    token_level_kl: bool = field(
        default=False,
        metadata={"help": "Apply KL penalty at token-level (in reward) or sequence-level (in loss)."},
    )
    reward_fn_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the Python file defining the `reward_fn(llm, prompt_text, completions_text, target, **kwargs)`."
        },
    )
    reward_fn_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=dict, metadata={"help": "Keyword arguments to pass to the reward function `reward_fn`."}
    )
    num_ppo_epochs: int = field(
        default=1, metadata={"help": "Number of optimization epochs per batch of generated data (usually 1 for RLOO)."}
    )

    should_self_edit: bool = field(
        default=False,
        metadata={
            "help": "Whether to use self-editing in the reward function. If `True`, the internal reward function will be called instead of the external one."
        },
    )

    # --- PEFT Specifics ---
    lora_rank: int = field(
        default=8, metadata={"help": "LoRA rank (dimension of the update matrices)."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "LoRA alpha scaling factor."}
    )
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Dropout probability for LoRA layers."}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "List of module names or regex patterns to apply LoRA to (e.g., ['q_proj', 'v_proj']). If None, PEFT auto-infers."}
    )
    lora_adapter_name: str = field(
        default="outer_lora",
        metadata={"help": "The name to use for the LoRA adapter being trained and saved/loaded."}
    )

    # --- Generation Specifics (vLLM) ---
    max_completion_length: int = field(
        default=256, metadata={"help": "Maximum number of new tokens to generate per completion."}
    )
    temperature: float = field(
        default=0.7, metadata={"help": "Sampling temperature for generation (higher means more random)."}
    )
    top_p: float = field(
        default=1.0, metadata={"help": "Nucleus sampling threshold (float < 1.0 to enable)."}
    )

    # --- vLLM API Interaction ---
    vllm_api_url: Optional[str] = field(
        default=None, metadata={"help": "Required: URL of the running vLLM OpenAI API server (e.g., http://localhost:8000)."}
    )
    adapter_save_dir: str = field( # Changed Optional[str] to str and removed None default, as it's required
        default="./adapter_checkpoints", metadata={"help": "Required: Directory to save temporary adapters for vLLM requests."}
    )
    vllm_adapter_name: str = field( # Changed Optional[str] to str and removed None default, as it's required
        default="dynamic_training_adapter",
        metadata={"help": "Required: The fixed name used when dynamically loading the adapter via the vLLM API."}
    )

    # --- Other Config ---
    exp_name: str = field(
        default="rloo_indirect", metadata={"help": "Experiment name for logging/wandb."}
    )
    num_sample_generations: int = field(
        default=10, metadata={"help": "Number of sample generation rounds during training (logged). Set 0 to disable."}
    )
    missing_eos_penalty: Optional[float] = field(
        default=None, metadata={"help": "Penalty subtracted from reward if EOS token is missing in generation. None to disable."}
    )
    stop_token: str = field(
        default="eos", metadata={"help": "If 'eos', use tokenizer's EOS token ID as stop sequence. Otherwise ignored."}
    )
    stop_token_id: Optional[int] = field(
        default=None, metadata={"help": "Explicit stop token ID (overrides stop_token='eos')."}
    )
    log_policy_entropy: bool = field(
        default=True, metadata={"help": "Whether to compute and log the policy entropy during training."}
    )


    # --- Inherited/Overridden TrainingArguments ---
    # Sensible defaults for RL, can be overridden by user
    learning_rate: float = field(
        default=1.41e-5, metadata={"help": "The initial learning rate for AdamW optimizer."}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size (number of prompts) per GPU/TPU core/CPU for training."}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Set to False for RL. Keeps extra columns (e.g. 'target') needed by reward function."}
    )
    logging_steps: float = field( # Use float for fraction support
         default=10, metadata={"help": "Log every X updates steps (can be < 1 for logging every fraction of an epoch)."}
    )
    save_steps: float = field( # Use float for fraction support
        default=100, metadata={"help": "Save checkpoint every X updates steps (can be < 1 for saving every fraction of an epoch)."}
    )


    # Internal state variable, not meant to be set by user
    local_dataloader_batch_size: int = field(init=False)

    def __post_init__(self):
        super().__post_init__() # Call parent's post_init

        # --- Validation Checks ---
        if self.rloo_k < 2:
            raise ValueError("rloo_k must be >= 2 for REINFORCE Leave-One-Out.")

        if self.reward_fn_path is None:
             raise ValueError("`reward_fn_path` must be provided and point to a valid Python file.")
        if not os.path.isfile(self.reward_fn_path):
            raise FileNotFoundError(f"Reward function file not found at path: {self.reward_fn_path}")

        if self.vllm_api_url is None:
            raise ValueError("`vllm_api_url` must be provided (URL of the vLLM API server).")
        if not self.adapter_save_dir: # Check if empty string, None should not happen due to typing/default
             raise ValueError("`adapter_save_dir` must be provided.")
        if not self.vllm_adapter_name: # Check if empty string, None should not happen due to typing/default
             raise ValueError("`vllm_adapter_name` must be provided.")


        # Ensure adapter save directory exists
        os.makedirs(self.adapter_save_dir, exist_ok=True)

        # Determine the batch size for the dataloader.
        # In RLOO, the dataloader yields prompts. The trainer then generates k samples per prompt.
        # So, the dataloader batch size should just be the number of prompts per device.
        self.local_dataloader_batch_size = self.per_device_train_batch_size
        print(
            f"RLOO Config: Using `per_device_train_batch_size` ({self.per_device_train_batch_size}) "
            f"as the number of prompts per device for data loading. "
            f"The trainer will generate `rloo_k` ({self.rloo_k}) completions for each prompt."
        )

        if self.remove_unused_columns:
            print(
                "Warning: `remove_unused_columns` is set to `True`. For RL training, it's often necessary to keep "
                "extra columns (like 'target' or metadata) for the reward function. Set to `False` if you encounter issues."
            )

        if self.token_level_kl and self.kl_coef == 0.0:
             print("Warning: `token_level_kl` is True, but `kl_coef` is 0.0. No KL penalty will be applied.")
        if not self.token_level_kl and self.kl_coef > 0.0:
             print(f"Info: Sequence-level KL penalty will be applied with `kl_coef`={self.kl_coef}.")
        elif self.token_level_kl and self.kl_coef > 0.0:
             print(f"Info: Token-level KL penalty will be applied with `kl_coef`={self.kl_coef}.")
