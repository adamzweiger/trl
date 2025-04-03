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

import gc
import math
import os
import textwrap
import time
import importlib.util
import warnings
import requests
from collections import defaultdict
from typing import Callable, Optional, Union, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator, PartialState
from accelerate.utils import broadcast_object_list, broadcast, gather_object, pad_across_processes, reduce, FullyShardedDataParallelPlugin
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
from peft.utils import get_peft_model_state_dict
from peft import PeftModel # Ensure PeftModel is imported
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    # batch_generation, # Replaced by vLLM
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward, # Keep forward for logprob calculation
    # get_reward, # Replaced by external function
    # prepare_deepspeed, # Not handling deepspeed explicitly here, Accelerator does
    print_rich_table,
    selective_log_softmax, # Keep for logprob calculation
    truncate_response,
)
from .rlooindirect_config import RLOOIndirectConfig
from .utils import generate_model_card, get_comet_experiment_url, log_table_to_comet_experiment # Reuse utils

if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
else:
    raise ImportError("PEFT is required for RLOOIndirectTrainer. Please install peft.")

if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0
LORA_ADAPTER_NAME = "outer_lora" # Name of the LoRA adapter to be trained with RL


# Type hint for the external reward function
RewardFnType = Callable[[str, List[str], Any, Dict[str, Any]], List[float]]

class RLOOIndirectTrainer(Trainer):
    """
    Trainer for RLOO (REINFORCE Leave-One-Out) using an indirect reward function and vLLM.

    This trainer implements the RLOO algorithm specifically for training PEFT LoRA adapters.
    Generation is performed using vLLM, and rewards are calculated by an external Python function
    provided via a file path.

    Args:
        model_name_or_path (`str`):
            The identifier of the pre-trained base model (e.g., "meta-llama/Llama-2-7b-hf").
        config (`RLOOIndirectConfig`):
            Configuration object for the trainer.
        processing_class (`PreTrainedTokenizerBase`):
            Tokenizer associated with the base model. Padding side must be 'left'.
        train_dataset (`Dataset`):
            Dataset for training. Must contain "prompt" and "target" columns.
            The "target" column's content will be passed to the external reward function.
        eval_dataset (`Optional[Union[Dataset, dict[str, Dataset]]]`):
            Dataset for evaluation. Must contain "prompt" and "target" columns.
        data_collator (`Optional[DataCollatorWithPadding]`):
            Data collator. If None, a default `DataCollatorWithPadding` is used.
        callbacks (`Optional[list[TrainerCallback]]`):
            Optional list of callbacks.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
             Optional tuple of optimizer and scheduler. Defaults to AdamW and linear warmup.
    """
    _tag_names = ["trl", "rloo-indirect", "peft", "vllm", "fsdp"]

    def __init__(
        self,
        model_name_or_path: str,
        config: RLOOIndirectConfig,
        processing_class: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        accelerator: Accelerator,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ) -> None:
        if not is_peft_available():
             raise ImportError("PEFT is required for RLOOIndirectTrainer but is not installed.")

        self.args = config # Use self.args consistently like Trainer base class
        args = config # Local alias for convenience

        if processing_class.padding_side != "left":
            raise ValueError("Tokenizer padding side must be 'left' for RLOOIndirectTrainer.")

        # --- Model Initialization ---
        model_kwargs = {
            "trust_remote_code": getattr(args, "trust_remote_code", False),
            "torch_dtype": args.torch_dtype,
            # Add other relevant kwargs from TrainingArguments if needed
        }
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        # Apply PEFT LoRA configuration
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )
        # If using k-bit training, prepare model first
        if getattr(args, "load_in_8bit", False) or getattr(args, "load_in_4bit", False):
            base_model = prepare_model_for_kbit_training(
                 base_model, use_gradient_checkpointing=args.gradient_checkpointing
            )

        self.model = get_peft_model(base_model, peft_config, adapter_name=LORA_ADAPTER_NAME)
        # self.model.print_trainable_parameters() # Useful for debugging

        # --- Data Collator ---
        if data_collator is None:
            data_collator = DataCollatorWithPadding(processing_class)

        super().__init__(
            model=self.model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # --- Dataset Handling ---
        self._validate_dataset_columns()

        # If remove_unused_columns was temporarily disabled, restore setting and maybe remove columns
        if hasattr(args, '_saved_remove_unused_columns'):
            self.args.remove_unused_columns = args._saved_remove_unused_columns
            if self.args.remove_unused_columns:
                 # Trainer usually handles this, but we ensure 'prompt' and 'target' stay if needed
                 # For safety, we won't remove columns here if they might be needed by reward_fn
                 pass # Let Trainer handle it, but be aware custom reward functions might break


        # --- Accelerator and Batch Sizes ---
        self.accelerator = accelerator
        try:
            if hasattr(config, 'gradient_accumulation_steps') and self.accelerator.gradient_accumulation_steps != config.gradient_accumulation_steps:
                 warnings.warn(
                     f"Configuration mismatch: RLOOIndirectConfig specified gradient_accumulation_steps={config.gradient_accumulation_steps}, "
                     f"but Accelerator was initialized with gradient_accumulation_steps={self.accelerator.gradient_accumulation_steps}. "
                     f"Using the Accelerator's value.",
                     UserWarning
                 )
        except AttributeError as e:
            warnings.warn(f"Could not check gradient_accumulation_steps during init: {e}", UserWarning)

        print(f"[Process Index {self.args.process_index} (Local Rank {self.args.local_rank}) / World Size {self.args.world_size}] Trainer Init using self.args.")

        # --- Batch Sizes & Steps Calculation ---
        # Note: args.per_device_train_batch_size is #prompts per device per micro-step
        # local_dataloader_batch_size is #prompts per device per dataloader fetch
        # Ensure consistency if gradient_accumulation_steps > 1
        if args.gradient_accumulation_steps > 1 and args.local_dataloader_batch_size != args.per_device_train_batch_size:
             warnings.warn(
                 f"RLOOIndirectConfig.local_dataloader_batch_size ({args.local_dataloader_batch_size}) "
                 f"!= RLOOIndirectConfig.per_device_train_batch_size ({args.per_device_train_batch_size}). "
                 f"Setting local_dataloader_batch_size = per_device_train_batch_size for simplicity "
                 f"when gradient_accumulation_steps > 1.", UserWarning
             )
             args.local_dataloader_batch_size = args.per_device_train_batch_size
        # Total samples per optimizer update = #prompts/dev * #devs * #accum_steps * k
        args.total_samples_per_update = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps * args.rloo_k
        # Total prompts per optimizer update = #prompts/dev * #devs * #accum_steps
        args.total_prompts_per_update = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
        # Effective batch size (prompts per global step before accumulation)
        args.effective_prompt_batch_size = args.per_device_train_batch_size * args.world_size

        # Calculate total training steps
        if args.max_steps > 0:
            num_training_steps = args.max_steps
            num_train_epochs = (args.max_steps * args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size) / len(self.train_dataset)
        else:
            # Steps per epoch based on prompts
            steps_per_epoch = math.ceil( len(self.train_dataset) / (args.local_dataloader_batch_size * args.world_size * args.gradient_accumulation_steps) )
            num_training_steps = math.ceil(args.num_train_epochs * steps_per_epoch)
            num_train_epochs = args.num_train_epochs
        args.num_training_steps = num_training_steps
        args.num_train_epochs = num_train_epochs

        time_tensor = torch.tensor(int(time.time()), device=self.args.device)
        time_int = broadcast(time_tensor, 0).item()
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + self.args.process_index * 100003
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, num_training_steps // args.num_sample_generations)

        # --- Optimizer and Scheduler ---
        if self.optimizer is None:
             self.create_optimizer_and_scheduler(num_training_steps=num_training_steps)

        # --- Trainer Internals (adapted from Trainer/RLOOTrainer) ---
        self.is_fsdp_enabled = None # Could set later after .prepare
        # Disable dropout in policy model
        disable_dropout_in_model(self.model)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id

        # --- Reward Function Loading ---
        self.reward_fn = self._load_reward_function(args.reward_fn_path)
        self.reward_fn_kwargs = args.reward_fn_kwargs or {}

        # --- vLLM API Client ---
        self.vllm_api_url = args.vllm_api_url
        self.adapter_save_path = args.adapter_save_dir # Path where current adapter is saved
        self.vllm_adapter_name = args.vllm_adapter_name # Name used for dynamic loading
        self.api_session = requests.Session() # Use a session

        # --- Final Accelerator Preparation ---
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader(eval_dataset) if eval_dataset else None

        # Prepare model, optimizer, dataloaders with Accelerator
        print(f"[Rank {self.accelerator.process_index}] ENV: MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')} RANK={os.environ.get('RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')} LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        print(f"[Rank {self.accelerator.process_index}] >>> Preparing components with Accelerator...")

        self.model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler
        )
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.is_fsdp_enabled = isinstance(self.accelerator.state.fsdp_plugin, FullyShardedDataParallelPlugin)
        if self.is_fsdp_enabled:
            print(f"[Rank {self.accelerator.process_index}] FSDP is enabled via Accelerate.")
            # You can access fsdp_plugin settings if needed:
            # print(f"[Rank {self.accelerator.process_index}] FSDP Plugin Config: {self.accelerator.state.fsdp_plugin}")
        else:
             print(f"[Rank {self.accelerator.process_index}] FSDP is NOT enabled via Accelerate.")
        if self.is_fsdp_enabled:
             print(f"[Rank {self.accelerator.process_index}] Model class after FSDP prepare: {type(self.model)}")

        # Add PEFT model tags
        if hasattr(self.model, "add_model_tags"):
            model_to_tag = self.model
            if self.is_fsdp_enabled and hasattr(self.model, 'module'):
                MAX_UNWRAP_DEPTH = 5
                for _ in range(MAX_UNWRAP_DEPTH):
                    if isinstance(model_to_tag, PeftModel): break
                    if hasattr(model_to_tag, 'module'): model_to_tag = model_to_tag.module
                    else: break
            try:
                if hasattr(model_to_tag, 'add_model_tags'):
                     model_to_tag.add_model_tags(self._tag_names)
                else:
                     print(f"[Rank {accelerator.process_index}] Warning: Could not find add_model_tags method on model object {type(model_to_tag)}.")
            except Exception as e:
                 print(f"[Rank {accelerator.process_index}] Warning: Failed to add model tags: {e}")

        # Create output directory if needed
        if self.is_world_process_zero():
             os.makedirs(args.output_dir, exist_ok=True)
             os.makedirs(self.adapter_save_path, exist_ok=True)
             print(f"[Rank 0] Ensured output directories exist: {args.output_dir}, {self.adapter_save_path}")

        # Add a barrier here just to ensure directory creation is done before proceeding
        # This is less likely to cause issues than saving.
        self.accelerator.wait_for_everyone()


        print(f"[Process Index {self.args.process_index}] RLOOIndirectTrainer initialization finished.")

    def _validate_dataset_columns(self):
        """Checks if train/eval datasets have 'prompt' and 'target' columns."""
        required_columns = {"prompt", "target"}
        if self.train_dataset:
            train_columns = set(self.train_dataset.column_names)
            if not required_columns.issubset(train_columns):
                raise ValueError(f"Train dataset must contain columns: {required_columns}. Found: {train_columns}")
        if self.eval_dataset:
            eval_columns = set(self.eval_dataset.column_names)
            if not required_columns.issubset(eval_columns):
                 raise ValueError(f"Eval dataset must contain columns: {required_columns}. Found: {eval_columns}")

    def _load_reward_function(self, path: str) -> RewardFnType:
        """Loads the `reward_fn` function from the specified Python file."""
        spec = importlib.util.spec_from_file_location("reward_module", path)
        if spec is None:
            raise ImportError(f"Could not load spec for module at path: {path}")
        reward_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward_module)
        if not hasattr(reward_module, "reward_fn"):
            raise AttributeError(f"File at {path} must define a function named 'reward_fn'.")
        return reward_module.reward_fn

    def _load_adapter_via_api(self, adapter_path: str, adapter_name: str) -> bool:
        """Dynamically loads or updates an adapter via the vLLM API."""
        if not self.is_world_process_zero():
            return True # Assume success on non-main processes

        load_url = f"{self.vllm_api_url}/v1/load_lora_adapter"
        payload = {
            "lora_name": adapter_name,
            "lora_path": adapter_path,
        }
        headers = {"Content-Type": "application/json"}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.api_session.post(load_url, json=payload, headers=headers, timeout=60) # Increased timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                print(f"Successfully loaded adapter '{adapter_name}' from '{adapter_path}' via API.")
                return True
            except requests.exceptions.RequestException as e:
                print(f"Warning: API call to load adapter failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Error: Failed to load adapter '{adapter_name}' via API after {max_retries} attempts.")
                    return False
                time.sleep(2) # Wait before retrying
        return False

    def _generate_via_vllm_api(self, prompts_text: List[str], adapter_name: str, sampling_params: Dict[str, Any]) -> List[str]:
        """Sends prompts to the vLLM API server and returns generated text."""
        if not self.is_world_process_zero():
            # Return dummy list of correct size for other processes
            return [""] * len(prompts_text) * sampling_params.get("n", 1)

        completions_url = f"{self.vllm_api_url}/v1/completions"
        headers = {"Content-Type": "application/json"}

        # Prepare payload
        payload = {
            "model": adapter_name,
            "prompt": prompts_text, # API supports list of prompts
            "n": sampling_params.get("n"),
            "max_tokens": sampling_params.get("max_tokens"),
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "stop": sampling_params.get("stop", []), # Pass stop tokens if any
        }
        # Filter out None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        all_generated_texts = []
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.api_session.post(completions_url, json=payload, headers=headers, timeout=180) # Long timeout for generation
                response.raise_for_status()
                response_data = response.json()

                # --- Parse the response ---
                # The response 'choices' list should contain N * K items.
                # Example for N=2 prompts, K=3 samples:
                # choices = [
                #   { "index": 0, "text": "...", "logprobs": null, "finish_reason": "stop" }, # Prompt 0, Sample 0
                #   { "index": 1, "text": "...", "logprobs": null, "finish_reason": "stop" }, # Prompt 0, Sample 1
                #   { "index": 2, "text": "...", "logprobs": null, "finish_reason": "stop" }, # Prompt 0, Sample 2
                #   { "index": 0, "text": "...", "logprobs": null, "finish_reason": "stop" }, # Prompt 1, Sample 0
                #   { "index": 1, "text": "...", "logprobs": null, "finish_reason": "stop" }, # Prompt 1, Sample 1
                #   { "index": 2, "text": "...", "logprobs": null, "finish_reason": "stop" }  # Prompt 1, Sample 2
                # ]
                # We need to flatten this correctly. The order seems to be grouped by prompt first.
                num_prompts = len(prompts_text)
                num_samples_per_prompt = sampling_params.get("n", 1)
                expected_choices = num_prompts * num_samples_per_prompt

                if "choices" not in response_data or len(response_data["choices"]) != expected_choices:
                     print(f"Error: API response missing 'choices' or has unexpected length. Expected {expected_choices}, got {len(response_data.get('choices', []))}")
                     print(f"Response content: {response.text}") # Log the raw response
                     # Handle error case - maybe return empty list or raise?
                     if attempt == max_retries - 1: return [""] * expected_choices # Return dummy on final failure
                     time.sleep(5)
                     continue # Retry

                # Assuming the order is [p0_s0, p0_s1, ..., p0_sk, p1_s0, p1_s1, ...]
                all_generated_texts = [choice["text"] for choice in response_data["choices"]]
                print(f"Successfully generated {len(all_generated_texts)} completions via API.")
                return all_generated_texts # Success

            except requests.exceptions.RequestException as e:
                print(f"Warning: API call for generation failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Error: Failed to generate completions via API after {max_retries} attempts.")
                    return [""] * len(prompts_text) * sampling_params.get("n", 1) # Return dummy list
                time.sleep(5) # Wait longer before retrying generation

        return [""] * len(prompts_text) * sampling_params.get("n", 1) # Should not be reached if retries work        

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.local_dataloader_batch_size, # Adjusted batch size
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True, # Ensure consistent batch sizes
            pin_memory=self.args.dataloader_pin_memory,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> Optional[DataLoader]:
        """Returns the evaluation dataloader."""
        if eval_dataset is None and self.eval_dataset is None:
            return None
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size, # Standard eval batch size
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
            num_workers=self.args.dataloader_num_workers,
        )

    def find_peft_model(model_to_search):
        MAX_UNWRAP_DEPTH = 5 # Adjust as needed
        peft_model_instance = None
        if isinstance(model_to_search, PeftModel):
            return model_to_search
        current_model = model_to_search
        for _ in range(MAX_UNWRAP_DEPTH):
            if isinstance(current_model, PeftModel):
                peft_model_instance = current_model
                break
            if hasattr(current_model, 'module'): # Common wrapper attribute (FSDP, DDP, etc.)
                current_model = current_model.module
            elif hasattr(current_model, 'base_model') and isinstance(current_model.base_model, PeftModel):
                # Specific check if base_model itself is the Peft layer (less common directly)
                peft_model_instance = current_model.base_model
                break
            else:
                break # Cannot unwrap further
        return peft_model_instance

    def train(self):
        """Main training loop."""
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device

        # Ensure adapter save directory exists
        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(self.adapter_save_path), exist_ok=True)
        # No wait_for_everyone needed here, makedirs is fast and idempotent.

        # Setup dataloader iterator
        dataloader = self.train_dataloader
        def repeat_generator():
            while True:
                for batch in dataloader:
                    yield batch
        iter_dataloader = iter(repeat_generator())

        accelerator.print("=== Training RLOO PEFT Adapter with vLLM ===")
        start_time = time.time()
        self.model.train() # Set policy model to train mode

        # Trainer state initialization
        self.state.global_step = 0
        self.state.epoch = 0
        total_train_samples_per_update = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps * args.rloo_k
        num_update_steps_per_epoch = math.ceil( len(self.train_dataset) / (args.local_dataloader_batch_size * args.world_size * args.gradient_accumulation_steps) )
        max_steps = args.num_training_steps

        accelerator.print(f"  Num prompt examples = {len(self.train_dataset)}")
        accelerator.print(f"  Num Epochs = {args.num_train_epochs:.2f}")
        accelerator.print(f"  Instantaneous batch size per device (# prompts) = {args.per_device_train_batch_size}")
        accelerator.print(f"  Total train batch size (# prompts per global step) = {args.effective_prompt_batch_size}")
        accelerator.print(f"  Total train batch size (# prompts per optimizer update) = {args.total_prompts_per_update}")
        accelerator.print(f"  Total train samples (# completions per optimizer update) = {args.total_samples_per_update}")
        accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        accelerator.print(f"  Total optimization steps = {max_steps}")

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Get models endpoint
        # This is just to check if the API is reachable and working
        models_endpoint = f"{self.vllm_api_url}/v1/models"
        try:
            response = requests.get(models_endpoint, timeout=10) # Add a timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            print(f"Status Code: {response.status_code}")
            print("Response JSON:")
            print(response.json()) # vLLM should return JSON data

        except requests.exceptions.RequestException as e:
            print(f"Error making request to {models_endpoint}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # --- Training Loop ---
        for step in range(max_steps):
            step_start_time = time.time()
            # Placeholders for data accumulated over gradient accumulation steps
            # Store on CPU to save GPU memory, move to GPU during optimization phase
            all_query_ids_list_cpu = []
            all_response_ids_list_cpu = []
            all_ref_logprobs_sum_list_cpu = []
            all_scores_list_cpu = []
            all_sequence_lengths_list_cpu = []

            # --- Experience Generation Phase (Accumulate gradients) ---
            for micro_step in range(args.gradient_accumulation_steps):
                micro_step_start_time = time.time()
                # Get batch for this micro-step (local batch size = per_device_train_batch_size)
                # Keys: 'input_ids', 'attention_mask', 'prompt', 'target'
                raw_batch = next(iter_dataloader)
                print(f"Process {accelerator.process_index} got batch: {raw_batch["prompt"]}")
                local_prompts_ids = raw_batch['input_ids'] # Shape: (local_bs, prompt_len)
                local_prompts_text = raw_batch['prompt']   # List of strings, len = local_bs
                local_targets = raw_batch['target']       # List of targets, len = local_bs
                local_bs = len(local_prompts_text)


                # --- Save Adapter Before vLLM Call ---
                accelerator.print(f"Rank {accelerator.process_index}, Step {step}, MicroStep {micro_step}: Entering adapter save block...")
                try:
                    # 1. Unwrap the model (on all ranks - this is generally lightweight)
                    unwrapped_model = accelerator.unwrap_model(self.model, keep_torch_compile=False) # Adjust compile flag if needed

                    peft_model_instance = None
                    # Check if the unwrapped model is directly the PeftModel
                    if isinstance(unwrapped_model, PeftModel):
                        peft_model_instance = unwrapped_model
                    # Add more robust search if necessary (like your existing logic)
                    elif hasattr(unwrapped_model, 'module') and isinstance(unwrapped_model.module, PeftModel):
                         peft_model_instance = unwrapped_model.module
                    else:
                        model_to_find = unwrapped_model
                        MAX_UNWRAP_DEPTH = 5
                        for _ in range(MAX_UNWRAP_DEPTH):
                            if isinstance(model_to_find, PeftModel):
                                peft_model_instance = model_to_find
                                break
                            if hasattr(model_to_find, 'module'):
                                model_to_find = model_to_find.module
                            else:
                                break

                    if peft_model_instance is None:
                         # Ensure consistent error handling/logging across ranks
                         accelerator.print(f"Rank {accelerator.process_index}: Error: Could not find PeftModel instance. Skipping save.")
                         # Decide how to proceed. Raising an error might be best.
                         # Use accelerator.set_trigger() or similar if you need coordinated failure.
                         raise RuntimeError(f"Rank {accelerator.process_index}: Could not find PeftModel instance during saving.")
                    else:
                        # 2. Ensure the target directory exists (main process responsibility)
                        if accelerator.is_main_process:
                            os.makedirs(self.adapter_save_path, exist_ok=True)

                        # 3. Barrier: Ensure directory is created before *any* rank proceeds
                        # This prevents potential race conditions if non-main ranks somehow
                        # interacted with the path earlier (though unlikely with save_pretrained)
                        accelerator.wait_for_everyone()

                        accelerator.print(f"Rank {accelerator.process_index}: Found PeftModel instance: {type(peft_model_instance)}. Calling save_pretrained for adapter '{LORA_ADAPTER_NAME}'...")

                        # 4. Call save_pretrained on ALL ranks.
                        #    - It triggers the necessary FSDP state_dict gathering internally across all ranks.
                        #    - The 'is_main_process' argument ensures only Rank 0 performs the file write.
                        peft_model_instance.save_pretrained(
                            self.adapter_save_path,
                            selected_adapters=[LORA_ADAPTER_NAME], # Use the instance variable
                            safe_serialization=True,
                            is_main_process=accelerator.is_main_process # CRITICAL argument
                        )

                        # Log completion (main process confirms write initiation)
                        if accelerator.is_main_process:
                            accelerator.print(f"Rank 0: Successfully completed save_pretrained call for adapter '{LORA_ADAPTER_NAME}' to {self.adapter_save_path}")
                        else:
                            accelerator.print(f"Rank {accelerator.process_index}: Participated in save_pretrained (state gathering).")

                except Exception as e:
                    accelerator.print(f"Rank {accelerator.process_index}: Error during adapter saving block: {e}")
                    # Consider more robust error handling/propagation if needed
                    raise e # Re-raise to stop training

                # 5. Barrier: Crucial! Ensure all processes wait here *after* the save call.
                # This guarantees the FSDP gather and the Rank 0 write are finished
                # before any process moves on to the vLLM API call.
                accelerator.print(f"Rank {accelerator.process_index}: Finished adapter save block logic, waiting at final barrier...")
                accelerator.wait_for_everyone()
                accelerator.print(f"Rank {accelerator.process_index}: Passed final adapter save barrier.")
                # --- End of Saving Block ---


                # --- vLLM Generation via API ---
                accelerator.print(f"Rank {accelerator.process_index}: Proceeding to load adapter via API...")
                # 2. Load adapter into vLLM server via API (main process only needs the path)
                outer_lora_path = os.path.join(self.adapter_save_path, LORA_ADAPTER_NAME)
                adapter_loaded = self._load_adapter_via_api(outer_lora_path, self.vllm_adapter_name)

                # Broadcast success/failure status needs device placement
                adapter_loaded_tensor = torch.tensor(1 if adapter_loaded else 0, device=accelerator.device) # Ensure tensor is on correct device

                # Broadcast success/failure status from rank 0
                if accelerator.num_processes > 1:
                     # Use accelerator's broadcast_object_list for potentially simpler handling or stick to tensor broadcast
                     # broadcast(adapter_loaded_tensor, from_process=0)
                     # Alternative using broadcast_object_list (might need adjustment based on object type)
                     status_list = [adapter_loaded] # Wrap boolean in a list for broadcast_object_list
                     broadcast_object_list(status_list, from_process=0)
                     adapter_loaded_on_rank = status_list[0] # Unpack result
                else:
                    adapter_loaded_on_rank = adapter_loaded

                if not adapter_loaded_on_rank:
                    # If loading failed on rank 0 (and broadcasted), raise error on all ranks
                    raise RuntimeError(f"Failed to load adapter '{self.vllm_adapter_name}' into vLLM server (Rank 0 reported failure). Stopping training.")
                else:
                    accelerator.print(f"Rank {accelerator.process_index}: Confirmed adapter load successful.")

                # No need for extra wait_for_everyone here, broadcast synchronizes the status check.
                # 3. Prepare API request parameters
                sampling_params_dict = {
                    "n": args.rloo_k,
                    "max_tokens": args.max_completion_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stop": [self.processing_class.eos_token] if args.stop_token_id == self.processing_class.eos_token_id else [], # Add other stop strings if needed
                }

                # 4. Generate using vLLM API (Gather prompts, main generates, broadcast results)
                # Gather prompts from all processes onto the main process
                all_prompts_text_gathered = gather_object(local_prompts_text)
                # Expected size on main process: world_size * local_bs
                print(f"Process {accelerator.process_index} gathered {len(all_prompts_text_gathered)} prompts: {all_prompts_text_gathered}")

                flat_generated_responses_text_global = []
                global_num_prompts = args.effective_prompt_batch_size # world_size * local_bs
                expected_global_completions = global_num_prompts * args.rloo_k

                if accelerator.is_main_process:
                    # Flatten the gathered list of lists of prompts
                    flat_prompts_global = []
                    for sublist in all_prompts_text_gathered:
                        flat_prompts_global.append(sublist)

                    print(f"Process {accelerator.process_index} flattened gathered prompts: {flat_prompts_global}")

                    if len(flat_prompts_global) != global_num_prompts:
                         raise RuntimeError(f"Gathered prompt list has wrong size on main process. Got {len(flat_prompts_global)}, expected {global_num_prompts}")

                    # Main process calls API with all prompts
                    flat_generated_responses_text_global = self._generate_via_vllm_api(
                        flat_prompts_global,
                        self.vllm_adapter_name,
                        sampling_params_dict
                    )
                    if len(flat_generated_responses_text_global) != expected_global_completions:
                         raise RuntimeError(f"vLLM API returned wrong number of completions. Got {len(flat_generated_responses_text_global)}, expected {expected_global_completions}")
                else:
                    # Non-main processes need a placeholder list of the correct *global* size for broadcast
                    flat_generated_responses_text_global = [""] * expected_global_completions
                print(f"Process {accelerator.process_index} generated {len(flat_generated_responses_text_global)} completions: {flat_generated_responses_text_global}")
                # Broadcast the complete list of generated texts from main process to all
                if accelerator.num_processes > 1:
                     broadcast_object_list(flat_generated_responses_text_global, from_process=0)
                # Now, flat_generated_responses_text_global (the full list) is available on all processes
                print(f"Process {accelerator.process_index} received {len(flat_generated_responses_text_global)} completions after broadcast: {flat_generated_responses_text_global}")

                # --- Process Generated Responses (Locally) ---
                # 5. Slice the global list to get texts corresponding to this process's prompts
                local_start_idx = accelerator.process_index * local_bs * args.rloo_k
                local_end_idx = local_start_idx + local_bs * args.rloo_k
                local_generated_responses_text = flat_generated_responses_text_global[local_start_idx:local_end_idx]
                print(f"Process {accelerator.process_index} sliced {len(local_generated_responses_text)} completions for local batch: {local_start_idx}:{local_end_idx}")

                # 6. Tokenize the local responses
                # Shape: (local_bs * k, resp_len)
                responses_tokenized = self.processing_class(
                    local_generated_responses_text,
                    padding='longest', # Pad to longest response *in this local batch*
                    truncation=True,
                    max_length=args.max_completion_length,
                    return_tensors="pt",
                ).to(device)
                local_responses_ids = responses_tokenized.input_ids

                # Remove potential leading BOS token if tokenizer adds it
                if self.processing_class.bos_token_id is not None and local_responses_ids.shape[1] > 0:
                    if (local_responses_ids[:, 0] == self.processing_class.bos_token_id).all():
                         local_responses_ids = local_responses_ids[:, 1:]

                # Truncate at stop token (if specified and different from EOS handled by padding)
                # Note: vLLM API's `stop` parameter should ideally handle this, but we can double-check.
                local_processed_responses_ids = local_responses_ids
                if args.stop_token_id is not None and args.stop_token_id != self.processing_class.pad_token_id:
                    local_processed_responses_ids = truncate_response(
                        args.stop_token_id, self.processing_class.pad_token_id, local_responses_ids
                    ) # Shape: (local_bs * k, proc_resp_len)

                # --- Log Prob Calculation (Requires Reference Model) ---
                print(f"Process {accelerator.process_index} preparing for logprob calculation...")
                # Repeat original local prompts k times for logprob calculation

                local_queries_repeated = local_prompts_ids.repeat_interleave(args.rloo_k, dim=0) # Shape: (local_bs * k, prompt_len)
                local_context_length = local_queries_repeated.shape[1]

                # Construct full input sequences (query + response) for logprob calculation
                local_query_responses_ids = torch.cat([local_queries_repeated, local_processed_responses_ids], dim=1)
                # Create attention mask for the combined sequence
                local_query_mask = torch.ones_like(local_queries_repeated, device=device) # Assume query part is never padded
                local_resp_attn_mask = (local_processed_responses_ids != self.processing_class.pad_token_id).long()
                local_query_responses_mask = torch.cat([local_query_mask, local_resp_attn_mask], dim=1)

                # Calculate reference logprobs (adapter disabled) using torch.no_grad()
                accelerator.print(f"Rank {accelerator.process_index}: Entering reference logprob calculation...")
                with torch.no_grad():
                    # Find the underlying PeftModel instance reliably
                    model_to_find = self.model
                    peft_model_instance = None
                    MAX_UNWRAP_DEPTH = 5
                    for _ in range(MAX_UNWRAP_DEPTH):
                         if isinstance(model_to_find, PeftModel):
                             peft_model_instance = model_to_find
                             break
                         if hasattr(model_to_find, 'module'):
                             model_to_find = model_to_find.module
                         else:
                             break

                    if peft_model_instance is None:
                         raise RuntimeError(f"Rank {accelerator.process_index}: Could not find underlying PeftModel instance to disable adapter.")

                    accelerator.print(f"Rank {accelerator.process_index}: Found PeftModel {type(peft_model_instance)}, attempting to disable adapter...")
                    
                    import ipdb; ipdb.set_trace()
                    outputs = forward(self.model, local_query_responses_ids, local_query_responses_mask)
                    # Apply disable_adapter context manager to the found PeftModel instance
                    with peft_model_instance.disable_adapter():
                        accelerator.print(f"Rank {accelerator.process_index}: Adapter disabled, running forward pass through FSDP model {type(self.model)}...")
                        # Run forward pass through the original FSDP-wrapped model
                        ref_outputs = forward(self.model, local_query_responses_ids, local_query_responses_mask)
                        accelerator.print(f"Rank {accelerator.process_index}: Forward pass for ref_logprobs completed.")

                        # --- Logit processing ---
                        ref_logits = ref_outputs.logits[:, local_context_length - 1 : -1]
                        ref_logits /= args.temperature + 1e-7
                        local_ref_logprobs = selective_log_softmax(ref_logits, local_processed_responses_ids)

                        del ref_outputs, ref_logits # Clean up memory
                        # Consider gc.collect() if memory pressure is high, but often not needed
                        # gc.collect()
                        # torch.cuda.empty_cache() # Use sparingly, can slow things down

                    accelerator.print(f"Rank {accelerator.process_index}: Adapter re-enabled after context.")

                accelerator.print(f"Rank {accelerator.process_index}: Exiting reference logprob calculation.")
                torch.cuda.empty_cache() # Maybe helpful after the no_grad block


                # --- Reward Calculation (Local) ---
                # Decode processed responses for the reward function
                local_processed_responses_text = self.processing_class.batch_decode(
                    local_processed_responses_ids, skip_special_tokens=True
                )
                local_scores = torch.zeros(local_bs * args.rloo_k, device=device, dtype=torch.float)

                # Call reward function for each original prompt in the local batch
                current_idx = 0
                for i in range(local_bs):
                     prompt_text = local_prompts_text[i]
                     target = local_targets[i]
                     # Get the k completions generated for this specific prompt
                     k_completions = local_processed_responses_text[current_idx : current_idx + args.rloo_k]

                     # Call external reward function
                     try:
                         import inspect
                         sig = inspect.signature(self.reward_fn)
                         reward_kwargs = self.reward_fn_kwargs.copy()
                         # Pass target if the function expects it
                         if 'target' in sig.parameters: reward_kwargs['target'] = target
                         # Warn/remove 'llm' if expected but not provided
                         if 'llm' in sig.parameters:
                             warnings.warn("Reward function expects 'llm' argument, but it's no longer provided when using vLLM API.", UserWarning, stacklevel=2)
                             reward_kwargs.pop('llm', None)

                         k_scores_list = self.reward_fn(
                             prompt_text=prompt_text,
                             completions_text=k_completions,
                             **reward_kwargs
                         )
                     except Exception as e:
                          print(f"Error calling reward function for prompt: {prompt_text[:100]}...")
                          print(f"Completions: {k_completions}")
                          print(f"Kwargs: {reward_kwargs}")
                          raise e


                     if not isinstance(k_scores_list, list) or len(k_scores_list) != args.rloo_k:
                         raise ValueError(f"Reward function must return a list of {args.rloo_k} floats. Got: {k_scores_list}")

                     # Ensure scores are floats before converting to tensor
                     try:
                         k_scores_float = [float(s) for s in k_scores_list]
                     except (ValueError, TypeError) as e:
                          raise ValueError(f"Reward function must return floats. Got: {k_scores_list}. Error: {e}")

                     local_scores[current_idx : current_idx + args.rloo_k] = torch.tensor(k_scores_float, device=device, dtype=torch.float)
                     current_idx += args.rloo_k

                # Post-process scores (e.g., missing EOS penalty)
                # Need to check against pad token ID since EOS might be replaced by pad during truncate_response
                # Or check the original local_responses_ids before potential truncation? Let's use processed.
                # A safer check might be if the sequence length is less than max_completion_length
                sequence_length_for_eos_check = first_true_indices(local_processed_responses_ids == self.processing_class.pad_token_id) -1
                # Check if EOS is present *before* padding begins
                contain_eos_token = torch.any(
                    (local_processed_responses_ids == self.processing_class.eos_token_id) &
                    (torch.arange(local_processed_responses_ids.shape[1], device=device) <= sequence_length_for_eos_check.unsqueeze(1)),
                    dim=1
                )
                # Alternative: Check if generation stopped early (length < max) - less reliable if max_tokens hit
                # stopped_early = sequence_length_for_eos_check < (args.max_completion_length -1)

                if args.missing_eos_penalty is not None:
                    local_scores[~contain_eos_token] -= args.missing_eos_penalty

                # --- Prepare data for storage ---
                # Create padding mask for logprobs based on actual generated length before padding
                local_sequence_lengths = first_true_indices(local_processed_responses_ids == self.processing_class.pad_token_id) - 1
                response_idxs = torch.arange(local_processed_responses_ids.shape[1], device=device).repeat(local_processed_responses_ids.shape[0], 1)
                local_padding_mask = response_idxs > local_sequence_lengths.unsqueeze(1)

                # Mask ref_logprobs where padding occurs
                local_ref_logprobs = torch.masked_fill(local_ref_logprobs, local_padding_mask, INVALID_LOGPROB)

                # Store data needed for optimization phase on CPU
                # We need: query IDs, response IDs, reference logprob sums, scores, sequence lengths
                all_query_ids_list_cpu.append(local_queries_repeated.cpu())
                all_response_ids_list_cpu.append(local_processed_responses_ids.cpu())
                all_ref_logprobs_sum_list_cpu.append(local_ref_logprobs.sum(1).cpu()) # Sum over sequence length
                all_scores_list_cpu.append(local_scores.cpu())
                all_sequence_lengths_list_cpu.append(local_sequence_lengths.cpu())

                # --- Micro-step cleanup ---
                del (raw_batch, local_prompts_ids, local_prompts_text, local_targets,
                     local_generated_responses_text, responses_tokenized, local_responses_ids,
                     local_processed_responses_ids, local_queries_repeated, local_query_responses_ids,
                     local_query_mask, local_resp_attn_mask, local_query_responses_mask,
                     local_ref_logprobs, local_processed_responses_text, local_scores,
                     local_sequence_lengths, local_padding_mask, contain_eos_token)
                if 'k_scores_list' in locals(): del k_scores_list
                if 'k_scores_float' in locals(): del k_scores_float
                # Empty cache periodically if accumulating many steps
                if micro_step % 4 == 0: # Adjust frequency as needed
                     torch.cuda.empty_cache()
                     gc.collect()
                accelerator.print(f"    Micro-step {micro_step+1}/{args.gradient_accumulation_steps} completed in {time.time() - micro_step_start_time:.2f}s")


            # --- End of Gradient Accumulation Loop ---
            accumulation_end_time = time.time()
            accelerator.print(f"  Gradient accumulation phase completed in {accumulation_end_time - step_start_time:.2f}s")

            # --- Optimization Phase ---
            if not all_scores_list_cpu:
                 accelerator.print("Warning: No data accumulated, skipping optimization step.")
                 continue # Skip if no data (e.g., first step failed)

            # Collate accumulated data from CPU lists to device tensors
            # This batch contains data from all accumulation steps for the current device
            batch_query_ids = torch.cat(all_query_ids_list_cpu, dim=0).to(device)
            batch_response_ids = torch.cat(all_response_ids_list_cpu, dim=0).to(device)
            batch_ref_logprobs_sum = torch.cat(all_ref_logprobs_sum_list_cpu, dim=0).to(device)
            batch_scores = torch.cat(all_scores_list_cpu, dim=0).to(device)
            batch_seq_lengths = torch.cat(all_sequence_lengths_list_cpu, dim=0).to(device)
            local_total_samples_in_batch = len(batch_scores) # Should be local_bs * k * accum_steps

            # Clear CPU lists
            del (all_query_ids_list_cpu, all_response_ids_list_cpu, all_ref_logprobs_sum_list_cpu,
                 all_scores_list_cpu, all_sequence_lengths_list_cpu)
            gc.collect()

            # RLOO updates typically happen once per batch of experience (no inner PPO epochs)
            # Prepare data for the single optimization pass
            mb_query_ids = batch_query_ids
            mb_response_ids = batch_response_ids
            mb_ref_logprobs_sum = batch_ref_logprobs_sum
            mb_scores = batch_scores
            mb_seq_lengths = batch_seq_lengths

            # Recompute policy logprobs with gradients enabled within the accelerator context
            with accelerator.accumulate(self.model):
                optim_start_time = time.time()
                # Construct inputs for the policy forward pass
                mb_query_responses_ids = torch.cat([mb_query_ids, mb_response_ids], dim=1)
                mb_query_mask = torch.ones_like(mb_query_ids, device=device)
                # Recompute response attention mask based on sequence lengths
                mb_response_idxs = torch.arange(mb_response_ids.shape[1], device=device).repeat(mb_response_ids.shape[0], 1)
                mb_resp_attn_mask = (mb_response_idxs <= mb_seq_lengths.unsqueeze(1)).long() # <= because seq_length is 0-indexed length
                mb_query_responses_mask = torch.cat([mb_query_mask, mb_resp_attn_mask], dim=1)
                mb_context_length = mb_query_ids.shape[1]

                # Forward pass for policy logprobs
                output = forward(self.model, mb_query_responses_ids, mb_query_responses_mask)
                logits = output.logits[:, mb_context_length - 1 : -1] # Logits for response tokens
                logits /= args.temperature + 1e-7 # Apply temperature

                # Compute new logprobs (token level)
                new_logprobs_token = selective_log_softmax(logits, mb_response_ids)

                # Apply padding mask (recompute based on seq lengths)
                mb_padding_mask = mb_response_idxs > mb_seq_lengths.unsqueeze(1)
                new_logprobs_token = torch.masked_fill(new_logprobs_token, mb_padding_mask, INVALID_LOGPROB)
                # Sum logprobs for the sequence (ignoring padded parts)
                # Use masked_fill with 0 before summing, or sum where mask is False
                new_logprobs_sum = torch.where(~mb_padding_mask, new_logprobs_token, torch.tensor(0.0, device=device)).sum(1)

                # --- Compute Advantages ---
                # KL divergence per sequence: KL = policy_logprob_sum - ref_logprob_sum
                kl_sum = new_logprobs_sum - mb_ref_logprobs_sum # Shape: (local_total_samples_in_batch,)

                # Normalize raw scores if needed (across the accumulated batch on this device)
                # Note: Normalizing here might differ slightly from normalizing across global batch if done earlier
                if args.normalize_reward:
                    # Gather scores across all processes for this accumulation batch
                    gathered_scores = accelerator.gather(mb_scores)
                    if gathered_scores.numel() > 1:
                         mean_score = gathered_scores.mean()
                         std_score = gathered_scores.std()
                         # Apply normalization to local scores
                         mb_scores = (mb_scores - mean_score) / (std_score + 1e-8)
                         mb_scores = torch.clamp(mb_scores, -args.reward_clip_range, args.reward_clip_range)
                    else:
                         warnings.warn("Could not normalize rewards: Not enough valid values gathered.")

                # Combine score and KL penalty to get RLHF reward
                # Using sequence-level KL penalty added to reward
                non_score_reward_per_seq = -args.kl_coef * kl_sum
                rlhf_reward = non_score_reward_per_seq + mb_scores # Shape: (local_total_samples_in_batch,)

                # Calculate RLOO baseline and advantages
                # Reshape rewards based on how data was accumulated (num_prompts * k)
                num_prompts_in_local_batch = local_total_samples_in_batch // args.rloo_k
                if local_total_samples_in_batch % args.rloo_k != 0:
                     raise ValueError("Total samples in batch is not divisible by rloo_k.")

                # Reshape: (local_total_samples,) -> (k, num_prompts_local)
                # Need to ensure the order matches: [p0_s0, p0_s1.. p0_sk, p1_s0, p1_s1..]
                try:
                    # This assumes the concatenated lists maintained the [p0_s0, p0_s1,...] order within each micro-batch
                    rlhf_reward_grouped = rlhf_reward.reshape(num_prompts_in_local_batch, args.rloo_k).transpose(0, 1)
                    # Shape check: (k, num_prompts_local)
                except Exception as e:
                    raise RuntimeError(f"Failed to reshape rlhf_reward. Shape: {rlhf_reward.shape}, k: {args.rloo_k}, num_prompts: {num_prompts_in_local_batch}. Error: {e}")


                # Calculate baseline: mean reward of other k-1 samples for the same prompt
                # Sum across k dim, subtract self, divide by k-1
                baseline = (rlhf_reward_grouped.sum(0, keepdim=True) - rlhf_reward_grouped) / (args.rloo_k - 1)
                # Calculate advantages
                advantages_grouped = rlhf_reward_grouped - baseline # Shape: (k, num_prompts_local)
                # Flatten advantages back to match the order of samples
                advantages = advantages_grouped.transpose(0, 1).flatten() # Shape: (local_total_samples_in_batch,)

                # Normalize advantages if needed (gather across all processes for mean/std)
                if args.normalize_advantage:
                     gathered_advantages = accelerator.gather(advantages)
                     # Filter NaNs/Infs just in case
                     valid_advantages = gathered_advantages[~torch.isnan(gathered_advantages) & ~torch.isinf(gathered_advantages)]
                     if valid_advantages.numel() > 1:
                         mean_adv = valid_advantages.mean()
                         std_adv = valid_advantages.std()
                         advantages = (advantages - mean_adv) / (std_adv + 1e-8)
                     elif valid_advantages.numel() > 0:
                         # If std is zero or only one valid value, just center
                         mean_adv = valid_advantages.mean()
                         advantages = advantages - mean_adv
                         if valid_advantages.numel() == 1:
                            warnings.warn("Only one valid advantage value found after gathering. Centering advantages but not scaling.")
                         else: # std is zero
                            warnings.warn("Standard deviation of advantages is zero after gathering. Centering advantages but not scaling.")
                     else:
                         warnings.warn("Could not normalize advantages: No valid values found after gathering.")

                # --- RLOO Loss Calculation ---
                # Loss is - E[Adv * log P(response | prompt)]
                # We use the sum of logprobs for the sequence.
                pg_loss = -advantages * new_logprobs_sum # Use the recomputed policy logprobs sum
                pg_loss = pg_loss.mean() # Average over the local batch

                loss = pg_loss # Total loss (KL was included in reward)

                # Backward pass (managed by accelerator.accumulate)
                accelerator.backward(loss)
                optim_end_time = time.time()
                accelerator.print(f"    Optimization forward/backward pass completed in {optim_end_time - optim_start_time:.2f}s")


                # --- Log Stats (inside accumulation context, before optimizer step) ---
                # Store stats from this optimization step (which covers the accumulated batch)
                # These will be gathered and reduced *after* the optimizer step if gradients were synced
                with torch.no_grad():
                     policy_loss_item = pg_loss.item()
                     # Entropy calculation
                     prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                     entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                     # Mask entropy for padding and average over non-padded tokens
                     masked_entropy = entropy.masked_fill(mb_padding_mask, 0.0)
                     # Sum entropy over sequence and average over batch, or average over all valid tokens?
                     # Let's average over valid tokens:
                     mean_entropy_item = (masked_entropy.sum() / (~mb_padding_mask).sum()).item()

                     # Store metrics calculated *before* potential advantage normalization for logging clarity
                     kl_sum_detached = kl_sum.detach().cpu()
                     rlhf_reward_detached = rlhf_reward.detach().cpu()
                     non_score_reward_detached = non_score_reward_per_seq.detach().cpu()
                     # Store advantages *after* potential normalization
                     advantages_detached = advantages.detach().cpu()
                     scores_detached = mb_scores.detach().cpu() # Scores potentially normalized if normalize_reward=True

            # --- End Accumulate Context ---

            # --- Optimizer Step, LR Scheduling, Logging ---
            if accelerator.sync_gradients:
                optimizer_step_start_time = time.time()
                # Clip gradients if needed
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad() # Zero gradients *after* optimizer step

                self.lr_scheduler.step()
                self.state.global_step += 1 # Increment global step only when optimizer steps

                accelerator.print(f"    Optimizer step completed in {time.time() - optimizer_step_start_time:.2f}s")

                # --- Gather and Log Metrics ---
                # Gather stats computed *during* the optimization phase across all GPUs for logging
                # Perform reduction only on the main process after gathering
                if accelerator.is_main_process:
                     # Gather stats from all processes. Use gather_object for lists/tensors stored on CPU.
                     # Note: gather() works on tensors currently on GPU.
                     gathered_kl_sum = accelerator.gather(kl_sum_detached.to(device)).float()
                     gathered_rlhf_reward = accelerator.gather(rlhf_reward_detached.to(device)).float()
                     gathered_non_score_reward = accelerator.gather(non_score_reward_detached.to(device)).float()
                     gathered_advantages = accelerator.gather(advantages_detached.to(device)).float()
                     gathered_scores = accelerator.gather(scores_detached.to(device)).float()
                     gathered_seq_lengths = accelerator.gather(batch_seq_lengths.to(device)) # Already on device from batch_*

                     # Reduce metrics that were calculated per-device during optimization
                     # Convert single items to tensors for reduction
                     policy_loss_tensor = torch.tensor(policy_loss_item, device=device)
                     mean_entropy_tensor = torch.tensor(mean_entropy_item, device=device)

                     mean_pg_loss_red = reduce(policy_loss_tensor, reduction='mean').item()
                     mean_entropy_red = reduce(mean_entropy_tensor, reduction='mean').item()

                     # Calculate means/stds from gathered tensors
                     mean_score_red = gathered_scores.mean().item()
                     mean_adv_red = gathered_advantages.mean().item()
                     std_adv_red = gathered_advantages.std().item()
                     mean_rlhf_reward_red = gathered_rlhf_reward.mean().item()
                     mean_non_score_reward_red = gathered_non_score_reward.mean().item()
                     mean_kl_red = gathered_kl_sum.mean().item() # Mean KL per sequence
                     mean_seq_len_red = gathered_seq_lengths.float().mean().item()


                     metrics = {}
                     # Log based on optimizer steps (global_step)
                     metrics["train/episode"] = self.state.global_step * args.total_prompts_per_update # Log based on prompts processed per update
                     metrics["train/reward_score"] = mean_score_red
                     metrics["train/reward_rlhf"] = mean_rlhf_reward_red
                     metrics["train/reward_non_score"] = mean_non_score_reward_red # Should be ~ -kl_coef * kl_ref_policy
                     metrics["train/advantage_mean"] = mean_adv_red
                     metrics["train/advantage_std"] = std_adv_red
                     metrics["train/kl_ref_policy"] = mean_kl_red # KL(policy || ref) per sequence
                     metrics["train/policy_entropy"] = mean_entropy_red
                     metrics["train/loss_policy"] = mean_pg_loss_red
                     metrics["train/seq_length"] = mean_seq_len_red
                     metrics["train/lr"] = self.lr_scheduler.get_last_lr()[0]
                     # Calculate epoch based on global steps and steps per epoch
                     current_epoch = self.state.global_step / num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0
                     metrics["train/epoch"] = current_epoch

                     self.log(metrics) # Log the aggregated metrics

                # Trigger callbacks, checkpointing, evaluation checks based on global_step
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                if self.control.should_save:
                     self._save_checkpoint() # Saves adapter to checkpoint dir based on global_step
                     self.control = self.callback_handler.on_save(self.args, self.state, self.control)

                if self.control.should_evaluate:
                     # Evaluation needs adapting for API generation too
                     accelerator.print("Triggering evaluation...")
                     # Ensure model is in eval mode for evaluation
                     self.model.eval()
                     metrics = self.evaluate() # Call evaluate method
                     # Ensure model is back in train mode after evaluation
                     self.model.train()
                     self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)


                # Sample generation check (moved inside sync_gradients block)
                if accelerator.is_main_process and args.num_sample_generations > 0 and self.state.global_step > 0 and \
                   (self.state.global_step % self.sample_generations_freq == 0 or self.control.should_training_stop):
                     accelerator.print("Generating samples for logging...")
                     # Ensure model is in eval mode for generation
                     self.model.eval()
                     self.generate_completions(sampling=True) # Use API for generation
                     # Ensure model is back in train mode
                     self.model.train()


            # --- End Sync Gradients Block ---

            if self.control.should_training_stop:
                 accelerator.print("Training stopping signal received.")
                 break

            # Update epoch state (handled by logging logic)
            step_end_time = time.time()
            accelerator.print(f"  Full step {step}/{max_steps} completed in {step_end_time - step_start_time:.2f}s (Global Step: {self.state.global_step})")
            # Optional: Add a small sleep if GPU util is 100% and hitting issues
            # time.sleep(0.1)

        # --- End of Training Loop ---
        accelerator.print("=== Finished Training ===")
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Final save
        if self.control.should_save:
            self._save_checkpoint()
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        accelerator.wait_for_everyone() # Ensure all processes finish cleanly

    def _save_checkpoint(self, trial=None, metrics=None):
         """Saves the PEFT adapter during training."""
         # Use a barrier BEFORE saving to ensure all ranks are ready
         self.accelerator.wait_for_everyone()

         if not self.is_world_process_zero():
             return

         save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
         print(f"Saving adapter checkpoint to {save_path}")

         try:
             # Important: Use accelerator.unwrap_model() before saving with FSDP
             unwrapped_model = self.accelerator.unwrap_model(self.model)

             # Ensure we're saving the PeftModel if it was wrapped
             model_to_save = unwrapped_model
             MAX_UNWRAP_DEPTH = 5 # Safety limit
             for _ in range(MAX_UNWRAP_DEPTH):
                 if isinstance(model_to_save, PeftModel):
                     break # Found the PeftModel
                 # Check if FSDP is still wrapping the unwrapped model (can happen)
                 if 'FullyShardedDataParallel' in str(type(model_to_save)) and hasattr(model_to_save, 'module'):
                     model_to_save = model_to_save.module
                 else:
                     break # No more .module attribute or not FSDP

             if isinstance(model_to_save, PeftModel):
                 model_to_save.save_pretrained(save_path)
                 self.processing_class.save_pretrained(save_path) # Save tokenizer too
                 print(f"Successfully saved checkpoint: {save_path}")
             else:
                 print(f"Error in _save_checkpoint: Could not find underlying PeftModel to save. Found type: {type(model_to_save)}")

             # Optionally save trainer state
             self.state.save_to_json(os.path.join(save_path, "trainer_state.json"))

         except Exception as e:
             print(f"Error during _save_checkpoint: {e}")


    @torch.no_grad()
    def generate_completions(self, sampling: bool = False, dataloader: Optional[DataLoader] = None):
         """Generates completions for evaluation or logging using vLLM API."""
         if not self.is_world_process_zero():
             return

         args = self.args
         if not self.vllm_api_url:
             print("Warning: vLLM API URL not configured, skipping completion generation.")
             return

         eval_dataloader = dataloader if dataloader else self.get_eval_dataloader()
         if eval_dataloader is None:
             print("Warning: No evaluation dataloader found, skipping completion generation.")
             return

         print(f"\n=== Generating Completions at Step {self.state.global_step} ===")

         # Determine which adapter to load
         # Use the latest checkpoint if available, otherwise the 'current' one used during training
         checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
         if os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")):
             adapter_path_to_load = checkpoint_dir
             adapter_name_to_use = f"eval_adapter_{self.state.global_step}" # Use unique name for eval
             print(f"Using checkpoint adapter: {adapter_path_to_load}")
         else:
             adapter_path_to_load = self.adapter_save_path # Path to 'current_adapter'
             adapter_name_to_use = self.vllm_adapter_name # Reuse training name if no checkpoint
             print(f"Using current training adapter: {adapter_path_to_load}")

         # Load the chosen adapter via API
         outer_lora_path = os.path.join(adapter_path_to_load, LORA_ADAPTER_NAME)
         if not self._load_adapter_via_api(outer_lora_path, adapter_name_to_use):
             print(f"Error: Failed to load adapter '{adapter_name_to_use}' for evaluation. Skipping generation.")
             return

         # Define sampling parameters for evaluation (n=1)
         eval_sampling_params = {
             "n": 1,
             "max_tokens": args.max_completion_length,
             "temperature": 0.1 if not sampling else args.temperature,
             "top_p": 1.0 if not sampling else args.top_p,
             "stop": [self.processing_class.eos_token] if args.stop_token_id == self.processing_class.eos_token_id else [],
         }

         table = defaultdict(list)
         # Limit number of samples for logging if needed
         max_eval_samples = args.eval_samples if args.eval_samples > 0 else float('inf')
         samples_generated = 0

         for batch in eval_dataloader:
             if samples_generated >= max_eval_samples:
                 break

             prompts_ids = batch["input_ids"]
             targets = batch["target"]
             prompts_text = self.processing_class.batch_decode(prompts_ids, skip_special_tokens=True)

             # Limit batch size if it exceeds remaining samples needed
             num_needed = max_eval_samples - samples_generated
             if len(prompts_text) > num_needed:
                 prompts_text = prompts_text[:num_needed]
                 targets = targets[:num_needed]

             if not prompts_text: continue

             # Generate via API
             completions_text = self._generate_via_vllm_api(
                 prompts_text,
                 adapter_name_to_use,
                 eval_sampling_params
             )

             # Calculate rewards
             scores = []
             for i in range(len(prompts_text)):
                 import inspect
                 sig = inspect.signature(self.reward_fn)
                 reward_kwargs = self.reward_fn_kwargs.copy()
                 if 'target' in sig.parameters: reward_kwargs['target'] = targets[i]
                 if 'llm' in sig.parameters: reward_kwargs.pop('llm', None) # Remove llm if present

                 # Pass list with single completion
                 score_list = self.reward_fn(
                     prompt_text=prompts_text[i],
                     completions_text=[completions_text[i]],
                     **reward_kwargs
                 )
                 scores.append(score_list[0] if isinstance(score_list, list) else score_list)

             table["prompt"].extend(prompts_text)
             table["target"].extend([str(t) for t in targets])
             table["model_response"].extend(completions_text)
             table["score"].extend(scores)

             samples_generated += len(prompts_text)

             if sampling: # Only generate one batch if sampling for logs during training
                 break

         df = pd.DataFrame(table)

         if self.is_world_process_zero(): # Log only on main process
             print_rich_table(df.head(20)) # Show more examples
             log_file_path = os.path.join(args.output_dir, f"completions_step_{self.state.global_step}.csv")
             try:
                 df.to_csv(log_file_path, index=False)
                 print(f"Saved completions log to {log_file_path}")
             except Exception as e:
                 print(f"Error saving completions log: {e}")


             if "wandb" in args.report_to and is_wandb_available() and wandb.run is not None:
                 try:
                     # Log a limited number of rows to avoid large tables
                     log_df = df.head(min(len(df), 50)) # Log max 50 rows
                     wandb.log({f"eval/completions_step_{self.state.global_step}": wandb.Table(dataframe=log_df)})
                 except Exception as e:
                     print(f"Warning: Failed to log table to wandb: {e}")

             if "comet_ml" in args.report_to:
                 try:
                     log_df = df.head(min(len(df), 50))
                     log_table_to_comet_experiment(
                         name=f"completions_step_{self.state.global_step}.csv",
                         table=log_df,
                     )
                 except Exception as e:
                      print(f"Warning: Failed to log table to comet_ml: {e}")

    def create_model_card(
         self,
         model_name: Optional[str] = None,
         dataset_name: Optional[str] = None,
         tags: Union[str, list[str], None] = None,
     ):
         """Creates a draft model card for the trained PEFT adapter."""
         if not self.is_world_process_zero():
             return

         # Use base model name from the PEFT model's config
         base_model_name = self.model.base_model.config._name_or_path

         tags = tags or []
         if isinstance(tags, str):
             tags = [tags]
         tags.extend(self._tag_names) # Add trainer tags

         # RLOO Citation (same as original)
         citation = textwrap.dedent("""\
         @inproceedings{ahmadian2024back,
             title        = {{Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs}},
             author       = {Arash Ahmadian and Chris Cremer and Matthias Gall{\'{e}} and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet {\"{U}}st{\"{u}}n and Sara Hooker},
             year         = 2024,
             booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
             publisher    = {Association for Computational Linguistics},
             pages        = {12248--12267},
             editor       = {Lun{-}Wei Ku and Andre Martins and Vivek Srikumar},
         }""")

         model_card = generate_model_card(
             base_model=base_model_name,
             model_name=model_name or f"{base_model_name}-{LORA_ADAPTER_NAME}-RLOO-Indirect",
             hub_model_id=self.hub_model_id, # Trainer doesn't set this by default, needs push_to_hub setup
             dataset_name=dataset_name,
             tags=tags,
             wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
             comet_url=get_comet_experiment_url(),
             trainer_name="RLOOIndirect",
             trainer_citation=citation,
             paper_title="Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs",
             paper_id="2402.14740",
             # Add PEFT config details
             peft_config=self.model.peft_config[LORA_ADAPTER_NAME].to_dict()
         )

         # Save the model card
         output_path = os.path.join(self.args.output_dir, "README.md")
         model_card.save(output_path)
         print(f"Model card saved to {output_path}")
