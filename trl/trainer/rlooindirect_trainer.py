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
from accelerate.utils import broadcast, gather_object, pad_across_processes, reduce
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

from ..models.utils import unwrap_model_for_generation # Keep potentially useful utils
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
LORA_ADAPTER_NAME = "outer_lora" # Fixed name for the adapter we train


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
    _tag_names = ["trl", "rloo-indirect", "peft", "vllm"]

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
        # self.processing_class = processing_class

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
        # self.data_collator = data_collator

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
        # self.train_dataset = train_dataset
        # self.eval_dataset = eval_dataset
        self._validate_dataset_columns()

        # If remove_unused_columns was temporarily disabled, restore setting and maybe remove columns
        if hasattr(args, '_saved_remove_unused_columns'):
            self.args.remove_unused_columns = args._saved_remove_unused_columns
            if self.args.remove_unused_columns:
                 # Trainer usually handles this, but we ensure 'prompt' and 'target' stay if needed
                 # For safety, we won't remove columns here if they might be needed by reward_fn
                 pass # Let Trainer handle it, but be aware custom reward functions might break


        # --- Accelerator and Batch Sizes ---
        # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        # self.accelerator = accelerator
        self.accelerator = accelerator
        try:
            # This reads accelerator.state.gradient_accumulation_steps, might still fail
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
        # Note: args.local_dataloader_batch_size is now #prompts per device
        # Total samples per update involves rloo_k
        args.total_samples_per_update = args.local_dataloader_batch_size * args.world_size * args.gradient_accumulation_steps * args.rloo_k
        args.local_batch_size_per_step = args.local_dataloader_batch_size * args.gradient_accumulation_steps # Num prompts per device per optimizer step
        args.effective_batch_size = args.local_dataloader_batch_size * args.world_size # Num prompts per global step
        args.total_batch_size_per_update = args.local_batch_size_per_step * args.world_size # Total prompts per optimizer update

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
        # self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
             self.create_optimizer_and_scheduler(num_training_steps=num_training_steps)

        # --- Trainer Internals (adapted from Trainer/RLOOTrainer) ---
        self.is_fsdp_enabled = None
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
        print("hello0")


        # Prepare model, optimizer, dataloaders with Accelerator
        print(f"[Rank {self.accelerator.process_index}] ENV: MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')} RANK={os.environ.get('RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')} LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        print(f"[Rank {self.accelerator.process_index}] >>> Preparing components with Accelerator...")

        self.model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler
        )
        print("hello10")
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Add PEFT model tags
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(args.output_dir, exist_ok=True)
            # Ensure the specific adapter save path exists
            os.makedirs(self.adapter_save_path, exist_ok=True)
            # Save initial PEFT config
            try:
                # Need unwrapped model if using FSDP/DDP
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(args.output_dir)
            except Exception as e:
                 print(f"Warning: Could not save initial PEFT config: {e}")
        
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

    def train(self):
        """Main training loop."""
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device

        # Ensure adapter save directory exists
        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(self.adapter_save_path), exist_ok=True)
        accelerator.wait_for_everyone()

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

        print(f"  Num prompt examples = {len(self.train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs:.2f}")
        print(f"  Instantaneous batch size per device (# prompts) = {args.per_device_train_batch_size}")
        print(f"  Total train batch size (# prompts per update) = {args.total_batch_size_per_update}")
        print(f"  Total train batch size (# samples per update) = {total_train_samples_per_update}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_steps}")

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # --- Training Loop ---
        for step in range(max_steps):
            all_query_ids_list = []
            all_response_ids_list = []
            all_logprobs_list = []
            all_ref_logprobs_list = []
            all_scores_list = []
            all_advantages_list = []
            all_sequence_lengths_list = []
            all_kl_list = []
            all_rlhf_rewards_list = []
            all_non_score_rewards_list = []
            approxkl_stats_accum = []
            pg_loss_stats_accum = []
            entropy_stats_accum = []
            ratio_stats_accum = []

            # --- Experience Generation Phase ---
            # Accumulate gradients for args.gradient_accumulation_steps
            for micro_step in range(args.gradient_accumulation_steps):
                # Get batch: keys 'input_ids', 'attention_mask', 'prompt', 'target'
                # Batch size is local_dataloader_batch_size (# prompts)
                raw_batch = next(iter_dataloader)
                prompts_data = raw_batch['input_ids'].to(device) # Shape: (local_bs, prompt_len)
                targets = raw_batch['target'] # List of targets, len = local_bs

                # --- vLLM Generation via API ---
                # 1. Save current adapter state (main process)
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(self.model)
                    unwrapped_model.save_pretrained(self.adapter_save_path)
                accelerator.wait_for_everyone() # Ensure save completes

                # 2. Load adapter into vLLM server via API (main process)
                adapter_loaded = self._load_adapter_via_api(self.adapter_save_path, self.vllm_adapter_name)
                # Broadcast success/failure? For now, assume it works or fails globally after wait.
                adapter_loaded_tensor = torch.tensor(1 if adapter_loaded else 0, device=device)
                adapter_loaded_tensor = broadcast(adapter_loaded_tensor)
                if adapter_loaded_tensor.item() == 0:
                    raise RuntimeError(f"Failed to load adapter '{self.vllm_adapter_name}' into vLLM server. Stopping training.")
                accelerator.wait_for_everyone() # Ensure load call finishes everywhere

                # 3. Prepare API request parameters
                sampling_params_dict = {
                    "n": args.rloo_k,
                    "max_tokens": args.max_completion_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stop": [self.processing_class.eos_token] if args.stop_token_id == self.processing_class.eos_token_id else [], # Add other stop strings if needed
                }
                # Decode prompts for API
                prompts_text = self.processing_class.batch_decode(prompts_data, skip_special_tokens=True)

                # 4. Generate using vLLM API (main process calls, result broadcasted)
                # Result is flat list: [p0_s0, p0_s1.., p1_s0, p1_s1.., ...] size = local_bs * k
                generated_responses_text = self._generate_via_vllm_api(
                    prompts_text,
                    self.vllm_adapter_name,
                    sampling_params_dict
                )
                # Broadcast results (necessary if only main process called API)
                generated_responses_text = gather_object(generated_responses_text) # Gather from all processes (even if only main had data)
                # Filter out potential empty strings from non-main processes if broadcast was used instead of gather_object
                # If using gather_object, main process has the full list
                if accelerator.is_main_process:
                    # Flatten the list of lists gathered
                    flat_generated_responses_text = []
                    for sublist in generated_responses_text:
                        flat_generated_responses_text.extend(sublist)
                    generated_responses_text = flat_generated_responses_text
                else:
                    generated_responses_text = [""] * args.effective_batch_size * args.rloo_k # Placeholder size

                # Broadcast the final list from main process
                generated_responses_text = broadcast(generated_responses_text)

                # --- Process Generated Responses ---
                # 5. Tokenize responses
                # Shape: (global_bs * k, resp_len) - Need to handle on each process
                # Each process receives the full list of generated texts
                local_start_index = accelerator.process_index * args.local_dataloader_batch_size * args.rloo_k
                local_end_index = local_start_index + args.local_dataloader_batch_size * args.rloo_k
                local_generated_responses_text = generated_responses_text[local_start_index:local_end_index]

                responses_tokenized = self.processing_class(
                    local_generated_responses_text,
                    padding='longest',
                    truncation=True,
                    max_length=args.max_completion_length,
                    return_tensors="pt",
                ).to(device)
                responses_ids = responses_tokenized.input_ids # Shape: (local_bs * k, resp_len)

                # Remove potential leading BOS token
                if self.processing_class.bos_token_id is not None and responses_ids.shape[1] > 0:
                    if (responses_ids[:, 0] == self.processing_class.bos_token_id).all():
                         responses_ids = responses_ids[:, 1:]

                # Truncate at stop token
                processed_responses_ids = responses_ids
                if args.stop_token_id is not None:
                    processed_responses_ids = truncate_response(
                        args.stop_token_id, self.processing_class.pad_token_id, responses_ids
                    ) # Shape: (local_bs * k, proc_resp_len)

                # --- Log Prob Calculation ---
                # Repeat original prompts k times for logprob calculation
                queries_repeated = prompts_data.repeat_interleave(args.rloo_k, dim=0) # Shape: (local_bs * k, prompt_len)
                context_length = queries_repeated.shape[1]

                # Construct full input sequences (query + response)
                query_responses_ids = torch.cat([queries_repeated, processed_responses_ids], dim=1)
                # Create attention mask
                query_mask = torch.ones_like(queries_repeated)
                resp_padding_mask = (processed_responses_ids == self.processing_class.pad_token_id)
                resp_attn_mask = ~resp_padding_mask
                query_responses_mask = torch.cat([query_mask, resp_attn_mask], dim=1)

                # Need logprobs from *current* model state (before optimizer step)
                # Use torch.no_grad() for reference model, but not for policy model if inside accumulation context?
                # Policy logprobs (adapter enabled) - Calculate outside no_grad context if using accumulate
                # Reference logprobs (adapter disabled) - Calculate inside no_grad context

                with torch.no_grad():
                     # Reference logprobs (adapter disabled)
                     with accelerator.unwrap_model(self.model).disable_adapter():
                         ref_outputs = forward(self.model, query_responses_ids, query_responses_mask)
                         ref_logits = ref_outputs.logits[:, context_length - 1 : -1]
                         ref_logits /= args.temperature + 1e-7 # Apply temperature? Yes, consistency.
                         ref_logprobs = selective_log_softmax(ref_logits, processed_responses_ids)
                         del ref_outputs, ref_logits
                         torch.cuda.empty_cache()

                # Policy logprobs (adapter enabled) - Calculated later during loss computation with grads enabled

                # --- Reward Calculation ---
                processed_responses_text = self.processing_class.batch_decode(processed_responses_ids, skip_special_tokens=True)
                scores = torch.zeros(args.local_dataloader_batch_size * args.rloo_k, device=device, dtype=torch.float)

                # Call reward function for each original prompt
                current_idx = 0
                # Original prompts text already decoded: prompts_text (len = local_bs)
                for i in range(args.local_dataloader_batch_size):
                     prompt_text = prompts_text[i]
                     target = targets[i]
                     k_completions = processed_responses_text[current_idx : current_idx + args.rloo_k]

                     # Call external reward function
                     import inspect
                     sig = inspect.signature(self.reward_fn)
                     reward_kwargs = self.reward_fn_kwargs.copy()
                     # No LLM object to pass anymore
                     # if 'llm' in sig.parameters: reward_kwargs['llm'] = None # Or remove if not needed
                     if 'target' in sig.parameters: reward_kwargs['target'] = target

                     # Check if reward_fn still expects 'llm'
                     if 'llm' in sig.parameters:
                         warnings.warn("Reward function expects 'llm' argument, but it's no longer provided when using vLLM API.", UserWarning)
                         # Remove llm from kwargs if present to avoid error
                         reward_kwargs.pop('llm', None)


                     k_scores = self.reward_fn(
                         prompt_text=prompt_text,
                         completions_text=k_completions,
                         **reward_kwargs
                     )

                     if not isinstance(k_scores, list) or len(k_scores) != args.rloo_k:
                         raise ValueError(f"Reward function must return a list of {args.rloo_k} floats.")

                     scores[current_idx : current_idx + args.rloo_k] = torch.tensor(k_scores, device=device, dtype=torch.float)
                     current_idx += args.rloo_k

                # Post-process scores
                contain_eos_token = torch.any(processed_responses_ids == self.processing_class.eos_token_id, dim=-1)
                if args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= args.missing_eos_penalty

                # Create padding mask for logprobs
                sequence_lengths = first_true_indices(processed_responses_ids == self.processing_class.pad_token_id) - 1
                response_idxs = torch.arange(processed_responses_ids.shape[1], device=device).repeat(processed_responses_ids.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)

                # Mask ref_logprobs (policy logprobs masked later)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # --- RLOO Advantage Calculation (Requires Policy Logprobs - calculated later) ---
                # For now, calculate KL component of reward
                # kl = logprobs - ref_logprobs # Cannot calculate yet

                # Normalize raw scores if needed
                if args.normalize_reward:
                    # Gather scores across all processes within the accumulation step
                    gathered_scores = accelerator.gather(scores)
                    # Exclude padding/dummy scores if any? Assume all scores are valid here.
                    mean_score = gathered_scores.mean()
                    std_score = gathered_scores.std()
                    scores = (scores - mean_score) / (std_score + 1e-8)
                    scores = torch.clamp(scores, -args.reward_clip_range, args.reward_clip_range)

                # Store data needed for optimization phase
                # We need queries_repeated, processed_responses_ids, ref_logprobs, scores, padding_mask
                all_query_ids_list.append(queries_repeated.cpu())
                all_response_ids_list.append(processed_responses_ids.cpu())
                # Store ref_logprobs sum and scores for advantage calculation later
                all_ref_logprobs_list.append(ref_logprobs.sum(1).cpu())
                all_scores_list.append(scores.cpu())
                # Store sequence lengths / padding mask info if needed, or recompute
                all_sequence_lengths_list.append(sequence_lengths.cpu())

                # --- End of Experience Generation (within gradient accumulation loop) ---
                del (prompts_data, queries_repeated, responses_ids, processed_responses_ids,
                     query_responses_ids, query_responses_mask, ref_logprobs, scores, padding_mask,
                     sequence_lengths)
                if 'k_scores' in locals(): del k_scores
                if 'ref_logits' in locals(): del ref_logits
                torch.cuda.empty_cache()
                gc.collect()

            # --- End of Gradient Accumulation Loop ---

            # --- Optimization Phase ---
            if not all_scores_list: continue # Skip if no data

            # Collate accumulated data from CPU lists to device tensors
            batch_query_ids = torch.cat(all_query_ids_list, dim=0).to(device)
            batch_response_ids = torch.cat(all_response_ids_list, dim=0).to(device)
            batch_ref_logprobs_sum = torch.cat(all_ref_logprobs_list, dim=0).to(device)
            batch_scores = torch.cat(all_scores_list, dim=0).to(device)
            batch_seq_lengths = torch.cat(all_sequence_lengths_list, dim=0).to(device)
            local_accumulation_batch_size = len(batch_scores) # Total samples (prompts*k*accum) on this device

            # RLOO updates once per batch of experience
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = torch.randperm(local_accumulation_batch_size, device=device)
                # Minibatch size = #prompts * k * world_size? No, use local size.
                # local_batch_size_per_step was #prompts * accum.
                # Let's define minibatch size based on samples
                local_samples_per_step = args.local_dataloader_batch_size * args.rloo_k * args.gradient_accumulation_steps
                # Use a fraction of this for minibatch? Or just one big batch? RLOO often uses one batch.
                # Let's stick to one batch for simplicity, matching original RLOO structure.
                # If memory is an issue, minibatching can be added here.

                mini_batch_inds = b_inds # Use all indices

                # Get minibatch data (which is the whole accumulated batch)
                mb_query_ids = batch_query_ids[mini_batch_inds]
                mb_response_ids = batch_response_ids[mini_batch_inds]
                mb_ref_logprobs_sum = batch_ref_logprobs_sum[mini_batch_inds]
                mb_scores = batch_scores[mini_batch_inds]
                mb_seq_lengths = batch_seq_lengths[mini_batch_inds]

                # Recompute policy logprobs with gradients enabled
                mb_query_responses_ids = torch.cat([mb_query_ids, mb_response_ids], dim=1)
                mb_query_mask = torch.ones_like(mb_query_ids)
                mb_resp_padding_mask = (mb_response_ids == self.processing_class.pad_token_id)
                mb_resp_attn_mask = ~mb_resp_padding_mask
                mb_query_responses_mask = torch.cat([mb_query_mask, mb_resp_attn_mask], dim=1)
                mb_context_length = mb_query_ids.shape[1]

                # Use accelerator.accumulate context for gradient handling
                with accelerator.accumulate(self.model):
                    # Forward pass for policy logprobs
                    output = forward(self.model, mb_query_responses_ids, mb_query_responses_mask)
                    logits = output.logits[:, mb_context_length - 1 : -1]
                    logits /= args.temperature + 1e-7

                    # Compute new logprobs (token level)
                    new_logprobs_token = selective_log_softmax(logits, mb_response_ids)

                    # Apply padding mask (recompute or use stored seq lengths)
                    mb_padding_mask = torch.arange(mb_response_ids.shape[1], device=device).repeat(mb_response_ids.shape[0], 1) > mb_seq_lengths.unsqueeze(1)
                    new_logprobs_token = torch.masked_fill(new_logprobs_token, mb_padding_mask, INVALID_LOGPROB)
                    new_logprobs_sum = new_logprobs_token.sum(1) # Sum logprobs for sequence

                    # --- Compute Advantages (now that we have policy logprobs) ---
                    kl = new_logprobs_sum - mb_ref_logprobs_sum # KL per sequence

                    # Combine score and KL penalty
                    if args.token_level_kl:
                        # This requires token-level KL, which is complex to integrate here.
                        # Let's stick to sequence-level KL penalty added to reward.
                        warnings.warn("token_level_kl=True is not fully supported with API generation, using sequence-level KL penalty.")
                        non_score_reward_per_step = -args.kl_coef * kl
                        rlhf_reward = non_score_reward_per_step + mb_scores
                    else:
                        # Sequence-level KL
                        non_score_reward_per_step = -args.kl_coef * kl
                        rlhf_reward = non_score_reward_per_step + mb_scores

                    # Calculate RLOO baseline and advantages
                    # Reshape rewards based on local batch size and k
                    # local_accumulation_batch_size = local_bs * k * accum_steps
                    num_prompts_in_batch = local_accumulation_batch_size // args.rloo_k
                    rlhf_reward_grouped = rlhf_reward.reshape(args.rloo_k, num_prompts_in_batch)
                    baseline = (rlhf_reward_grouped.sum(0, keepdim=True) - rlhf_reward_grouped) / (args.rloo_k - 1)
                    advantages = rlhf_reward_grouped - baseline
                    advantages = advantages.flatten() # Back to (local_bs * k * accum_steps)

                    # Normalize advantages if needed (gather across all processes for mean/std)
                    if args.normalize_advantage:
                         gathered_advantages = accelerator.gather(advantages)
                         # Filter NaNs/Infs just in case
                         gathered_advantages = gathered_advantages[~torch.isnan(gathered_advantages) & ~torch.isinf(gathered_advantages)]
                         if gathered_advantages.numel() > 1:
                             mean_adv = gathered_advantages.mean()
                             std_adv = gathered_advantages.std()
                             advantages = (advantages - mean_adv) / (std_adv + 1e-8)
                         else:
                             warnings.warn("Could not normalize advantages: Not enough valid values.")


                    # --- RLOO Loss Calculation ---
                    # Loss is - E[Adv * log P(response | prompt)]
                    # We use the sum of logprobs for the sequence.
                    pg_loss = -advantages * new_logprobs_sum # Use the recomputed logprobs
                    pg_loss = pg_loss.mean()

                    loss = pg_loss # Total loss (KL was included in reward)

                    # Backward pass (managed by accelerator.accumulate)
                    accelerator.backward(loss)

                    # --- Log Stats (inside accumulation context) ---
                    with torch.no_grad():
                         # Approx KL between old and new policy (using detached old logprobs requires storing them)
                         # Let's skip approx KL for now, focus on KL vs ref
                         policy_loss = pg_loss.item()
                         # Entropy calculation
                         prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                         entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                         masked_entropy = entropy.masked_fill(mb_padding_mask, 0.0)
                         mean_entropy = (masked_entropy.sum() / (~mb_padding_mask).sum()).item()

                         # Store stats from this minibatch/accumulation step
                         pg_loss_stats_accum.append(policy_loss)
                         entropy_stats_accum.append(mean_entropy)
                         # Store metrics calculated before advantage normalization
                         all_kl_list.append(kl.cpu()) # Store sequence KL (policy vs ref)
                         all_rlhf_rewards_list.append(rlhf_reward.cpu())
                         all_non_score_rewards_list.append(non_score_reward_per_step.cpu())
                         all_advantages_list.append(advantages.cpu()) # Store potentially normalized advantages


                    # End accumulate context

                # --- End Minibatch Loop (only one minibatch here) ---
            # --- End PPO Epoch Loop (only one epoch here) ---

            # --- Optimizer Step, LR Scheduling, Logging ---
            if accelerator.sync_gradients:
                # Clip gradients if needed
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad() # Zero gradients *after* optimizer step

                self.lr_scheduler.step()
                self.state.global_step += 1

                # Gather accumulated stats across all GPUs for logging
                if accelerator.is_main_process:
                     # Collate metrics calculated during optimization
                     log_scores = torch.cat(all_scores_list, dim=0).float() # From generation phase
                     log_advantages = torch.cat(all_advantages_list, dim=0).float() # From optimization phase
                     log_rlhf_rewards = torch.cat(all_rlhf_rewards_list, dim=0).float()
                     log_non_score_rewards = torch.cat(all_non_score_rewards_list, dim=0).float()
                     log_kl_sum = torch.cat(all_kl_list, dim=0).float() # KL(policy || ref)
                     log_seq_lengths = torch.cat(all_sequence_lengths_list, dim=0) # From generation phase

                     # Aggregate stats from optimization phase loop (only one entry if no minibatching)
                     mean_pg_loss = torch.tensor(np.mean(pg_loss_stats_accum), device=device) if pg_loss_stats_accum else torch.tensor(0.0, device=device)
                     mean_entropy = torch.tensor(np.mean(entropy_stats_accum), device=device) if entropy_stats_accum else torch.tensor(0.0, device=device)

                     # Reduce metrics across processes
                     mean_score_red = reduce(log_scores.mean(), reduction='mean')
                     mean_adv_red = reduce(log_advantages.mean(), reduction='mean')
                     std_adv_red = reduce(log_advantages.std(), reduction='mean')
                     mean_rlhf_reward_red = reduce(log_rlhf_rewards.mean(), reduction='mean')
                     mean_non_score_reward_red = reduce(log_non_score_rewards.mean(), reduction='mean')
                     mean_kl_red = reduce(log_kl_sum.mean(), reduction='mean')
                     mean_seq_len_red = reduce(log_seq_lengths.float().mean(), reduction='mean')
                     mean_pg_loss_red = reduce(mean_pg_loss, reduction='mean')
                     mean_entropy_red = reduce(mean_entropy, reduction='mean')

                     metrics = {}
                     metrics["train/episode"] = self.state.global_step * args.total_batch_size_per_update # Log based on prompts processed
                     metrics["train/reward_score"] = mean_score_red.item()
                     metrics["train/reward_rlhf"] = mean_rlhf_reward_red.item()
                     metrics["train/reward_non_score"] = mean_non_score_reward_red.item() # Should be ~ -kl_coef * kl_ref_policy
                     metrics["train/advantage_mean"] = mean_adv_red.item()
                     metrics["train/advantage_std"] = std_adv_red.item()
                     metrics["train/kl_ref_policy"] = mean_kl_red.item() # KL vs reference policy
                     metrics["train/policy_entropy"] = mean_entropy_red.item()
                     metrics["train/loss_policy"] = mean_pg_loss_red.item()
                     # metrics["train/kl_approx"] = mean_approxkl_red.item() # Not calculated
                     # metrics["train/ratio"] = mean_ratio_red.item() # Not calculated
                     metrics["train/seq_length"] = mean_seq_len_red.item()
                     metrics["train/lr"] = self.lr_scheduler.get_last_lr()[0]
                     metrics["train/epoch"] = self.state.epoch + micro_step / args.gradient_accumulation_steps / num_update_steps_per_epoch # More precise epoch

                     self.log(metrics)

            # --- Callback Handling, Checkpointing, Evaluation, Sample Generation ---
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            if self.control.should_save:
                 self._save_checkpoint() # Saves adapter to checkpoint dir
                 self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_evaluate:
                 # Evaluation needs adapting for API generation too
                 # metrics = self.evaluate()
                 # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
                 accelerator.print("Evaluation not yet implemented for API generation.")
                 pass

            if accelerator.is_main_process and args.num_sample_generations > 0 and self.state.global_step > 0 and \
               (self.state.global_step % self.sample_generations_freq == 0 or self.control.should_training_stop):
                 self.generate_completions(sampling=True) # Needs update for API

            if self.control.should_training_stop:
                 break

            # Update epoch state (based on prompts processed)
            self.state.epoch += 1 / num_update_steps_per_epoch

        # --- End of Training Loop ---
        accelerator.print("=== Finished Training ===")
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Final save
        if self.control.should_save:
            self._save_checkpoint()
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint(self, trial=None, metrics=None):
         """Saves the PEFT adapter."""
         if not self.is_world_process_zero():
             return

         save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
         print(f"Saving adapter checkpoint to {save_path}")
         # Ensure model is not wrapped by accelerator's utilities before saving PEFT model
         unwrapped_model = self.accelerator.unwrap_model(self.model)
         unwrapped_model.save_pretrained(save_path)
         # Optionally save tokenizer
         self.processing_class.save_pretrained(save_path)
         # Optionally save trainer state
         self.state.save_to_json(os.path.join(save_path, "trainer_state.json"))


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
         if not self._load_adapter_via_api(adapter_path_to_load, adapter_name_to_use):
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
