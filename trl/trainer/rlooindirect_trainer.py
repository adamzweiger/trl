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
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
from peft import PeftModel
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
from transformers.utils import is_peft_available

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    # batch_generation, # Replaced by vLLM
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    # get_reward, # Replaced by external function
    # prepare_deepspeed, # Not handling deepspeed explicitly here, Accelerator does
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)
from .rlooindirect_config import RLOOIndirectConfig
from .utils import generate_model_card # Reuse utils

if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model
else:
    raise ImportError("PEFT is required for RLOOIndirectTrainer. Please install peft.")

if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0

# Type hint for the external reward function
RewardFnType = Callable[
    [
        List[str],                 # prompt_texts (batch of prompts)
        List[List[str]],           # completions_texts (batch of lists of k completions)
        str,                       # task_type ('math', 'fewshot', 'cpt')
        Union[List[str], List[Dict]], # task_specific_data (targets or questions)
        Any,                       # trainer_class (self)
        Dict[str, Any],            # **kwargs
    ],
    List[List[float]]              # Returns list of lists of scores
]

class RLOOIndirectTrainer(Trainer):
    """
    Trainer for RLOO (REINFORCE Leave-One-Out) using an indirect reward function and vLLM.
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

        if not hasattr(args, 'task_type') or args.task_type not in ['math', 'fewshot', 'cpt']:
            raise ValueError("RLOOIndirectConfig must have a 'task_type' attribute set to 'math', 'fewshot', or 'cpt'.")
        self.task_type = args.task_type
        print(f"Initialized RLOOIndirectTrainer for task_type: {self.task_type}")

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

        self.model = get_peft_model(base_model, peft_config, adapter_name=self.args.lora_adapter_name)
        # self.model.print_trainable_parameters() # debug

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
        if hasattr(args, '_saved_remove_unused_columns'):
            self.args.remove_unused_columns = args._saved_remove_unused_columns
            # Let Trainer handle removal if needed

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
        self.reward_fn = None
        if not args.reward_fn_path:
                raise ValueError("reward_fn_path must be provided.")
        self.reward_fn = self._load_reward_function(args.reward_fn_path)
        print("Loaded external reward function.")
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

        # debug start
        accelerator.print("--- Initial Optimizer State After Prepare ---")
        accelerator.print(f"Optimizer Type: {type(self.optimizer)}")
        # If wrapped by AcceleratedOptimizer, access the original via .optimizer
        if hasattr(self.optimizer, 'optimizer'):
            accelerator.print(f"Wrapped Optimizer Type: {type(self.optimizer.optimizer)}")
            accelerator.print(f"Wrapped Optimizer Param Groups: {self.optimizer.optimizer.param_groups}")
            accelerator.print(f"Wrapped Optimizer Defaults: {self.optimizer.optimizer.defaults}")
        else:
            accelerator.print(f"Optimizer Param Groups: {self.optimizer.param_groups}")
            accelerator.print(f"Optimizer Defaults: {self.optimizer.defaults}")
        accelerator.print("-" * 40)
        # debug end

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
        
        self.accelerator.wait_for_everyone()

        print(f"[Process Index {self.args.process_index}] RLOOIndirectTrainer initialization finished.")

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
            return True

        load_url = f"{self.vllm_api_url}/v1/load_lora_adapter"
        payload = {
            "lora_name": adapter_name,
            "lora_path": adapter_path,
        }
        headers = {"Content-Type": "application/json"}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.api_session.post(load_url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                print(f"Successfully loaded adapter '{adapter_name}' from '{adapter_path}' via API.")
                return True
            except requests.exceptions.RequestException as e:
                print(f"Warning: API call to load adapter failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Error: Failed to load adapter '{adapter_name}' via API after {max_retries} attempts.")
                    return False
                time.sleep(2)
        return False
    
    def _unload_adapter_via_api(self, adapter_name: str) -> bool:
        """Dynamically unloads an adapter via the vLLM API."""
        if not self.is_world_process_zero():
            return True

        unload_url = f"{self.vllm_api_url}/v1/unload_lora_adapter"
        payload = {"lora_name": adapter_name}
        headers = {"Content-Type": "application/json"}
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.api_session.post(unload_url, json=payload, headers=headers, timeout=30)
                if response.status_code == 404:
                    print(f"Warning: API endpoint {unload_url} not found (404). Assuming unload is not supported or needed.")
                    return True
                response.raise_for_status()
                print(f"Successfully unloaded adapter '{adapter_name}' via API.")
                return True
            except requests.exceptions.RequestException as e:
                print(f"Warning: API call to unload adapter failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Warning: Failed to unload adapter '{adapter_name}' via API after {max_retries} attempts. Continuing...")
                    return False
                time.sleep(2)
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
                time.sleep(5)

        return [""] * len(prompts_text) * sampling_params.get("n", 1) # Should not be reached if retries work
    
    def _find_peft_model(self, model_to_search):
        MAX_UNWRAP_DEPTH = 5
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
                # Specific check if base_model itself is the Peft layer
                peft_model_instance = current_model.base_model
                break
            else:
                break # Cannot unwrap further
        return peft_model_instance

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


    def train_RL(self):
        """Main training loop."""
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            os.makedirs(self.adapter_save_path, exist_ok=True)
        accelerator.wait_for_everyone()

        dataloader = self.train_dataloader
        def repeat_generator():
            while True:
                for batch in dataloader:
                    yield batch
        iter_dataloader = iter(repeat_generator())

        accelerator.print("=== Training RLOO PEFT Adapter with vLLM ===")
        start_time = time.time()
        self.model.train()

        accelerator.print("Performing dummy forward pass to finalize model initialization...")
        try:
            # Use a fresh iterator/dataloader to avoid consuming from the main one
            with torch.no_grad():
                dummy_dataloader = self.get_train_dataloader()
                dummy_batch = next(iter(dummy_dataloader))
                # Move batch to device (Accelerator handles FSDP device placement)
                dummy_batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_batch.items()}
                 # Use only necessary inputs for forward pass
                dummy_inputs = {
                    "input_ids": dummy_batch_device["input_ids"],
                    "attention_mask": dummy_batch_device["attention_mask"],
                }
                # Ensure model is in eval for this if it influences layers like dropout, though dropout is disabled
                # self.model.eval() # Optional: Keep in train mode if dropout already disabled
                _ = self.model(**dummy_inputs)
                # self.model.train() # Switch back if eval was used

            accelerator.print("Dummy forward pass completed.")
            del dummy_dataloader, dummy_batch, dummy_batch_device, dummy_inputs # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            accelerator.print(f"Warning: Dummy forward pass failed: {e}. Proceeding anyway.")
        accelerator.wait_for_everyone() # Sync after dummy pass


        # Trainer state initialization
        self.state.global_step = 0
        self.state.epoch = 0
        num_update_steps_per_epoch = math.ceil( len(self.train_dataset) / (args.local_dataloader_batch_size * args.world_size * args.gradient_accumulation_steps) ) if args.local_dataloader_batch_size > 0 else 0
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

        # --- Initial Adapter Save (Crucial for first vLLM call) ---
        accelerator.print("Performing initial adapter save before training loop starts...")
        try:
            unwrapped_model_init = accelerator.unwrap_model(self.model, keep_torch_compile=False)
            peft_model_instance_init = self._find_peft_model(unwrapped_model_init) # Use the helper

            if peft_model_instance_init is None:
                raise RuntimeError(f"Rank {accelerator.process_index}: Could not find PeftModel instance for initial save.")
            else:
                if accelerator.is_main_process:
                    os.makedirs(self.adapter_save_path, exist_ok=True)
                accelerator.wait_for_everyone()

                peft_model_instance_init.save_pretrained(
                    self.adapter_save_path,
                    selected_adapters=[self.args.lora_adapter_name],
                    safe_serialization=True,
                    is_main_process=accelerator.is_main_process
                )
                if accelerator.is_main_process:
                    accelerator.print(f"Rank 0: Initial adapter '{self.args.lora_adapter_name}' saved to {self.adapter_save_path}")
                accelerator.wait_for_everyone() # Ensure save is complete everywhere
                accelerator.print("Initial adapter save complete.")

        except Exception as e:
            accelerator.print(f"Rank {accelerator.process_index}: Error during initial adapter saving: {e}")
            raise e

        # --- Training Loop ---
        for step in range(max_steps):
            step_start_time = time.time()
            self.model.train() # Ensure model is in train mode for each step start
            adapter_loaded_this_step = False # Flag for successful load in this step

            # Load the adapter saved in the *previous* step (or initial save for step 0)
            accelerator.print(f"[Step {step+1}/{max_steps}] Loading adapter '{self.vllm_adapter_name}' into vLLM for generation...")
            # Point to the subfolder where the specific adapter was saved
            adapter_path_for_vllm = os.path.join(self.adapter_save_path, self.args.lora_adapter_name)
            adapter_loaded = self._load_adapter_via_api(adapter_path_for_vllm, self.vllm_adapter_name)

            status_list = [adapter_loaded]
            if accelerator.num_processes > 1:
                broadcast_object_list(status_list, from_process=0)
            adapter_loaded_on_rank = status_list[0]

            if not adapter_loaded_on_rank:
                raise RuntimeError(f"Failed to load adapter '{self.vllm_adapter_name}' from path '{adapter_path_for_vllm}' into vLLM server (Rank {accelerator.process_index} sync'd failure). Stopping training.")
            else:
                adapter_loaded_this_step = True # Mark as successfully loaded for this step
                accelerator.print(f"Rank {accelerator.process_index}: Confirmed adapter '{self.vllm_adapter_name}' loaded successfully for step {step+1}.")

            # Placeholders for data accumulated over gradient accumulation steps (on CPU)
            all_query_ids_list_cpu = []
            all_response_ids_list_cpu = []
            all_ref_logprobs_sum_list_cpu = []
            # all_scores_list_cpu = [] # Moved: This will be populated *after* the loop
            all_sequence_lengths_list_cpu = []
            # New lists needed for deferred reward calculation
            all_prompts_text_list_cpu = []
            all_processed_responses_text_list_cpu = [] # Store decoded text for reward fn
            all_task_specific_data_list_cpu = [] # Will store targets or questions

            # --- Experience Generation Phase (NO GRADIENTS accumulated here) ---
            for micro_step in range(args.gradient_accumulation_steps):
                micro_step_start_time = time.time()
                # Get batch for this micro-step
                raw_batch = next(iter_dataloader)
                local_prompts_ids = raw_batch['input_ids'].to(device) # Move prompt IDs to device
                local_prompts_text = raw_batch['prompt']

                # Extract task-specific data based on task_type
                local_task_specific_data = None
                if self.task_type in ["math", "fewshot"]:
                    local_task_specific_data = raw_batch['target'] # List of strings
                elif self.task_type == "cpt":
                    local_task_specific_data = raw_batch['questions'] # List of dictionaries
                else:
                    # Should be caught earlier, but defensive check
                    raise ValueError(f"Invalid task type '{self.task_type}' encountered in training loop.")

                local_bs = len(local_prompts_text)
                accelerator.print(f"Rank {accelerator.process_index}, Step {step}, MicroStep {micro_step+1}/{args.gradient_accumulation_steps}: Got batch.")

                # --- vLLM Generation via API (using adapter loaded for this step) ---
                sampling_params_dict = {
                    "n": args.rloo_k,
                    "max_tokens": args.max_completion_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stop": [self.processing_class.eos_token] if args.stop_token_id == self.processing_class.eos_token_id else [],
                }

                # --- Gather prompts across devices for main process API call ---
                # Gather list of lists (or list of dicts for CPT) using gather_object
                all_prompts_text_gathered = gather_object(local_prompts_text)
                flat_generated_responses_text_global = []
                global_num_prompts = args.effective_prompt_batch_size
                expected_global_completions = global_num_prompts * args.rloo_k

                if accelerator.is_main_process:
                    if len(all_prompts_text_gathered) != global_num_prompts:
                         accelerator.print(f"Warning: Gathered prompt list size mismatch on main process. Got {len(all_prompts_text_gathered)}, expected {global_num_prompts}")
                         # Pad or truncate if necessary, although this indicates an issue
                         if len(all_prompts_text_gathered) < global_num_prompts:
                             all_prompts_text_gathered.extend(["[PAD PROMPT]"] * (global_num_prompts - len(all_prompts_text_gathered)))
                         else:
                             all_prompts_text_gathered = all_prompts_text_gathered[:global_num_prompts]


                    flat_generated_responses_text_global = self._generate_via_vllm_api(
                        all_prompts_text_gathered,
                        self.vllm_adapter_name,
                        sampling_params_dict
                    )
                    actual_len = len(flat_generated_responses_text_global)
                    if actual_len != expected_global_completions:
                         accelerator.print(f"Warning: vLLM returned {actual_len} completions, expected {expected_global_completions}. Padding/Truncating.")
                         if actual_len < expected_global_completions:
                            flat_generated_responses_text_global.extend([""] * (expected_global_completions - actual_len))
                         else:
                            flat_generated_responses_text_global = flat_generated_responses_text_global[:expected_global_completions]
                else:
                    # Allocate space on non-main processes
                    flat_generated_responses_text_global = [""] * expected_global_completions

                # Broadcast generated text to all processes
                object_list_to_broadcast = [flat_generated_responses_text_global]
                broadcast_object_list(object_list_to_broadcast, from_process=0)
                flat_generated_responses_text_global = object_list_to_broadcast[0]
                accelerator.print(f"Rank {accelerator.process_index}: Synced {len(flat_generated_responses_text_global)} completions.")


                # --- Process Generated Responses (Locally) ---
                local_start_idx = accelerator.process_index * local_bs * args.rloo_k
                local_end_idx = local_start_idx + local_bs * args.rloo_k
                local_generated_responses_text = flat_generated_responses_text_global[local_start_idx:local_end_idx]

                responses_tokenized = self.processing_class(
                    local_generated_responses_text, padding='longest', truncation=True,
                    max_length=args.max_completion_length, return_tensors="pt",
                ).to(device)
                local_responses_ids = responses_tokenized.input_ids

                # Remove potential BOS token if added by tokenizer
                if self.processing_class.bos_token_id is not None and local_responses_ids.shape[1] > 0:
                    if (local_responses_ids[:, 0] == self.processing_class.bos_token_id).all():
                        local_responses_ids = local_responses_ids[:, 1:]

                # Truncate at stop token if specified
                local_processed_responses_ids = local_responses_ids
                if args.stop_token_id is not None and args.stop_token_id != self.processing_class.pad_token_id:
                    local_processed_responses_ids = truncate_response(
                        args.stop_token_id, self.processing_class.pad_token_id, local_responses_ids
                    )

                # --- Calculate Reference Logprobs (Adapter Disabled, No Gradients) ---
                accelerator.print(f"Rank {accelerator.process_index}: Calculating reference logprobs...")
                local_queries_repeated = local_prompts_ids.repeat_interleave(args.rloo_k, dim=0)
                local_context_length = local_queries_repeated.shape[1]
                local_query_responses_ids = torch.cat([local_queries_repeated, local_processed_responses_ids], dim=1)
                local_query_mask = torch.ones_like(local_queries_repeated, device=device)
                local_resp_attn_mask = (local_processed_responses_ids != self.processing_class.pad_token_id).long()
                local_query_responses_mask = torch.cat([local_query_mask, local_resp_attn_mask], dim=1)

                local_ref_logprobs = None # Initialize
                try:
                    with torch.no_grad():
                        peft_model_instance_ref = self._find_peft_model(accelerator.unwrap_model(self.model))

                        if peft_model_instance_ref is None:
                            raise RuntimeError(f"Rank {accelerator.process_index}: Could not find PeftModel instance to disable adapter for reference pass.")

                        accelerator.print(f"Rank {accelerator.process_index}: Disabling adapters for reference pass...")
                        adapters_disabled = False
                        try:
                            peft_model_instance_ref.disable_adapter_layers()
                            adapters_disabled = True
                            accelerator.print(f"Rank {accelerator.process_index}: Adapters disabled, running ref forward pass...")
                            # Use the main model instance (potentially wrapped by FSDP)
                            ref_outputs = self.model(
                                input_ids=local_query_responses_ids,
                                attention_mask=local_query_responses_mask,
                                use_cache=False
                            )
                            accelerator.print(f"Rank {accelerator.process_index}: Ref forward pass completed.")

                            ref_logits = ref_outputs.logits[:, local_context_length - 1 : -1]
                            ref_logits /= (args.temperature + 1e-7)
                            local_ref_logprobs = selective_log_softmax(ref_logits, local_processed_responses_ids) # Shape: (bs*k, resp_len)

                            del ref_outputs, ref_logits

                        finally:
                            if adapters_disabled and peft_model_instance_ref is not None:
                                accelerator.print(f"Rank {accelerator.process_index}: Re-enabling adapters...")
                                peft_model_instance_ref.enable_adapter_layers()
                                accelerator.print(f"Rank {accelerator.process_index}: Adapters re-enabled.")

                except Exception as e:
                    accelerator.print(f"Rank {accelerator.process_index}: Error during reference logprob calculation: {e}")
                    import traceback
                    accelerator.print(traceback.format_exc())
                    # Attempt re-enable even on error
                    if adapters_disabled and peft_model_instance_ref is not None:
                         try:
                             accelerator.print(f"Rank {accelerator.process_index}: Attempting adapter re-enable after exception...")
                             peft_model_instance_ref.enable_adapter_layers()
                             accelerator.print(f"Rank {accelerator.process_index}: Adapters re-enabled after exception.")
                         except Exception as re_enable_e:
                             accelerator.print(f"Rank {accelerator.process_index}: Error during adapter re-enable after exception: {re_enable_e}")
                    raise e

                accelerator.print(f"Rank {accelerator.process_index}: Reference logprobs calculation block finished.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # --- Prepare and Store Data on CPU for Batched Reward & Optimization Later ---
                local_sequence_lengths = first_true_indices(local_processed_responses_ids == self.processing_class.pad_token_id) - 1
                local_padding_mask = torch.arange(local_processed_responses_ids.shape[1], device=device) > local_sequence_lengths.unsqueeze(1)
                local_ref_logprobs = torch.masked_fill(local_ref_logprobs, local_padding_mask, INVALID_LOGPROB) # Mask padded ref logprobs

                # Decode text responses needed for batched reward function later
                local_processed_responses_text = self.processing_class.batch_decode(
                    local_processed_responses_ids, skip_special_tokens=True
                )

                # Store necessary data on CPU lists
                all_query_ids_list_cpu.append(local_queries_repeated.cpu())
                all_response_ids_list_cpu.append(local_processed_responses_ids.cpu()) # Still needed for EOS penalty & policy pass
                all_ref_logprobs_sum_list_cpu.append(local_ref_logprobs.sum(1).cpu())
                all_sequence_lengths_list_cpu.append(local_sequence_lengths.cpu())
                # Store text data
                all_prompts_text_list_cpu.extend(local_prompts_text) # list of strings
                all_processed_responses_text_list_cpu.extend(local_processed_responses_text) # list of strings
                all_task_specific_data_list_cpu.extend(local_task_specific_data) # list of strings or dicts

                # --- Micro-step cleanup ---
                del (raw_batch, local_prompts_ids, local_prompts_text, local_task_specific_data,
                     local_generated_responses_text, responses_tokenized, local_responses_ids,
                     local_processed_responses_ids, local_queries_repeated, local_query_responses_ids,
                     local_query_mask, local_resp_attn_mask, local_query_responses_mask,
                     local_ref_logprobs, local_processed_responses_text, # Removed local_scores
                     local_sequence_lengths, local_padding_mask)
                # Less aggressive cache clearing
                if micro_step % 8 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                       torch.cuda.empty_cache()

                accelerator.print(f"    Micro-step {micro_step+1}/{args.gradient_accumulation_steps} completed (Data Gen/Ref Logprobs) in {time.time() - micro_step_start_time:.2f}s")

            # --- End of Experience Generation Loop ---
            accumulation_end_time = time.time()
            accelerator.print(f"  Experience Generation phase completed in {accumulation_end_time - step_start_time:.2f}s")

            # --- Unload Adapter from vLLM ---
            if adapter_loaded_this_step and accelerator.is_main_process:
                accelerator.print(f"[Step {step+1}/{max_steps}] Unloading adapter '{self.vllm_adapter_name}' at end of step...")
                unload_success = self._unload_adapter_via_api(self.vllm_adapter_name)
                if not unload_success:
                     accelerator.print(f"Warning: Failed to unload adapter '{self.vllm_adapter_name}' at end of step {step+1}.")
            # Sync after potential unload attempt before optimization
            accelerator.wait_for_everyone()

            # --- Reward Calculation Phase (using accumulated data) ---
            reward_calc_start_time = time.time()
            accelerator.print(f"Rank {accelerator.process_index}: Starting reward calculation phase...")
            # Note: all_*_list_cpu variables contain data accumulated over gradient_accumulation_steps

            # Store optimizer state on CPU before reward calculation if self-editing
            main_optimizer_state_memory = None
            accelerator.print(f"Rank {accelerator.process_index}: Offloading main optimizer state to CPU for self-edit reward calculation.")
            try:
                options = StateDictOptions(cpu_offload=True)
                main_optimizer_state_memory = get_optimizer_state_dict(
                    model=self.model, # Pass the Accelerator-prepared model
                    optimizers=self.optimizer,
                    options=options
                )
                accelerator.print(f"Rank {accelerator.process_index}: Optimizer state retrieved (potentially offloaded to CPU).")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                    accelerator.print(f"CRITICAL ERROR: Failed to get optimizer state: {e}. Aborting.")
                    raise e

            accelerator.print(f"Rank {accelerator.process_index}: Aggregating data for single external reward_fn call...")

            # --- Prepare inputs for the single batched reward function call ---
            # 1. Prompts: Already a flat list of all prompts accumulated
            all_prompts = all_prompts_text_list_cpu
            total_num_prompts = len(all_prompts)

            # 2. Completions: Reshape the flat list of all completions
            all_flat_completions = all_processed_responses_text_list_cpu
            expected_total_completions = total_num_prompts * args.rloo_k
            if len(all_flat_completions) != expected_total_completions:
                 raise ValueError(f"Expected {expected_total_completions} total completions accumulated, but found {len(all_flat_completions)}.")

            # Reshape flat list [p0_s0..p0_sk, p1_s0..p1_sk, ...] into [[p0_s0..p0_sk], [p1_s0..p1_sk], ...]
            all_nested_completions = [
                all_flat_completions[j * args.rloo_k : (j + 1) * args.rloo_k]
                for j in range(total_num_prompts)
            ]

            # 3. Task-specific data: Already a flat list corresponding to prompts
            all_task_specific_data = all_task_specific_data_list_cpu
            if len(all_task_specific_data) != total_num_prompts:
                raise ValueError(f"Mismatch between number of prompts ({total_num_prompts}) and task-specific data ({len(all_task_specific_data)}).")


            # --- Single call to the external reward function ---
            accelerator.print(f"Rank {accelerator.process_index}: Calling external reward_fn once for {total_num_prompts} prompts...")
            try:
                reward_kwargs = self.reward_fn_kwargs.copy()
                # Pass the aggregated lists directly
                nested_scores_list = self.reward_fn(
                    prompt_texts=all_prompts,
                    completions_texts=all_nested_completions,
                    task_type=self.task_type,
                    task_specific_data=all_task_specific_data,
                    trainer_class=self,
                    **reward_kwargs
                )
            except Exception as e:
                accelerator.print(f"Rank {accelerator.process_index}: Error calling external reward fn: {e}")
                import traceback
                accelerator.print(traceback.format_exc())
                raise e

            # --- Validate and process the returned scores ---
            if not isinstance(nested_scores_list, list) or len(nested_scores_list) != total_num_prompts:
                raise ValueError(f"External reward function must return a list of {total_num_prompts} lists (one per prompt). Got: {type(nested_scores_list)} with length {len(nested_scores_list)}")

            flat_scores = []
            for idx, sublist in enumerate(nested_scores_list):
                if not isinstance(sublist, list) or len(sublist) != args.rloo_k:
                    raise ValueError(f"External reward function must return lists of {args.rloo_k} floats for each prompt. Got for prompt {idx}: {type(sublist)} with length {len(sublist)}")
                try:
                    float_sublist = [float(s) for s in sublist]
                    flat_scores.extend(float_sublist)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"External reward function returned non-float score in sublist for prompt {idx}. Sublist: {sublist}. Error: {e}")

            # Convert the final flattened list of scores to a tensor
            batch_scores_cpu = torch.tensor(flat_scores, dtype=torch.float)
            if batch_scores_cpu.shape != (expected_total_completions,):
                 raise ValueError(f"Internal error: Shape mismatch after flattening scores. Expected ({expected_total_completions},), got {batch_scores_cpu.shape}")


            # --- Apply EOS Penalty (to the full batch) ---
            # Collate response IDs and sequence lengths from all micro-batches first
            batch_response_ids_cpu = torch.cat(all_response_ids_list_cpu, dim=0)
            batch_sequence_lengths_cpu = torch.cat(all_sequence_lengths_list_cpu, dim=0)

            if batch_response_ids_cpu.shape[0] != expected_total_completions or \
               batch_sequence_lengths_cpu.shape[0] != expected_total_completions:
                raise ValueError(f"Full batch shape mismatch between scores ({batch_scores_cpu.shape[0]}) and response data ({batch_response_ids_cpu.shape[0]}, {batch_sequence_lengths_cpu.shape[0]})")

            contain_eos_token = torch.any(
                (batch_response_ids_cpu == self.processing_class.eos_token_id) &
                (torch.arange(batch_response_ids_cpu.shape[1]) <= batch_sequence_lengths_cpu.unsqueeze(1)),
                dim=1
            )
            if args.missing_eos_penalty is not None:
                batch_scores_cpu[~contain_eos_token] -= args.missing_eos_penalty

            # Restore optimizer state from CPU if it was offloaded
            if main_optimizer_state_memory is not None:
                accelerator.print(f"Rank {accelerator.process_index}: Reloading main optimizer state from CPU.")
                try:
                    set_optimizer_state_dict(
                        model=self.model,
                        optimizers=self.optimizer,
                        optim_state_dict=main_optimizer_state_memory,
                    )

                    # Restore default hyperparameters if necessary (as before)
                    if hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
                         default_lr = getattr(self.args, "learning_rate", 1e-5)
                         default_betas = (getattr(self.args, "adam_beta1", 0.9), getattr(self.args, "adam_beta2", 0.999))
                         default_eps = getattr(self.args, "adam_epsilon", 1e-8)
                         default_weight_decay = getattr(self.args, "weight_decay", 0.0)
                         for group in self.optimizer.param_groups:
                             group.setdefault('lr', default_lr)
                             group.setdefault('betas', default_betas)
                             group.setdefault('eps', default_eps)
                             group.setdefault('weight_decay', default_weight_decay)
                             if 'betas' not in group:
                                  accelerator.print(f"CRITICAL WARNING: 'betas' still missing in param_group after setdefault attempt.")
                    accelerator.print(f"Rank {accelerator.process_index}: Optimizer state reloaded. Param group keys: {self.optimizer.param_groups[0].keys() if self.optimizer.param_groups else 'None'}")
                    del main_optimizer_state_memory # Clear memory
                    gc.collect() # Encourage garbage collection
                except Exception as e:
                    accelerator.print(f"CRITICAL WARNING: Failed to reload optimizer state: {e}. Training might be compromised.")
                    raise e # Stop training if restore fails

            accelerator.print(f"Rank {accelerator.process_index}: Reward calculation phase finished in {time.time() - reward_calc_start_time:.2f}s. Processed {expected_total_completions} samples.")
            # Clean up temporary aggregated tensors/lists used for EOS penalty
            del batch_response_ids_cpu, batch_sequence_lengths_cpu

            # --- Optimization Phase (Policy Forward/Backward) ---
            if batch_scores_cpu is None: # Check if scores were actually calculated (should always be non-None now)
                accelerator.print("Warning: No scores calculated, skipping optimization step.")
                continue

            # Collate accumulated data from CPU lists to device tensors
            try:
                batch_query_ids = torch.cat(all_query_ids_list_cpu, dim=0).to(device)
                batch_response_ids = torch.cat(all_response_ids_list_cpu, dim=0).to(device)
                batch_ref_logprobs_sum = torch.cat(all_ref_logprobs_sum_list_cpu, dim=0).to(device)
                # === Use the final batch_scores_cpu tensor calculated above ===
                batch_scores = batch_scores_cpu.to(device)
                # =============================================================
                batch_seq_lengths = torch.cat(all_sequence_lengths_list_cpu, dim=0).to(device)
                local_total_samples_in_batch = len(batch_scores)
                # Ensure consistency check
                if local_total_samples_in_batch != expected_total_completions:
                     accelerator.print(f"Warning: Collated batch size {local_total_samples_in_batch} does not match expected {expected_total_completions}")

            except RuntimeError as e:
                accelerator.print(f"Rank {accelerator.process_index}: Error during collation: {e}. Check accumulated data shapes.")
                raise e

            # Clear CPU lists immediately after collation
            del (all_query_ids_list_cpu, all_response_ids_list_cpu, all_ref_logprobs_sum_list_cpu,
                # No all_scores_list_cpu to delete
                all_sequence_lengths_list_cpu, batch_scores_cpu, # Clear the CPU scores tensor
                all_prompts_text_list_cpu, all_task_specific_data_list_cpu, all_processed_responses_text_list_cpu) # Clear text lists too
            gc.collect()

            # Recompute policy logprobs *with gradients enabled* within the accelerator context
            with accelerator.accumulate(self.model):
                optim_start_time = time.time()
                # Construct inputs for the policy forward pass
                mb_query_ids = batch_query_ids
                mb_response_ids = batch_response_ids
                mb_ref_logprobs_sum = batch_ref_logprobs_sum
                mb_scores = batch_scores # Use the scores calculated earlier
                mb_seq_lengths = batch_seq_lengths

                mb_query_responses_ids = torch.cat([mb_query_ids, mb_response_ids], dim=1)
                mb_query_mask = torch.ones_like(mb_query_ids, device=device)
                mb_response_idxs = torch.arange(mb_response_ids.shape[1], device=device).repeat(mb_response_ids.shape[0], 1)
                # Use <= for mask to include the token *at* the sequence length index if not padding
                mb_resp_attn_mask = (mb_response_idxs <= mb_seq_lengths.unsqueeze(1)).long()
                mb_query_responses_mask = torch.cat([mb_query_mask, mb_resp_attn_mask], dim=1)
                mb_context_length = mb_query_ids.shape[1]

                # Forward pass for policy logprobs (requires grad)
                self.model.train() # Ensure train mode
                output = self.model(
                    input_ids=mb_query_responses_ids,
                    attention_mask=mb_query_responses_mask,
                    use_cache=False
                )
                logits = output.logits[:, mb_context_length - 1 : -1]
                logits /= (args.temperature + 1e-7) # Apply temperature scaling

                new_logprobs_token = selective_log_softmax(logits, mb_response_ids) # Shape: (batch_total, resp_len)

                # Apply padding mask (use > to exclude padding tokens from sum)
                mb_padding_mask = mb_response_idxs > mb_seq_lengths.unsqueeze(1)
                new_logprobs_token_masked = torch.masked_fill(new_logprobs_token, mb_padding_mask, 0.0)
                new_logprobs_sum = new_logprobs_token_masked.sum(1) # Sum over sequence length

                # --- Compute Advantages ---
                kl_sum = new_logprobs_sum - mb_ref_logprobs_sum # KL per sequence

                # Normalize raw scores if needed
                current_scores_for_norm = mb_scores.clone() # Use a clone
                if args.normalize_reward:
                    # Gather scores across all processes for mean/std calculation
                    gathered_scores_all = accelerator.gather(current_scores_for_norm).float()
                    valid_scores = gathered_scores_all[~torch.isinf(gathered_scores_all) & ~torch.isnan(gathered_scores_all)]
                    if valid_scores.numel() > 1:
                        mean_score = valid_scores.mean()
                        std_score = valid_scores.std()
                        # Apply normalization using global stats back to local scores
                        current_scores_for_norm = (current_scores_for_norm - mean_score) / (std_score + 1e-8)
                        # Clip normalized scores
                        current_scores_for_norm = torch.clamp(current_scores_for_norm, -args.reward_clip_range, args.reward_clip_range)
                    elif valid_scores.numel() == 1:
                         mean_score = valid_scores.mean()
                         current_scores_for_norm = current_scores_for_norm - mean_score # Center only
                         accelerator.print("Warning: Only one valid reward score found after gathering. Centering rewards but not scaling.")
                    else: # valid_scores.numel() == 0
                         accelerator.print("Warning: Could not normalize rewards: No valid values gathered.")
                         # Keep original scores if normalization failed

                # Combine score and KL penalty
                non_score_reward_per_seq = -args.kl_coef * kl_sum
                rlhf_reward = non_score_reward_per_seq + current_scores_for_norm # Use potentially normalized scores

                # Calculate RLOO baseline and advantages
                # Ensure calculation uses the correct local batch size dimensions
                num_prompts_in_local_batch = local_total_samples_in_batch // args.rloo_k
                if local_total_samples_in_batch % args.rloo_k != 0:
                     # This should ideally not happen if data loading and accumulation are correct
                     raise ValueError(f"Total samples {local_total_samples_in_batch} not divisible by rloo_k {args.rloo_k}.")

                try:
                    # Reshape assuming order [p0_s0...p0_sk-1, p1_s0...p1_sk-1, ...]
                    rlhf_reward_grouped = rlhf_reward.reshape(num_prompts_in_local_batch, args.rloo_k).transpose(0, 1) # Shape: (k, num_prompts_local)
                except Exception as e:
                     # Add more context to the error message
                    raise RuntimeError(f"Failed reshape rlhf_reward. Shape: {rlhf_reward.shape}, k: {args.rloo_k}, num_prompts_local: {num_prompts_in_local_batch}, local_total_samples: {local_total_samples_in_batch}. Err: {e}")


                if args.rloo_k > 1:
                    baseline = (rlhf_reward_grouped.sum(0, keepdim=True) - rlhf_reward_grouped) / (args.rloo_k - 1)
                else:
                    baseline = torch.zeros_like(rlhf_reward_grouped) # No baseline if k=1
                advantages_grouped = rlhf_reward_grouped - baseline
                advantages = advantages_grouped.transpose(0, 1).flatten() # Shape: (local_total_samples_in_batch,)

                # Normalize advantages if needed
                if args.normalize_advantage:
                    gathered_advantages_all = accelerator.gather(advantages).float()
                    valid_advantages = gathered_advantages_all[~torch.isnan(gathered_advantages_all) & ~torch.isinf(gathered_advantages_all)]
                    if valid_advantages.numel() > 1:
                        mean_adv = valid_advantages.mean()
                        std_adv = valid_advantages.std()
                        advantages = (advantages - mean_adv) / (std_adv + 1e-8) # Normalize local advantages using global stats
                    elif valid_advantages.numel() > 0: # Handle case with only one valid advantage (center only)
                        mean_adv = valid_advantages.mean()
                        advantages = advantages - mean_adv
                        accelerator.print(f"Warning: Centering advantages but not scaling (num valid: {valid_advantages.numel()}, std: {valid_advantages.std() if valid_advantages.numel() > 1 else 'N/A'}).")
                    else: # No valid advantages found
                        accelerator.print("Warning: Could not normalize advantages: No valid values found after gathering.")
                        # Keep original advantages if normalization failed

                # --- RLOO Loss Calculation ---
                # Detach advantages - gradients flow through logprobs only
                pg_loss = (-advantages.detach() * new_logprobs_sum).mean()
                loss = pg_loss

                # Backward pass (managed by accelerator.accumulate)
                accelerator.backward(loss)
                optim_end_time = time.time()
                accelerator.print(f"    Optimization forward/backward completed in {optim_end_time - optim_start_time:.2f}s")

                # --- Log Stats (inside accumulation context, before optimizer step) ---
                # Store detached tensors for potential gathering later
                stats_to_log = {
                    "policy_loss": pg_loss.detach(),
                    "kl_sum": kl_sum.detach().mean(), # Log mean KL per sequence
                    "rlhf_reward": rlhf_reward.detach(), # Reward incl. KL penalty
                    "non_score_reward": non_score_reward_per_seq.detach(), # Just KL penalty part
                    "advantages": advantages.detach(), # Potentially normalized advantages
                    "scores": mb_scores.detach(), # Original scores from reward func + EOS penalty
                    "seq_lengths": batch_seq_lengths.detach().float() # Use the collated lengths
                }
                # Calculate entropy (optional)
                if args.log_policy_entropy:
                    with torch.no_grad():
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                        entropy_token = torch.sum(-prob_dist * torch.log(prob_dist + 1e-9), dim=-1) # Avoid log(0)
                        masked_entropy = entropy_token.masked_fill(mb_padding_mask, 0.0)
                        # Average entropy over non-padded tokens across the batch
                        num_non_pad_tokens = (~mb_padding_mask).sum()
                        mean_entropy = masked_entropy.sum() / num_non_pad_tokens if num_non_pad_tokens > 0 else torch.tensor(0.0, device=device)
                        stats_to_log["policy_entropy"] = mean_entropy.detach()

            # --- End Accumulate Context ---

            # --- Optimizer Step, LR Scheduling, Logging, Saving ---
            if accelerator.sync_gradients:
                optimizer_step_start_time = time.time()
                # Clip gradients *before* optimizer step
                if args.max_grad_norm is not None:
                    # Ensure model parameters are passed correctly, especially with FSDP
                    model_params_to_clip = self.model.parameters()
                    accelerator.clip_grad_norm_(model_params_to_clip, args.max_grad_norm)

                # Optimizer Step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.state.global_step += 1

                accelerator.print(f"    Optimizer step completed in {time.time() - optimizer_step_start_time:.2f}s")

                # --- Gather and Log Metrics ---
                # Gather tensors from all processes ONLY IF sync_gradients is True
                gathered_stats_tensors = {}
                for name, tensor in stats_to_log.items():
                    try:
                        # Gather the tensor across all processes. Result is on rank 0 by default.
                        gathered_tensor = accelerator.gather(tensor).float() # Gather returns on rank 0
                        gathered_stats_tensors[name] = gathered_tensor # Store on rank 0
                    except Exception as e:
                        gathered_stats_tensors[name] = None # Mark as failed on rank 0
                        accelerator.print(f"Rank {accelerator.process_index}: Warning during gather for metric '{name}': {e}")


                # Process and log gathered results ONLY on the main process
                if accelerator.is_main_process:
                    metrics = {}
                    for name, gathered_tensor in gathered_stats_tensors.items():
                        if gathered_tensor is None:
                            metrics[f"train/{name}_mean"] = float('nan')
                            if name in ["advantages", "scores", "rlhf_reward"]:
                                metrics[f"train/{name}_std"] = float('nan')
                            continue

                        try:
                            # Calculate mean/std from the gathered tensor (now on rank 0)
                            valid_tensor = gathered_tensor[~torch.isnan(gathered_tensor) & ~torch.isinf(gathered_tensor)]
                            if valid_tensor.numel() > 0:
                                metrics[f"train/{name}_mean"] = valid_tensor.mean().item()
                                if name in ["advantages", "scores", "rlhf_reward", "seq_lengths"]: # Log std for key metrics
                                    metrics[f"train/{name}_std"] = valid_tensor.std().item() if valid_tensor.numel() > 1 else 0.0
                            else:
                                metrics[f"train/{name}_mean"] = float('nan')
                                if name in ["advantages", "scores", "rlhf_reward", "seq_lengths"]:
                                    metrics[f"train/{name}_std"] = float('nan')
                        except Exception as e:
                            accelerator.print(f"Warning: Failed to process gathered metric '{name}': {e}")
                            metrics[f"train/{name}_mean"] = float('nan')
                            if name in ["advantages", "scores", "rlhf_reward", "seq_lengths"]:
                                metrics[f"train/{name}_std"] = float('nan')

                    # Rename specific metrics for clarity
                    if "train/policy_loss_mean" in metrics: metrics["train/loss_policy"] = metrics.pop("train/policy_loss_mean")
                    if "train/kl_sum_mean" in metrics: metrics["train/kl_divergence"] = metrics.pop("train/kl_sum_mean") # Renamed from sum to mean earlier
                    if "train/rlhf_reward_mean" in metrics: metrics["train/reward_rlhf_mean"] = metrics.pop("train/rlhf_reward_mean")
                    if "train/rlhf_reward_std" in metrics: metrics["train/reward_rlhf_std"] = metrics.pop("train/rlhf_reward_std")
                    if "train/non_score_reward_mean" in metrics: metrics["train/reward_non_score_mean"] = metrics.pop("train/non_score_reward_mean")
                    if "train/advantages_mean" in metrics: metrics["train/advantages_mean"] = metrics.pop("train/advantages_mean")
                    if "train/advantages_std" in metrics: metrics["train/advantages_std"] = metrics.pop("train/advantages_std")
                    if "train/scores_mean" in metrics: metrics["train/reward_scores_mean"] = metrics.pop("train/scores_mean") # Reward from score func + penalty
                    if "train/scores_std" in metrics: metrics["train/reward_scores_std"] = metrics.pop("train/scores_std")
                    if "train/seq_lengths_mean" in metrics: metrics["train/response_length_mean"] = metrics.pop("train/seq_lengths_mean")
                    if "train/seq_lengths_std" in metrics: metrics["train/response_length_std"] = metrics.pop("train/seq_lengths_std")
                    if "train/policy_entropy_mean" in metrics: metrics["train/policy_entropy"] = metrics.pop("train/policy_entropy_mean")

                    # Add other standard metrics
                    metrics["train/lr"] = self.lr_scheduler.get_last_lr()[0]
                    current_epoch = (self.state.global_step * args.gradient_accumulation_steps * args.local_dataloader_batch_size * args.world_size) / len(self.train_dataset) if len(self.train_dataset) > 0 else 0
                    metrics["train/epoch"] = current_epoch
                    metrics["train/step"] = self.state.global_step # Use global step for x-axis

                    # Log the final aggregated metrics dictionary
                    self.log(metrics)

                # --- Save Adapter Checkpoint (AFTER optimizer step) ---
                # This adapter state will be used by vLLM in the *next* step's generation phase
                save_adapter_start_time = time.time()
                accelerator.print(f"Rank {accelerator.process_index}: Saving adapter after optimizer step {self.state.global_step}...")
                try:
                    # It's crucial to unwrap the model correctly, especially under FSDP
                    unwrapped_model_save = accelerator.unwrap_model(self.model, keep_torch_compile=False)
                    peft_model_instance_save = self._find_peft_model(unwrapped_model_save)

                    if peft_model_instance_save is None:
                        raise RuntimeError(f"Rank {accelerator.process_index}: Could not find PeftModel instance for saving after step {self.state.global_step}.")
                    else:
                        # Define the specific subdirectory for this adapter
                        adapter_save_subdir = os.path.join(self.adapter_save_path, self.args.lora_adapter_name)
                        if accelerator.is_main_process:
                            os.makedirs(adapter_save_subdir, exist_ok=True) # Ensure sub-directory exists
                        accelerator.wait_for_everyone() # Sync before save call

                        # Call save_pretrained on ALL ranks. FSDP handles gathering parameters.
                        peft_model_instance_save.save_pretrained(
                            adapter_save_subdir, # Save to the specific adapter's subfolder
                            selected_adapters=[self.args.lora_adapter_name],
                            safe_serialization=True,
                            is_main_process=accelerator.is_main_process # Rank 0 writes files
                        )

                        if accelerator.is_main_process:
                             accelerator.print(f"Rank 0: Initiated save_pretrained call to {adapter_save_subdir} for step {self.state.global_step}.")

                        accelerator.wait_for_everyone() # Ensure save is complete before proceeding
                        accelerator.print(f"Rank {accelerator.process_index}: Adapter save completed in {time.time() - save_adapter_start_time:.2f}s.")

                except Exception as e:
                    accelerator.print(f"Rank {accelerator.process_index}: Error during adapter saving after step {self.state.global_step}: {e}")
                    raise e # Re-raise to stop training

                # --- Callbacks, Evaluation, Checkpointing (based on global_step) ---
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                if self.control.should_save:
                    accelerator.print(f"Triggering checkpoint save at step {self.state.global_step}...")
                    self._save_checkpoint() # Saves full training state via Accelerator
                    self.control = self.callback_handler.on_save(self.args, self.state, self.control)

                # # Sample generation check
                # if accelerator.is_main_process and args.num_sample_generations > 0 and \
                #    (self.state.global_step % self.sample_generations_freq == 0 or self.control.should_training_stop):
                #     accelerator.print(f"Generating samples at step {self.state.global_step}...")
                #     self.model.eval()
                #     # Ensure generate_completions uses the *current* model state (or vLLM if adapted)
                #     self.generate_completions(sampling=True)
                #     self.model.train()

            # --- End Sync Gradients Block ---
            # if self.control.should_training_stop:
            #     accelerator.print("Training stopping signal received.")
            #     break

            step_end_time = time.time()
            accelerator.print(f"  Full step {step+1}/{max_steps} (Global Step: {self.state.global_step}) completed in {step_end_time - step_start_time:.2f}s")
            gc.collect() # Collect garbage at end of step

        # --- End of Training Loop ---
        end_time = time.time()
        accelerator.print(f"Total training time: {end_time - start_time:.2f}s")
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        accelerator.wait_for_everyone()
        accelerator.print("=== Finished Training ===")

    def _save_checkpoint(self, trial=None, metrics=None):
            """Saves the training state using Accelerator for FSDP compatibility."""
            # Accelerator's save_state handles FSDP state saving across processes.
            # All processes must call this function.
            save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")

            if self.is_world_process_zero():
                os.makedirs(save_path, exist_ok=True)
                print(f"Rank 0: Preparing to save checkpoint to {save_path}")
            self.accelerator.wait_for_everyone()

            try:
                print(f"Rank {self.accelerator.process_index}: Calling accelerator.save_state to {save_path}")
                # save_state saves the model (including PEFT adapters if integrated correctly),
                # optimizer, scheduler, and any registered custom states.
                self.accelerator.save_state(save_path)
                self.accelerator.wait_for_everyone() # Ensure saving is complete everywhere
                if self.is_world_process_zero():
                    print(f"Rank 0: Successfully saved checkpoint: {save_path}")
                    self.processing_class.save_pretrained(save_path)
                    self.state.save_to_json(os.path.join(save_path, "trainer_state.json"))

            except Exception as e:
                print(f"Rank {self.accelerator.process_index}: Error during accelerator.save_state: {e}")
                raise e

            self.accelerator.wait_for_everyone()


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
            model_name=model_name or f"{base_model_name}-{self.args.lora_adapter_name}-RLOO-Indirect",
            hub_model_id=self.hub_model_id, # Trainer doesn't set this by default, needs push_to_hub setup
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="RLOOIndirect",
            trainer_citation=citation,
            paper_title="Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs",
            paper_id="2402.14740",
            # peft_config=self.model.peft_config[self.args.lora_adapter_name].to_dict()
        )

        # Save the model card
        output_path = os.path.join(self.args.output_dir, "README.md")
        model_card.save(output_path)
        print(f"Model card saved to {output_path}")
