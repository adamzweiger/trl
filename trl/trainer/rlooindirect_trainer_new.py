# rlooindirect_trainer_new.py
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
import zmq
import time
import json
import textwrap
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
    TrainerState, # Use standard TrainerState if OnlineTrainerState is not essential
    DefaultFlowCallback, # Import standard callbacks
    PrinterCallback,
    ProgressCallback,
    is_wandb_available,
)
from transformers.trainer_callback import CallbackHandler # Correct import
from transformers.utils import is_peft_available

# Use standard reporting integrations
# from transformers.integrations import get_reporting_integration_callbacks, WandbCallback

from ..models.utils import unwrap_model_for_generation # Keep this if used
from ..trainer.utils import (
    # OnlineTrainerState, # Replaced with TrainerState for now
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
    # prepare_deepspeed, # Accelerator handles this
)
from .rlooindirect_config import RLOOIndirectConfig
from .utils import generate_model_card # Reuse utils

if is_peft_available():
    from peft import LoraConfig, PeftModel, get_peft_model
else:
    raise ImportError("PEFT is required for RLOOIndirectTrainer. Please install peft.")

if is_wandb_available():
    import wandb
    # Ensure WandbCallback is imported if used
    try:
        from transformers.integrations import WandbCallback
    except ImportError:
        WandbCallback = None # Handle case where it might not be available

INVALID_LOGPROB = 1.0
DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# ZMQ placeholder (uncomment and configure when ready)
# ZMQ_CONTEXT = zmq.Context()
# ZMQ_SOCKET = ZMQ_CONTEXT.socket(zmq.REQ)
# ZMQ_SOCKET.connect("tcp://localhost:5555") # Replace with your ZMQ server address


def log_on_all_ranks(accelerator, message):
    """Prints a message prefixed with the rank index on all processes."""
    # if accelerator.is_main_process:
    #     print(f"[Rank {accelerator.process_index}] {message}")
    print(f"[Rank {accelerator.process_index}] {message}")
    # Optional: Add a barrier if you want sync, but usually not needed for prints.
    # accelerator.wait_for_everyone()

def safe_stats(tensor: Optional[torch.Tensor], name: str = "") -> Dict[str, float]:
    """Calculates safe mean, std, min, max, filtering NaNs/Infs."""
    stats = {'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'max': float('nan'), 'numel': 0, 'num_nan': 0, 'num_inf': 0}
    prefix = f"{name}_" if name else ""

    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        stats_named = {f"{prefix}{k}": v for k, v in stats.items()}
        return stats_named

    try:
        t_float = tensor.detach().float() # Work with float32 for stats
        stats['numel'] = t_float.numel()

        nan_mask = torch.isnan(t_float)
        inf_mask = torch.isinf(t_float)
        valid_mask = ~nan_mask & ~inf_mask

        stats['num_nan'] = nan_mask.sum().item()
        stats['num_inf'] = inf_mask.sum().item()

        valid_tensor = t_float[valid_mask]

        if valid_tensor.numel() > 0:
            stats['mean'] = valid_tensor.mean().item()
            stats['std'] = valid_tensor.std().item() if valid_tensor.numel() > 1 else 0.0
            stats['min'] = valid_tensor.min().item()
            stats['max'] = valid_tensor.max().item()
    except Exception as e:
        print(f"[WARN] Error calculating stats for tensor '{name}': {e}")
        # Return stats as they are (likely NaNs)

    stats_named = {f"{prefix}{k}": v for k, v in stats.items()}
    return stats_named


class RLOOIndirectTrainer2(Trainer):
    """
    Trainer for RLOO (REINFORCE Leave-One-Out) using an indirect reward function,
    vLLM generation, PEFT, and a PPO-style optimization loop.
    """
    _tag_names = ["trl", "rloo-indirect", "peft", "vllm", "fsdp", "ppo-style"]

    def __init__(
        self,
        model_name_or_path: str,
        config: RLOOIndirectConfig,
        processing_class: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
    ) -> None:
        if not is_peft_available():
            raise ImportError("PEFT is required for RLOOIndirectTrainer but is not installed.")

        self.args = config # Use self.args consistently like Trainer base class
        args = config # Local alias

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
        }
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Policy model (the one we train)
        self.policy_model = get_peft_model(base_model, peft_config, adapter_name=args.lora_adapter_name)
        # Disable dropout
        disable_dropout_in_model(self.policy_model)
        # We don't need a separate ref_policy object, we'll disable adapters on policy_model

        # --- Tokenizer & Data Collator ---
        self.processing_class = processing_class # Rename for clarity within class
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)
        self.data_collator = data_collator

        # --- Accelerator ---
        # Initialize Accelerator early to get world size, etc.
        # Gradient accumulation now happens *within* the PPO loop based on mini-batch vs device batch size
        
        # self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps) # Used for accumulate context
        # print(f"[Process Index {self.accelerator.process_index} / World Size {args.world_size}] Accelerator initialized.")

        # --- Batch Size Calculations (PPO-style) ---
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.eval_dataset = eval_dataset

        # args.per_device_train_batch_size = # prompts per device per *micro* step (within PPO loop accumulation)
        # args.gradient_accumulation_steps = # micro steps per optimizer update (within PPO loop)

        # Rollout batch size: How many prompts are generated *before* starting the PPO optimization phase.
        # Let's assume `args.local_batch_size_rollout` in config defines the prompts per device *per rollout step*
        # and `args.rollout_accumulation_steps` defines how many such steps form one full rollout batch.
        # If these aren't in config, we derive from PPO params. Let's use PPO naming directly.

        # `args.per_device_train_batch_size`: Prompts per device for a single forward/backward within the PPO loop.
        # `args.gradient_accumulation_steps`: How many fwd/bwd passes per optimizer step *within* the PPO loop.

        # --- Call Super Init ---
        # This handles: optimizer/scheduler creation, callback setup, state init,
        # logging setup, device placement, distributed setup coordination.
        super().__init__(
            model=self.policy_model,
            args=args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.processing_class,
            callbacks=callbacks,
            optimizers=optimizers
        )

        # Validate required PPO parameters exist in args
        required_ppo_args = ['per_device_train_batch_size', 'gradient_accumulation_steps',
                             'num_ppo_epochs', 'num_mini_batches', 'rloo_k', 'max_steps']
        for arg_name in required_ppo_args:
            if not hasattr(args, arg_name) or getattr(args, arg_name) is None:
                raise ValueError(f"RLOOIndirectConfig is missing required argument: {arg_name}")
            # Add basic positivity checks
            if arg_name in ['per_device_train_batch_size', 'gradient_accumulation_steps', 'num_ppo_epochs', 'num_mini_batches', 'rloo_k', 'max_steps'] and getattr(args, arg_name) <= 0:
                 raise ValueError(f"Argument `{arg_name}` must be positive, got {getattr(args, arg_name)}")

        # Use args.max_steps as the definitive total PPO steps
        total_ppo_steps = args.max_steps
        print(f"Using configured `max_steps` (total PPO optimization steps): {total_ppo_steps}")
        self.state.max_steps = total_ppo_steps # Ensure Trainer state reflects this
        self.state.logging_steps = args.logging_steps
        self.state.save_steps = args.save_steps
        self.state.eval_steps = args.eval_steps

        # Calculate PPO batch sizes based on config and world size
        args.mini_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
        args.batch_size = args.mini_batch_size * args.num_mini_batches
        args.local_batch_size = exact_div(args.batch_size, args.world_size, "`batch_size` / world_size")
        args.local_mini_batch_size = exact_div(args.local_batch_size, args.num_mini_batches, "`local_batch_size` / num_mini_batches")

        # Calculate num_total_batches (rollouts) based on max_steps
        steps_per_rollout = args.num_ppo_epochs * args.num_mini_batches
        if steps_per_rollout <= 0:
             raise ValueError("`num_ppo_epochs * num_mini_batches` must be positive to calculate rollouts.")
        args.num_total_batches = math.ceil(total_ppo_steps / steps_per_rollout)
        print(f"Calculated `num_total_batches` (rollouts) based on max_steps: {args.num_total_batches}")

        # Calculate dataloader batch size
        self.local_dataloader_batch_size = exact_div(args.local_mini_batch_size, args.rloo_k, "`local_mini_batch_size` / rloo_k")
        print(f"Calculated `local_dataloader_batch_size` (prompts per fetch): {self.local_dataloader_batch_size}")

        # Calculate effective number of epochs/episodes based on the derived rollouts
        # These are informational and not used to control training length
        if args.batch_size > 0 :
            calculated_total_episodes = args.num_total_batches * args.batch_size
        else:
            calculated_total_episodes = 0 # Should not happen if checks pass
        self.state.num_train_epochs = calculated_total_episodes / self.train_dataset_len if self.train_dataset_len > 0 else 0
        print(f"Informational: Equivalent num_train_epochs ~ {self.state.num_train_epochs:.2f}")

        # --- Optimizer and Scheduler ---
        if self.optimizer is None:
            self.create_optimizer_and_scheduler(num_training_steps=total_ppo_steps)

        # --- Run Name & Seed ---
        time_tensor = torch.tensor(int(time.time()), device=self.accelerator.device)
        time_int = broadcast(time_tensor, 0).item()
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + self.accelerator.process_index * 100003
        if hasattr(args, 'num_sample_generations') and args.num_sample_generations > 0:
             self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        else:
             self.sample_generations_freq = -1 # Disable if not specified

        # --- Optimizer and Scheduler ---
        # Note: PPO typically steps scheduler once per rollout/batch
        # self.optimizer = None
        # self.lr_scheduler = None
        # self.optimizer_cls_and_kwargs = None
        # self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

        # --- Trainer Internals (adapted from Trainer/RLOOTrainer/Reference) ---
        # Call super().__init__ AFTER setting up model, optimizer etc.
        # We bypass some standard Trainer init by setting members directly.
        self.model_wrapped = None # Set after prepare
        # self.ref_policy = None # Not needed, handled via adapter enable/disable

        # # Callbacks
        # default_callbacks = DEFAULT_CALLBACKS[:] # Use default Transformer callbacks
        # # Add reporting callbacks if specified
        # # if args.report_to:
        # #     reporting_callbacks = get_reporting_integration_callbacks(args.report_to)
        # #     default_callbacks.extend(reporting_callbacks)

        # self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        # self.callback_handler = CallbackHandler(
        #     self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        # )
        # # Add progress callback
        # self.add_callback(PrinterCallback if args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        # self.control = TrainerControl()
        # # Use standard TrainerState
        # self.state = TrainerState(
        #     is_local_process_zero=self.is_local_process_zero(),
        #     is_world_process_zero=self.is_world_process_zero(),
        # )

        # Other Trainer internals
        self.is_fsdp_enabled = isinstance(self.accelerator.state.fsdp_plugin, FullyShardedDataParallelPlugin)
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None # Check Deepspeed

        # Add PEFT model tags
        if hasattr(self.model, "add_model_tags"):
            try:
                # Need to find the PeftModel instance potentially wrapped by FSDP
                peft_model_instance = self._find_peft_model(self.model) # Find before prepare
                if peft_model_instance:
                    peft_model_instance.add_model_tags(self._tag_names)
                else:
                    print(f"[Rank {self.accelerator.process_index}] Warning: Could not find PeftModel instance to add tags before prepare.")
            except Exception as e:
                 print(f"[Rank {self.accelerator.process_index}] Warning: Failed to add model tags: {e}")

        # --- Stop Token ID ---
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id
        elif args.stop_token:
             # Maybe allow specifying other stop tokens?
             print(f"Warning: Custom stop_token '{args.stop_token}' specified but only 'eos' is directly handled for stop_token_id assignment.")
             args.stop_token_id = None # Or look up ID if needed elsewhere
        else:
            args.stop_token_id = None

        # --- Reward Function Loading / ZMQ ---
        self.reward_fn_kwargs = args.reward_fn_kwargs or {}
        # self.zmq_context = ZMQ_CONTEXT # Use shared context/socket
        # self.zmq_socket = ZMQ_SOCKET

        # --- vLLM API Client ---
        self.vllm_api_url = args.vllm_api_url
        self.adapter_save_path = args.adapter_save_dir # Path where current adapter is saved
        self.vllm_adapter_name = args.vllm_adapter_name # Name used for dynamic loading
        self.api_session = requests.Session() # Use a session

        # --- Output Dir ---
        if self.is_world_process_zero():
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(self.adapter_save_path, exist_ok=True)
            print(f"[Rank 0] Ensured output directories exist: {args.output_dir}, {self.adapter_save_path}")
        self.accelerator.wait_for_everyone()

        # --- Prepare Components ---
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader() if self.eval_dataset else None

        # Sync random states for DataLoader shuffle before prepare
        log_on_all_ranks(self.accelerator, f"Pre-prepare seed sync: Setting manual seed {args.seed}")
        torch.manual_seed(args.seed)

        log_on_all_ranks(self.accelerator, ">>> Preparing model, optimizer, dataloaders with Accelerator...")
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler
        )
        self.model_wrapped = self.model # Store the wrapped model

        # Reset local seed after prepare
        log_on_all_ranks(self.accelerator, f"Post-prepare seed reset: Setting manual seed {self.local_seed}")
        torch.manual_seed(self.local_seed)

        # Update FSDP status after prepare
        self.is_fsdp_enabled = isinstance(self.accelerator.state.fsdp_plugin, FullyShardedDataParallelPlugin)
        if self.is_fsdp_enabled:
            log_on_all_ranks(self.accelerator, f"FSDP is enabled via Accelerate. Model class: {type(self.model)}")
        else:
            log_on_all_ranks(self.accelerator, "FSDP is NOT enabled via Accelerate.")

        log_on_all_ranks(self.accelerator, "RLOOIndirectTrainer initialization finished.")


    def _find_peft_model(self, model_to_search):
        """Finds the underlying PeftModel instance, potentially unwrapping wrappers."""
        MAX_UNWRAP_DEPTH = 5
        peft_model_instance = None
        current_model = model_to_search
        for _ in range(MAX_UNWRAP_DEPTH):
            if isinstance(current_model, PeftModel):
                peft_model_instance = current_model
                break
            # Check common wrapper attributes like 'module' (used by FSDP, DDP)
            if hasattr(current_model, 'module'):
                current_model = current_model.module
            # Add checks for other potential wrappers if necessary
            # elif hasattr(current_model, 'model') and isinstance(current_model.model, PeftModel):
            #     peft_model_instance = current_model.model
            #     break
            else:
                break # Cannot unwrap further
        return peft_model_instance

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size, # Fetch prompts for one rollout "micro-step"
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True, # Necessary for consistent batch sizes
            pin_memory=self.args.dataloader_pin_memory,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> Optional[DataLoader]:
        """Returns the evaluation dataloader."""
        if eval_dataset is None and self.eval_dataset is None:
            return None
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None: # Still could be None
             return None

        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=getattr(self.args, 'dataloader_drop_last', False), # Default to False for eval
            pin_memory=self.args.dataloader_pin_memory,
            num_workers=self.args.dataloader_num_workers,
        )

    def _load_adapter_via_api(self, adapter_path: str, adapter_name: str) -> bool:
        """Dynamically loads or updates an adapter via the vLLM API."""
        if not self.is_world_process_zero():
            return True # Assume success on non-zero ranks, rely on broadcast

        load_url = f"{self.vllm_api_url}/v1/load_lora_adapter"
        payload = {"lora_name": adapter_name, "lora_path": adapter_path}
        headers = {"Content-Type": "application/json"}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log_on_all_ranks(self.accelerator, f"Attempting to load adapter '{adapter_name}' from '{adapter_path}' via API (Attempt {attempt+1})...")
                response = self.api_session.post(load_url, json=payload, headers=headers, timeout=120) # Increased timeout
                response.raise_for_status()
                log_on_all_ranks(self.accelerator, f"Successfully loaded adapter '{adapter_name}' from '{adapter_path}' via API.")
                return True
            except requests.exceptions.RequestException as e:
                log_on_all_ranks(self.accelerator, f"Warning: API call to load adapter failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    log_on_all_ranks(self.accelerator, f"Error: Failed to load adapter '{adapter_name}' via API after {max_retries} attempts.")
                    return False
                time.sleep(5) # Wait before retrying
        return False # Should not be reached

    def _unload_adapter_via_api(self, adapter_name: str) -> bool:
        """Dynamically unloads an adapter via the vLLM API."""
        if not self.is_world_process_zero():
            return True # Assume success

        unload_url = f"{self.vllm_api_url}/v1/unload_lora_adapter"
        payload = {"lora_name": adapter_name}
        headers = {"Content-Type": "application/json"}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log_on_all_ranks(self.accelerator, f"Attempting to unload adapter '{adapter_name}' via API (Attempt {attempt+1})...")
                response = self.api_session.post(unload_url, json=payload, headers=headers, timeout=60)
                if response.status_code == 404:
                    log_on_all_ranks(self.accelerator, f"Warning: API endpoint {unload_url} not found (404). Assuming unload is not supported or needed.")
                    return True # Treat as success if endpoint doesn't exist
                if response.status_code == 400 and "not loaded" in response.text.lower():
                     log_on_all_ranks(self.accelerator, f"Info: Adapter '{adapter_name}' was already not loaded according to API.")
                     return True # Treat as success if already unloaded
                response.raise_for_status() # Raise for other errors (e.g., 500)
                log_on_all_ranks(self.accelerator, f"Successfully unloaded adapter '{adapter_name}' via API.")
                return True
            except requests.exceptions.RequestException as e:
                log_on_all_ranks(self.accelerator, f"Warning: API call to unload adapter failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    log_on_all_ranks(self.accelerator, f"Warning: Failed to unload adapter '{adapter_name}' via API after {max_retries} attempts. Continuing...")
                    return False # Indicate failure but maybe not fatal
                time.sleep(2)
        return False # Should not be reached

    def _generate_via_vllm_api(self, prompts_text: List[str], adapter_name: str, sampling_params: Dict[str, Any]) -> List[str]:
        """Sends prompts to the vLLM API server and returns generated text."""
        if not self.is_world_process_zero():
            # Return dummy list of correct size for other processes
            return [""] * len(prompts_text) * sampling_params.get("n", 1)

        completions_url = f"{self.vllm_api_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        num_prompts = len(prompts_text)
        num_samples_per_prompt = sampling_params.get("n", 1)
        expected_choices = num_prompts * num_samples_per_prompt

        payload = {
            "model": adapter_name, # Use the adapter name as the model identifier for vLLM
            "prompt": prompts_text,
            "n": num_samples_per_prompt,
            "max_tokens": sampling_params.get("max_tokens"),
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "stop": sampling_params.get("stop", []),
            # Add other sampling params if needed (e.g., presence_penalty, frequency_penalty)
        }
        payload = {k: v for k, v in payload.items() if v is not None} # Filter None values

        all_generated_texts = []
        max_retries = 5 # Increase retries for generation
        # Dynamic timeout based on expected workload
        base_timeout = 60
        timeout_per_completion = 10 # Seconds per completion requested
        timeout = base_timeout + expected_choices * timeout_per_completion

        log_on_all_ranks(self.accelerator, f"Sending {num_prompts} prompts (n={num_samples_per_prompt}) to vLLM API ({completions_url}). Timeout: {timeout}s")

        for attempt in range(max_retries):
            try:
                response = self.api_session.post(completions_url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
                response_data = response.json()

                if "choices" not in response_data or len(response_data["choices"]) != expected_choices:
                    log_on_all_ranks(self.accelerator, f"Error: API response missing 'choices' or has unexpected length. Expected {expected_choices}, got {len(response_data.get('choices', []))}")
                    log_on_all_ranks(self.accelerator, f"Response content: {response.text[:500]}...") # Log beginning of response
                    if attempt == max_retries - 1:
                        log_on_all_ranks(self.accelerator, "Error: Final attempt failed. Returning dummy list.")
                        return ["ERROR: FAILED GENERATION"] * expected_choices
                    time.sleep(5 + attempt * 2) # Exponential backoff
                    continue # Retry

                # Assuming vLLM returns in order: [p0_s0, p0_s1, ..., p0_sk-1, p1_s0, ...]
                all_generated_texts = [choice["text"] for choice in response_data["choices"]]
                log_on_all_ranks(self.accelerator, f"Successfully generated {len(all_generated_texts)} completions via API.")
                return all_generated_texts # Success

            except requests.exceptions.Timeout:
                 log_on_all_ranks(self.accelerator, f"Warning: API call for generation timed out (Attempt {attempt+1}/{max_retries}). Retrying...")
                 timeout *= 1.5 # Increase timeout for next retry
            except requests.exceptions.RequestException as e:
                log_on_all_ranks(self.accelerator, f"Warning: API call for generation failed (Attempt {attempt+1}/{max_retries}): {e}")
                # Could check for specific errors (e.g., connection error vs server error)
            except json.JSONDecodeError as e:
                 log_on_all_ranks(self.accelerator, f"Warning: Failed to decode JSON response from API (Attempt {attempt+1}/{max_retries}): {e}")
                 log_on_all_ranks(self.accelerator, f"Response content: {response.text[:500]}...")

            if attempt == max_retries - 1:
                log_on_all_ranks(self.accelerator, f"Error: Failed to generate completions via API after {max_retries} attempts.")
                return ["ERROR: FAILED GENERATION"] * expected_choices # Return dummy list indicating failure
            time.sleep(5 + attempt * 2) # Exponential backoff before retrying non-timeout errors

        return ["ERROR: FAILED GENERATION"] * expected_choices # Should not be reached

    def _call_zmq_reward_server(self, prompts, completions, task_type, task_data, kwargs, evaluate=False) -> List[List[float]]:
        """Sends data to ZMQ server and returns rewards."""
        num_prompts = len(prompts)
        k = len(completions[0]) if num_prompts > 0 and completions and completions[0] else self.args.rloo_k
        if not self.accelerator.is_main_process:
             return [[0.0] * k for _ in range(num_prompts)]

        # --- Dummy Reward Implementation ---
        # Replace this with your actual ZMQ call when ready
        log_on_all_ranks(self.accelerator, "[DUMMY REWARD] Calculating dummy rewards based on length...")
        target_length = 1024.0 # Example target length
        nested_scores = []
        for i, comps_for_prompt in enumerate(completions):
             scores_for_prompt = [] # Initialize list for THIS prompt's k scores
             for j, comp_text in enumerate(comps_for_prompt):
                  score = 0.0 # Default score
                  if "ERROR: FAILED GENERATION" in comp_text:
                      score = -10.0 # Penalize failed generations heavily
                  else:
                    try:
                        comp_len = len(self.processing_class.encode(comp_text, add_special_tokens=False))
                        score = max(0.0, 1.0 - (abs(comp_len - target_length) / target_length))
                    except Exception as e:
                         log_on_all_ranks(self.accelerator, f"Warning: Error encoding completion text during dummy reward calc: {e}")
                         score = -5.0 # Penalize encoding errors less severely?

                  # Append the score for THIS specific completion (j)
                  scores_for_prompt.append(score)

             # After processing all k completions, append the list of k scores
             if len(scores_for_prompt) != k:
                  log_on_all_ranks(self.accelerator, f"Warning: Dummy reward generated {len(scores_for_prompt)} scores for prompt {i}, expected k={k}.")
                  # Pad or truncate if necessary, though ideally should match k
                  while len(scores_for_prompt) < k: scores_for_prompt.append(0.0) # Pad with 0
                  scores_for_prompt = scores_for_prompt[:k] # Truncate
             if evaluate:
                    scores_for_prompt.append(0.0) # Append dummy score for evaluation

             nested_scores.append(scores_for_prompt) # Append the list [score_0, score_1, ..., score_k-1]

        log_on_all_ranks(self.accelerator, f"[DUMMY REWARD] Finished calculating {len(nested_scores)} dummy reward lists, each expected length {k}.")
        return nested_scores
        # --- End Dummy Reward Implementation ---

        # --- ZMQ Implementation (Uncomment and adapt) ---
        # log_on_all_ranks(self.accelerator, f"Sending {len(prompts)} prompts to ZMQ reward server...")
        # try:
        #     message = {
        #         "prompt_texts": prompts,
        #         "completions_texts": completions,
        #         "task_type": task_type,
        #         "task_specific_data": task_data,
        #         "trainer_info": {"class": str(self.__class__.__name__), "rank": self.accelerator.process_index}, # Example info
        #         **kwargs
        #     }
        #     # Ensure JSON serializability
        #     # Might need to convert complex objects (like dicts in task_data for CPT) if necessary
        #     serializable_message = json.loads(json.dumps(message))

        #     self.zmq_socket.send_json(serializable_message)
        #     response = self.zmq_socket.recv_json()
        #     log_on_all_ranks(self.accelerator, "Received response from ZMQ reward server.")

        #     nested_scores_list = response.get("rewards", None)

        #     # Validation
        #     if not isinstance(nested_scores_list, list) or len(nested_scores_list) != len(prompts):
        #         raise ValueError(f"ZMQ reward function returned invalid data. Expected list of {len(prompts)} lists. Got: {type(nested_scores_list)} len {len(nested_scores_list)}")
        #     for i, sublist in enumerate(nested_scores_list):
        #          k = len(completions[i])
        #          if not isinstance(sublist, list) or len(sublist) != k:
        #               raise ValueError(f"ZMQ reward sublist for prompt {i} invalid. Expected list of {k} floats. Got: {type(sublist)} len {len(sublist)}")
        #          # Optional: Check if values are floats? Handled by ZMQ server ideally.

        #     return nested_scores_list

        # except zmq.ZMQError as e:
        #     log_on_all_ranks(self.accelerator, f"ZMQ Error communicating with reward server: {e}")
        #     # Handle error appropriately - maybe return default low scores or raise
        #     raise RuntimeError(f"Failed to get rewards from ZMQ server: {e}") from e
        # except Exception as e:
        #     log_on_all_ranks(self.accelerator, f"Error processing ZMQ reward request/response: {e}")
        #     raise RuntimeError(f"Failed during ZMQ communication: {e}") from e
        # --- End ZMQ Implementation ---


    def train(self):
        """Main training loop following the PPO-style structure."""
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device
        optimizer = self.optimizer
        model = self.model # This is the wrapped model

        log_on_all_ranks(accelerator, "=== Starting Training (PPO-Style RLOO PEFT Adapter with vLLM) ===")
        start_time = time.time()

        # Ensure model is in train mode initially
        model.train()

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

        # --- Initial Adapter Save ---
        # Crucial for the *first* vLLM generation call in the loop
        log_on_all_ranks(accelerator, "Performing initial adapter save before training loop starts...")
        try:
            # Save requires unwrapped model on main process, FSDP handles gathering
            unwrapped_model_init = accelerator.unwrap_model(model)
            peft_model_instance_init = self._find_peft_model(unwrapped_model_init)

            if peft_model_instance_init is None:
                 raise RuntimeError(f"Rank {accelerator.process_index}: Could not find PeftModel instance for initial save.")

            # Define the specific subdirectory for this adapter
            if accelerator.is_main_process:
                os.makedirs(self.adapter_save_path, exist_ok=True)
            accelerator.wait_for_everyone()

            # Call save_pretrained on ALL ranks if FSDP, main process handles writing.
            peft_model_instance_init.save_pretrained(
                self.adapter_save_path, # Save to the specific adapter's subfolder
                selected_adapters=[args.lora_adapter_name],
                safe_serialization=True,
                is_main_process=accelerator.is_main_process
            )
            accelerator.wait_for_everyone() # Ensure save is complete
            log_on_all_ranks(accelerator, f"Initial adapter '{args.lora_adapter_name}' saved to {self.adapter_save_path}")

        except Exception as e:
            log_on_all_ranks(accelerator, f"Error during initial adapter saving: {e}")
            import traceback
            log_on_all_ranks(accelerator, traceback.format_exc())
            raise e

        # --- Initial Evaluation ---
        if args.eval_steps > 0 and self.eval_dataloader is not None:
            self.accelerator.print("Running initial evaluation before training...")
            self.evaluate(global_step=0)
            self.accelerator.print("Initial evaluation finished.")
            model.train() # Ensure back to train mode
        else:
            self.accelerator.print("Skipping initial evaluation.")

        # --- Dataloader Iterator ---
        dataloader = self.train_dataloader
        def repeat_generator():
            while True:
                for batch in dataloader:
                    yield batch
        iter_dataloader = iter(repeat_generator())

        log_on_all_ranks(accelerator, f"  Num Train Examples = {self.train_dataset_len}")
        log_on_all_ranks(accelerator, f"  Num Eval Examples = {len(self.eval_dataset) if self.eval_dataset else 0}")
        log_on_all_ranks(accelerator, f"  Max PPO Steps (self.state.max_steps) = {self.state.max_steps}")
        log_on_all_ranks(accelerator, f"  Num Rollouts (args.num_total_batches) = {args.num_total_batches}")
        log_on_all_ranks(accelerator, f"  Rollout Batch Size (prompts) = {args.batch_size}")
        log_on_all_ranks(accelerator, f"  PPO Mini-Batch Size (total) = {args.mini_batch_size}")
        log_on_all_ranks(accelerator, f"  PPO Mini-Batch Size (local) = {args.local_mini_batch_size}")
        log_on_all_ranks(accelerator, f"  PPO Micro-Batch Size (per device) = {args.per_device_train_batch_size}")
        log_on_all_ranks(accelerator, f"  PPO Grad Accum Steps = {args.gradient_accumulation_steps}")
        log_on_all_ranks(accelerator, f"  PPO Epochs per Rollout = {args.num_ppo_epochs}")
        log_on_all_ranks(accelerator, f"  RLOO K = {args.rloo_k}")
        log_on_all_ranks(accelerator, f"  Logging Steps = {self.state.logging_steps}")
        log_on_all_ranks(accelerator, f"  Save Steps = {self.state.save_steps}")
        log_on_all_ranks(accelerator, f"  Eval Steps = {self.state.eval_steps}")

        # --- Training Loop (Outer: Rollouts / Updates) ---
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            update_start_time = time.time()
            model.train() # Ensure train mode at start of update

            # --- Rollout Phase ---
            rollout_start_time = time.time()
            log_on_all_ranks(accelerator, f"--- Rollout Phase {update}/{args.num_total_batches} ---")

            # Placeholders for the full batch data (accumulated locally on CPU first)
            batch_prompts_text_list_cpu = []
            batch_task_specific_data_list_cpu = []
            batch_query_ids_list_cpu = []
            batch_response_ids_list_cpu = [] # Store processed response IDs
            batch_ref_logprobs_sum_list_cpu = []
            batch_sequence_lengths_list_cpu = []
            batch_processed_responses_text_list_cpu = [] # For reward function

            # Load the adapter saved in the *previous* update for generation
            adapter_path_for_vllm = os.path.join(self.adapter_save_path, args.lora_adapter_name)
            log_on_all_ranks(accelerator, f"[Rollout {update}] Loading adapter '{self.vllm_adapter_name}' from '{adapter_path_for_vllm}' into vLLM...")
            adapter_loaded = self._load_adapter_via_api(adapter_path_for_vllm, self.vllm_adapter_name)
            status_list = [adapter_loaded]
            if accelerator.num_processes > 1:
                broadcast_object_list(status_list, from_process=0)
            adapter_loaded_on_rank = status_list[0]

            if not adapter_loaded_on_rank:
                raise RuntimeError(f"Failed to load adapter '{self.vllm_adapter_name}' into vLLM server (Rank {accelerator.process_index} sync'd failure). Stopping training.")
            log_on_all_ranks(accelerator, f"Rank {accelerator.process_index}: Confirmed adapter '{self.vllm_adapter_name}' loaded successfully for rollout {update}.")

            # Accumulate data over `num_mini_batches` dataloader fetches
            for rollout_step in range(args.num_mini_batches):
                rollout_step_start_time = time.time()
                log_on_all_ranks(accelerator, f"  Rollout micro-step {rollout_step+1}/{args.num_mini_batches}")

                # Fetch a batch of unique prompts (size: local_dataloader_batch_size)
                raw_batch = next(iter_dataloader)
                local_unique_prompts_ids = raw_batch['input_ids'].to(device)
                local_unique_prompts_text = raw_batch['prompt']
                local_unique_task_data = None
                if self.task_type in ["math", "fewshot"]:
                    local_unique_task_data = raw_batch['target']
                elif self.task_type == "cpt":
                    local_unique_task_data = raw_batch['questions']

                # Repeat prompts k times for RLOO generation
                local_prompts_ids = local_unique_prompts_ids.repeat_interleave(args.rloo_k, dim=0)
                local_prompts_text = [p for p in local_unique_prompts_text for _ in range(args.rloo_k)]
                local_task_data = [d for d in local_unique_task_data for _ in range(args.rloo_k)]

                local_micro_batch_size = len(local_prompts_text) # Should be local_mini_batch_size
                log_on_all_ranks(accelerator, f"    Fetched {len(local_unique_prompts_text)} unique prompts, repeated to {local_micro_batch_size} for k={args.rloo_k}")

                # --- vLLM Generation via API ---
                sampling_params_dict = {
                    "n": 1, # We generate k responses by repeating the prompt k times
                    "max_tokens": args.max_response_length, # Use max_response_length from config
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stop": [self.processing_class.eos_token] if args.stop_token_id == self.processing_class.eos_token_id else [],
                }

                # Gather prompts across devices
                all_prompts_text_gathered = gather_object(local_prompts_text)
                flat_generated_responses_text_global = []
                global_num_prompts_micro = args.mini_batch_size # Total prompts in this micro-step
                expected_global_completions_micro = global_num_prompts_micro * 1 # n=1

                if accelerator.is_main_process:
                    # Basic check, padding/truncating if needed (shouldn't happen with drop_last=True)
                    if len(all_prompts_text_gathered) != global_num_prompts_micro:
                         log_on_all_ranks(accelerator, f"Warning: Gathered prompt list size mismatch in rollout. Got {len(all_prompts_text_gathered)}, expected {global_num_prompts_micro}")
                         while len(all_prompts_text_gathered) < global_num_prompts_micro: all_prompts_text_gathered.append("[PAD]")
                         all_prompts_text_gathered = all_prompts_text_gathered[:global_num_prompts_micro]

                    flat_generated_responses_text_global = self._generate_via_vllm_api(
                        all_prompts_text_gathered,
                        self.vllm_adapter_name,
                        sampling_params_dict
                    )
                    actual_len = len(flat_generated_responses_text_global)
                    if actual_len != expected_global_completions_micro:
                        log_on_all_ranks(accelerator, f"Warning: vLLM returned {actual_len} completions, expected {expected_global_completions_micro}. Padding/Truncating.")
                        while len(flat_generated_responses_text_global) < expected_global_completions_micro: flat_generated_responses_text_global.append("")
                        flat_generated_responses_text_global = flat_generated_responses_text_global[:expected_global_completions_micro]
                else:
                    flat_generated_responses_text_global = [""] * expected_global_completions_micro

                # Broadcast generated text
                object_list_to_broadcast = [flat_generated_responses_text_global]
                broadcast_object_list(object_list_to_broadcast, from_process=0)
                flat_generated_responses_text_global = object_list_to_broadcast[0]
                # log_on_all_ranks(accelerator, f"    Rank {accelerator.process_index}: Synced {len(flat_generated_responses_text_global)} completions.")

                # --- Process Generated Responses Locally ---
                local_start_idx = accelerator.process_index * local_micro_batch_size
                local_end_idx = local_start_idx + local_micro_batch_size
                local_generated_responses_text = flat_generated_responses_text_global[local_start_idx:local_end_idx]

                responses_tokenized = self.processing_class(
                    local_generated_responses_text, padding='longest', truncation=True,
                    max_length=args.max_response_length, return_tensors="pt",
                ).to(device)
                local_responses_ids_raw = responses_tokenized.input_ids

                # Remove potential BOS token
                if self.processing_class.bos_token_id is not None and local_responses_ids_raw.shape[1] > 0:
                    if (local_responses_ids_raw[:, 0] == self.processing_class.bos_token_id).all():
                        local_responses_ids_raw = local_responses_ids_raw[:, 1:]

                # Truncate at stop token
                local_processed_responses_ids = local_responses_ids_raw
                if args.stop_token_id is not None and args.stop_token_id != self.processing_class.pad_token_id:
                    local_processed_responses_ids = truncate_response(
                        args.stop_token_id, self.processing_class.pad_token_id, local_responses_ids_raw
                    )
                # log_on_all_ranks(accelerator, f"    Processed responses shape: {local_processed_responses_ids.shape}")

                # --- Calculate Reference Logprobs (Adapter Disabled, No Gradients) ---
                local_context_length = local_prompts_ids.shape[1]
                local_query_responses_ids = torch.cat([local_prompts_ids, local_processed_responses_ids], dim=1)
                local_query_mask = torch.ones_like(local_prompts_ids, device=device)
                local_resp_attn_mask = (local_processed_responses_ids != self.processing_class.pad_token_id).long()
                local_query_responses_mask = torch.cat([local_query_mask, local_resp_attn_mask], dim=1)

                local_ref_logprobs = None
                adapters_disabled = False
                try:
                    with torch.no_grad():
                        # Find PeftModel within the potentially wrapped model structure
                        # Use accelerator.unwrap_model to handle FSDP/DDP correctly
                        peft_model_instance_ref = self._find_peft_model(accelerator.unwrap_model(model))
                        if peft_model_instance_ref is None:
                             raise RuntimeError(f"Rank {accelerator.process_index}: Could not find PeftModel instance for reference pass.")

                        # log_on_all_ranks(accelerator, "    Disabling adapters for reference pass...")
                        peft_model_instance_ref.disable_adapter_layers()
                        adapters_disabled = True

                        # Use the main model instance (potentially wrapped by FSDP/DDP)
                        ref_outputs = model( # Use self.model (the potentially wrapped one)
                            input_ids=local_query_responses_ids,
                            attention_mask=local_query_responses_mask,
                            use_cache=False,
                            output_hidden_states=False,
                            output_attentions=False,
                        )
                        # log_on_all_ranks(accelerator, "    Ref forward pass completed.")

                        ref_logits = ref_outputs.logits[:, local_context_length - 1 : -1]
                        ref_logits /= (args.temperature + 1e-7)
                        local_ref_logprobs = selective_log_softmax(ref_logits, local_processed_responses_ids)

                        # Mask padding for summation
                        local_sequence_lengths = first_true_indices(local_processed_responses_ids == self.processing_class.pad_token_id) - 1
                        local_padding_mask = torch.arange(local_processed_responses_ids.shape[1], device=device) > local_sequence_lengths.unsqueeze(1)
                        local_ref_logprobs_masked = torch.masked_fill(local_ref_logprobs, local_padding_mask, 0.0) # Use 0 for sum
                        local_ref_logprobs_sum = local_ref_logprobs_masked.sum(1)

                        del ref_outputs, ref_logits, local_ref_logprobs # Free memory
                except Exception as e:
                     log_on_all_ranks(accelerator, f"Rank {accelerator.process_index}: Error during reference logprob calculation: {e}")
                     import traceback
                     log_on_all_ranks(accelerator, traceback.format_exc())
                     # Re-raise after attempting to re-enable adapters
                finally:
                    if adapters_disabled and peft_model_instance_ref is not None:
                        # log_on_all_ranks(accelerator, "    Re-enabling adapters...")
                        peft_model_instance_ref.enable_adapter_layers()
                        # log_on_all_ranks(accelerator, "    Adapters re-enabled.")

                # Decode responses for reward function
                local_processed_responses_text = self.processing_class.batch_decode(
                    local_processed_responses_ids, skip_special_tokens=True
                )

                # --- Store accumulated data on CPU ---
                batch_prompts_text_list_cpu.extend(local_prompts_text)
                batch_task_specific_data_list_cpu.extend(local_task_data)
                batch_query_ids_list_cpu.append(local_prompts_ids.cpu())
                batch_response_ids_list_cpu.append(local_processed_responses_ids.cpu())
                batch_ref_logprobs_sum_list_cpu.append(local_ref_logprobs_sum.cpu())
                batch_sequence_lengths_list_cpu.append(local_sequence_lengths.cpu())
                batch_processed_responses_text_list_cpu.extend(local_processed_responses_text)

                log_on_all_ranks(accelerator, f"    Rollout micro-step {rollout_step+1} finished in {time.time() - rollout_step_start_time:.2f}s.")
                # Micro-step cleanup
                del (raw_batch, local_unique_prompts_ids, local_unique_prompts_text, local_unique_task_data,
                     local_prompts_ids, local_prompts_text, local_task_data, responses_tokenized,
                     local_responses_ids_raw, local_processed_responses_ids, local_query_responses_ids,
                     local_query_mask, local_resp_attn_mask, local_query_responses_mask, local_ref_logprobs_sum,
                     local_sequence_lengths, local_processed_responses_text, local_padding_mask)
                if rollout_step % 4 == 0: # Less frequent cleanup
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

            # --- End Rollout Accumulation Loop ---

            # Unload adapter from vLLM server
            if adapter_loaded_on_rank and accelerator.is_main_process:
                log_on_all_ranks(accelerator, f"[Rollout {update}] Unloading adapter '{self.vllm_adapter_name}' from vLLM...")
                unload_success = self._unload_adapter_via_api(self.vllm_adapter_name)
                if not unload_success:
                     log_on_all_ranks(accelerator, f"Warning: Failed to unload adapter '{self.vllm_adapter_name}' after rollout {update}.")
            accelerator.wait_for_everyone() # Sync after potential unload

            log_on_all_ranks(accelerator, f"--- Rollout Phase {update} completed in {time.time() - rollout_start_time:.2f}s ---")

            # --- Pad tensors in collected lists BEFORE collation ---
            pad_start_time = time.time()
            log_on_all_ranks(accelerator, f"Padding collected tensors before collation...")

            # Determine max sequence lengths across all collected batches
            max_prompt_len = 0
            if batch_query_ids_list_cpu:
                 max_prompt_len = max(t.shape[1] for t in batch_query_ids_list_cpu)

            max_resp_len = 0
            if batch_response_ids_list_cpu:
                 max_resp_len = max(t.shape[1] for t in batch_response_ids_list_cpu)

            log_on_all_ranks(accelerator, f"Max prompt len: {max_prompt_len}, Max response len: {max_resp_len}")

            pad_token_id = self.processing_class.pad_token_id
            if pad_token_id is None:
                 # Fallback if tokenizer somehow has no pad token (shouldn't happen with check)
                 pad_token_id = self.processing_class.eos_token_id
                 log_on_all_ranks(accelerator, f"Warning: Using EOS token ({pad_token_id}) for padding.")

            padded_query_ids_list = []
            for query_ids in batch_query_ids_list_cpu:
                current_len = query_ids.shape[1]
                if current_len < max_prompt_len:
                    pad_len = max_prompt_len - current_len
                    # Pad LEFT for prompts (input_ids for decoder)
                    padded_query_ids = torch.nn.functional.pad(
                        query_ids, (pad_len, 0), mode='constant', value=pad_token_id
                    )
                    padded_query_ids_list.append(padded_query_ids)
                else:
                    padded_query_ids_list.append(query_ids)

            padded_response_ids_list = []
            for resp_ids in batch_response_ids_list_cpu:
                current_len = resp_ids.shape[1]
                if current_len < max_resp_len:
                    pad_len = max_resp_len - current_len
                    # Pad RIGHT for responses (labels/generated sequence)
                    padded_response_ids = torch.nn.functional.pad(
                        resp_ids, (0, pad_len), mode='constant', value=pad_token_id
                    )
                    padded_response_ids_list.append(padded_response_ids)
                else:
                    padded_response_ids_list.append(resp_ids)

            log_on_all_ranks(accelerator, f"Padding finished in {time.time() - pad_start_time:.2f}s.")
            # --- End Padding Logic ---


            # --- Collate Full Batch Data (CPU) ---
            try:
                full_batch_prompts_text = batch_prompts_text_list_cpu
                full_batch_task_data = batch_task_specific_data_list_cpu
                # Use the PADDED lists for concatenation
                full_batch_query_ids = torch.cat(padded_query_ids_list, dim=0)
                full_batch_response_ids = torch.cat(padded_response_ids_list, dim=0)
                # These lists contain 1D tensors or Python lists, no padding needed for dim 1
                full_batch_ref_logprobs_sum = torch.cat(batch_ref_logprobs_sum_list_cpu, dim=0)
                full_batch_seq_lengths = torch.cat(batch_sequence_lengths_list_cpu, dim=0)
                full_batch_responses_text = batch_processed_responses_text_list_cpu

                # Basic validation
                expected_total_samples = args.local_batch_size # Samples per device in the full rollout
                # Check shapes after padding and concatenation
                if full_batch_query_ids.shape[0] != expected_total_samples or \
                   len(full_batch_prompts_text) != expected_total_samples:
                     log_on_all_ranks(accelerator, f"Warning: Collated batch size check. Query IDs shape: {full_batch_query_ids.shape}, Expected samples: {expected_total_samples}, Num prompts text: {len(full_batch_prompts_text)}")
                     # Add more detailed checks if needed
                     # raise ValueError(f"Rank {accelerator.process_index}: Collated batch size mismatch after padding.")

            except Exception as e:
                 log_on_all_ranks(accelerator, f"Error during full batch collation: {e}")
                 raise e # Re-raise the exception after logging
            finally:
                 # Clear intermediate lists (including original unpadded lists)
                 del (batch_prompts_text_list_cpu, batch_task_specific_data_list_cpu,
                      batch_query_ids_list_cpu, batch_response_ids_list_cpu,
                      batch_ref_logprobs_sum_list_cpu, batch_sequence_lengths_list_cpu,
                      batch_processed_responses_text_list_cpu,
                      padded_query_ids_list, padded_response_ids_list) # Clear padded lists too
                 gc.collect()

            # --- Reward Calculation Phase ---
            reward_calc_start_time = time.time()
            log_on_all_ranks(accelerator, f"--- Reward Calculation Phase {update} ---")

            # Prepare data for reward function (gather all text data to main process)
            all_gathered_prompts = gather_object(full_batch_prompts_text)
            all_gathered_responses = gather_object(full_batch_responses_text)
            all_gathered_task_data = gather_object(full_batch_task_data)

            nested_scores_list_global = None # Will hold result on rank 0
            if accelerator.is_main_process:
                 total_rollout_samples_global = len(all_gathered_prompts)
                 log_on_all_ranks(accelerator, f"Rank 0: Gathered {total_rollout_samples_global} samples for reward calculation.")
                 if total_rollout_samples_global != args.batch_size:
                      log_on_all_ranks(accelerator, f"Warning: Gathered total samples {total_rollout_samples_global} != expected batch_size {args.batch_size}")

                 if total_rollout_samples_global > 0:
                    # Reshape flat responses into nested list for reward fn:
                    # [[p0_s0, p0_s1,...], [p1_s0, p1_s1,...], ...]
                    # Need to know how many unique prompts there were globally
                    num_unique_prompts_global = total_rollout_samples_global // args.rloo_k
                    nested_responses_for_reward = []
                    nested_task_data_for_reward = [] # Need unique task data per prompt
                    unique_prompts_for_reward = []

                    if total_rollout_samples_global % args.rloo_k != 0:
                        log_on_all_ranks(accelerator, f"Warning: Total gathered samples {total_rollout_samples_global} not divisible by k={args.rloo_k}. Reward reshaping might be incorrect.")
                        # Attempt graceful handling or raise error? For now, log warning.
                        # Adjust num_unique_prompts_global if needed, though it indicates upstream issue.
                        num_unique_prompts_global = total_rollout_samples_global // args.rloo_k


                    current_idx = 0
                    for i in range(num_unique_prompts_global):
                        start = i * args.rloo_k
                        end = start + args.rloo_k
                        if start < len(all_gathered_responses) and end <= len(all_gathered_responses):
                            nested_responses_for_reward.append(all_gathered_responses[start:end])
                            # Assume task data and prompts also repeat k times in the gathered list
                            if start < len(all_gathered_task_data):
                                nested_task_data_for_reward.append(all_gathered_task_data[start]) # Take the first instance for the unique prompt
                            else:
                                nested_task_data_for_reward.append(None) # Or some placeholder

                            if start < len(all_gathered_prompts):
                                 unique_prompts_for_reward.append(all_gathered_prompts[start])
                            else:
                                 unique_prompts_for_reward.append("[PAD PROMPT]")
                        else:
                            log_on_all_ranks(accelerator, f"Warning: Index out of bounds during reward reshaping ({start}-{end} vs len {len(all_gathered_responses)}). Skipping prompt {i}.")


                    # Call reward function (ZMQ or dummy)
                    nested_scores_list_global = self._call_zmq_reward_server(
                        prompts=unique_prompts_for_reward,
                        completions=nested_responses_for_reward,
                        task_type=self.task_type,
                        task_data=nested_task_data_for_reward,
                        kwargs=self.reward_fn_kwargs
                    )
                 else:
                     nested_scores_list_global = [] # No samples

            # Broadcast/Scatter results back
            # Rank 0 has nested_scores_list_global [[s0_0,..s0_k], [s1_0,..s1_k], ...]
            # We need to flatten it and distribute the correct portion to each rank.
            flat_scores_global_cpu = None
            if accelerator.is_main_process:
                flat_scores_global_cpu = torch.tensor(
                    [score for prompt_scores in nested_scores_list_global for score in prompt_scores],
                    dtype=torch.float
                )
                score_stats = safe_stats(flat_scores_global_cpu, "raw_scores_global")
                log_on_all_ranks(accelerator, f"[Reward Stats] Raw Global Scores (CPU): {score_stats}")

                # --- Validate size on main process ---
                if flat_scores_global_cpu.numel() != args.batch_size:
                    log_on_all_ranks(accelerator, f"Error: Global scores calculated ({flat_scores_global_cpu.numel()}) != expected global batch size ({args.batch_size}). Check reward function output. Adjusting tensor size.")
                    pad_value = float('nan') # Or other indicator
                    if flat_scores_global_cpu.numel() < args.batch_size:
                        padding = torch.full((args.batch_size - flat_scores_global_cpu.numel(),), pad_value, dtype=torch.float)
                        flat_scores_global_cpu = torch.cat([flat_scores_global_cpu, padding], dim=0)
                    else:
                        flat_scores_global_cpu = flat_scores_global_cpu[:args.batch_size]
                    log_on_all_ranks(accelerator, f"Adjusted global scores tensor shape to {flat_scores_global_cpu.shape}")

            # --- Broadcast the full scores tensor using accelerate.utils.broadcast ---
            # 1. Create the tensor buffer ON THE DEVICE on all processes.
            #    It needs to be the correct size (global batch size).
            global_scores_device_buffer = torch.zeros(args.batch_size, dtype=torch.float, device=accelerator.device)

            # 2. On main process, copy the data into the buffer.
            if accelerator.is_main_process:
                if flat_scores_global_cpu is not None:
                    global_scores_device_buffer.copy_(flat_scores_global_cpu.to(accelerator.device))
                # else: buffer remains zeros

            # 3. Broadcast the tensor *data* from rank 0's buffer to all other ranks' buffers.
            #    The `broadcast` function modifies the tensor in-place on the receiving ranks.
            log_on_all_ranks(accelerator, f"Broadcasting global scores tensor (shape: {global_scores_device_buffer.shape}) from rank 0...")
            broadcast(global_scores_device_buffer, from_process=0)
            log_on_all_ranks(accelerator, f"Broadcast finished.")
            # Now, global_scores_device_buffer on *all* ranks contains the full tensor data from rank 0.

            # 4. Slice the received global tensor locally to get this rank's scores
            local_start_index = accelerator.process_index * args.local_batch_size
            local_end_index = local_start_index + args.local_batch_size
            if local_end_index > args.batch_size:
                 log_on_all_ranks(accelerator, f"Warning: Calculated end index {local_end_index} > global batch size {args.batch_size}. Clamping.")
                 local_end_index = args.batch_size

            if local_start_index >= local_end_index:
                 log_on_all_ranks(accelerator, f"Warning: Invalid slice indices [{local_start_index}:{local_end_index}]. Using empty tensor.")
                 local_scores_device = torch.tensor([], dtype=torch.float, device=accelerator.device)
            else:
                 local_scores_device = global_scores_device_buffer[local_start_index:local_end_index].clone() # Use clone() to get a distinct tensor

            log_on_all_ranks(accelerator, f"Sliced local scores tensor (shape: {local_scores_device.shape})")

            # --- Optional: Clear the large global buffer ---
            del global_scores_device_buffer
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # --- Move local scores to CPU for EOS penalty application ---
            local_scores_cpu = local_scores_device.cpu()

            # --- Apply EOS Penalty (Locally on CPU) ---
            # Ensure full_batch_response_ids and lengths are on CPU for this
            contain_eos_token = torch.any(
                (full_batch_response_ids == self.processing_class.eos_token_id) &
                (torch.arange(full_batch_response_ids.shape[1]) <= full_batch_seq_lengths.unsqueeze(1)),
                dim=1
            )

            if local_scores_cpu.numel() == contain_eos_token.numel():
                if hasattr(args, 'missing_eos_penalty') and args.missing_eos_penalty is not None:
                    local_scores_cpu[~contain_eos_token] -= args.missing_eos_penalty
                    score_pen_stats = safe_stats(local_scores_cpu, "scores_w_penalty_local")
                    log_on_all_ranks(accelerator, f"[Reward Stats] Local Scores w/ Penalty: {score_pen_stats}")
            else:
                 log_on_all_ranks(accelerator, f"Warning: Size mismatch between local scores ({local_scores_cpu.numel()}) and EOS mask ({contain_eos_token.numel()}). Skipping EOS penalty.")


            log_on_all_ranks(accelerator, f"--- Reward Calculation Phase {update} finished in {time.time() - reward_calc_start_time:.2f}s ---")

            # --- Advantage Calculation Phase ---
            advantage_calc_start_time = time.time()
            log_on_all_ranks(accelerator, f"--- Advantage Calculation Phase {update} ---")

            # Move necessary tensors to device for calculation
            local_scores = local_scores_cpu.to(device) # Use the penalized scores
            # Ensure ref logprobs sum corresponds to the local batch data
            local_ref_logprobs_sum = full_batch_ref_logprobs_sum.to(device)

            # Rewards for advantage calculation (Score component only)
            # Normalize scores if needed (using global stats)
            current_scores_for_adv = local_scores.clone()
            if args.normalize_reward:
                gathered_scores_all = accelerator.gather(current_scores_for_adv).float()
                valid_scores = gathered_scores_all[~torch.isinf(gathered_scores_all) & ~torch.isnan(gathered_scores_all)]
                mean_score, std_score = torch.nan, torch.nan
                if valid_scores.numel() > 1:
                    mean_score = valid_scores.mean()
                    std_score = valid_scores.std()
                    current_scores_for_adv = (current_scores_for_adv - mean_score) / (std_score + 1e-8)
                    current_scores_for_adv = torch.clamp(current_scores_for_adv, -args.reward_clip_range, args.reward_clip_range)
                elif valid_scores.numel() == 1:
                    mean_score = valid_scores.mean(); std_score = 0.0
                    current_scores_for_adv = current_scores_for_adv - mean_score # Center only
                    log_on_all_ranks(accelerator, "Warning: Only one valid reward score found. Centering rewards.")
                else: # valid_scores.numel() == 0
                    log_on_all_ranks(accelerator, "Warning: Could not normalize rewards: No valid values gathered.")
                adv_score_stats = safe_stats(current_scores_for_adv, "scores_for_adv_local")
                log_on_all_ranks(accelerator, f"[Advantage Stats] Scores (Normalized: {args.normalize_reward}, Mean: {mean_score:.4f}, Std: {std_score:.4f}): {adv_score_stats}")

            # Calculate RLOO baseline and advantages using the potentially normalized scores
            num_unique_prompts_local = args.local_batch_size // args.rloo_k
            if args.local_batch_size % args.rloo_k != 0:
                raise ValueError(f"Internal error: local_batch_size {args.local_batch_size} not divisible by k {args.rloo_k}")

            try:
                # Reshape assumes order [p0_s0...p0_sk-1, p1_s0...p1_sk-1, ...]
                scores_grouped = current_scores_for_adv.reshape(num_unique_prompts_local, args.rloo_k).transpose(0, 1) # Shape: (k, num_prompts_local)
            except Exception as e:
                raise RuntimeError(f"Failed reshape scores_for_adv. Shape: {current_scores_for_adv.shape}, k: {args.rloo_k}, num_prompts_local: {num_unique_prompts_local}. Err: {e}")

            if args.rloo_k > 1:
                baseline = (scores_grouped.sum(0, keepdim=True) - scores_grouped) / (args.rloo_k - 1)
            else:
                baseline = torch.zeros_like(scores_grouped) # No baseline if k=1

            advantages_grouped = scores_grouped - baseline
            local_advantages = advantages_grouped.transpose(0, 1).flatten() # Shape: (local_batch_size,)
            adv_before_norm_stats = safe_stats(local_advantages, "adv_raw_local")

            # Normalize advantages if needed (using global stats)
            if args.normalize_advantage:
                gathered_advantages_all = accelerator.gather(local_advantages).float()
                valid_advantages = gathered_advantages_all[~torch.isnan(gathered_advantages_all) & ~torch.isinf(gathered_advantages_all)]
                mean_adv, std_adv = torch.nan, torch.nan
                if valid_advantages.numel() > 1:
                    mean_adv = valid_advantages.mean()
                    std_adv = valid_advantages.std()
                    local_advantages = (local_advantages - mean_adv) / (std_adv + 1e-8)
                elif valid_advantages.numel() > 0:
                    mean_adv = valid_advantages.mean(); std_adv = 0.0
                    local_advantages = local_advantages - mean_adv
                    log_on_all_ranks(accelerator, "Warning: Centering advantages but not scaling (std=0 or N=1).")
                else:
                    log_on_all_ranks(accelerator, "Warning: Could not normalize advantages: No valid values found.")
                adv_after_norm_stats = safe_stats(local_advantages, "adv_norm_local")
                log_on_all_ranks(accelerator, f"[Advantage Stats] Raw Local Adv: {adv_before_norm_stats}")
                log_on_all_ranks(accelerator, f"[Advantage Stats] Final Local Adv (Normalized: {args.normalize_advantage}, Mean: {mean_adv:.4f}, Std: {std_adv:.4f}): {adv_after_norm_stats}")

            log_on_all_ranks(accelerator, f"--- Advantage Calculation Phase {update} finished in {time.time() - advantage_calc_start_time:.2f}s ---")

            # Move all necessary batch data to device for PPO loop
            # Keep tensors needed for PPO loop on device
            batch_query_ids_device = full_batch_query_ids.to(device)
            batch_response_ids_device = full_batch_response_ids.to(device)
            batch_ref_logprobs_sum_device = local_ref_logprobs_sum # Already on device
            batch_advantages_device = local_advantages # Already on device
            batch_scores_device = local_scores # Already on device
            batch_seq_lengths_device = full_batch_seq_lengths.to(device)

            # Clear CPU tensors that are now on device
            del (full_batch_query_ids, full_batch_response_ids, full_batch_ref_logprobs_sum,
                 full_batch_seq_lengths, local_scores_cpu, local_scores, local_ref_logprobs_sum,
                 local_advantages, current_scores_for_adv)
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()


            # --- PPO Optimization Phase ---
            ppo_start_time = time.time()
            log_on_all_ranks(accelerator, f"--- PPO Optimization Phase {update} ({args.num_ppo_epochs} epochs) ---")

            # Stats tracking across PPO epochs/minibatches
            stats_shape = (args.num_ppo_epochs, args.num_mini_batches) # Track per minibatch update
            approxkl_stats = torch.zeros(stats_shape, device=device)
            pg_loss_stats = torch.zeros(stats_shape, device=device)
            entropy_stats = torch.zeros(stats_shape, device=device)
            # Add other stats if needed (e.g., clipfrac if using PPO loss)

            # PPO Epoch Loop
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                ppo_epoch_start_time = time.time()
                # Shuffle indices for the entire local batch
                local_indices = torch.randperm(args.local_batch_size, device=device)

                # Mini-batch Loop
                for minibatch_idx, mini_batch_start in enumerate(range(0, args.local_batch_size, args.local_mini_batch_size)):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_indices = local_indices[mini_batch_start:mini_batch_end]

                    # Accumulation loop within mini-batch (if per_device_train_batch_size < local_mini_batch_size)
                    # Handled by accelerator.accumulate context
                    with accelerator.accumulate(model):
                        # Slice the mini-batch data using indices
                        mb_query_ids = batch_query_ids_device[mini_batch_indices]
                        mb_response_ids = batch_response_ids_device[mini_batch_indices]
                        mb_ref_logprobs_sum = batch_ref_logprobs_sum_device[mini_batch_indices]
                        mb_advantages = batch_advantages_device[mini_batch_indices]
                        mb_seq_lengths = batch_seq_lengths_device[mini_batch_indices]

                        # --- Policy Forward Pass (Requires Grad) ---
                        mb_context_length = mb_query_ids.shape[1]
                        mb_query_responses_ids = torch.cat([mb_query_ids, mb_response_ids], dim=1)
                        mb_query_mask = torch.ones_like(mb_query_ids)
                        mb_response_idxs = torch.arange(mb_response_ids.shape[1], device=device).repeat(mb_response_ids.shape[0], 1)
                        mb_resp_attn_mask = (mb_response_idxs <= mb_seq_lengths.unsqueeze(1)).long()
                        mb_query_responses_mask = torch.cat([mb_query_mask, mb_resp_attn_mask], dim=1)

                        # Ensure model is in train mode for dropout/batchnorm etc. (though dropout is disabled)
                        model.train()
                        outputs = model(
                            input_ids=mb_query_responses_ids,
                            attention_mask=mb_query_responses_mask,
                            use_cache=False,
                        )
                        logits = outputs.logits[:, mb_context_length - 1 : -1]
                        logits /= (args.temperature + 1e-7)

                        # --- Calculate New Logprobs and KL ---
                        new_logprobs_token = selective_log_softmax(logits, mb_response_ids)

                        # Apply padding mask (use > to exclude padding from sum)
                        mb_padding_mask = mb_response_idxs > mb_seq_lengths.unsqueeze(1)
                        new_logprobs_token_masked = torch.masked_fill(new_logprobs_token, mb_padding_mask, 0.0)
                        new_logprobs_sum = new_logprobs_token_masked.sum(1) # Sum over sequence length

                        kl_sum = new_logprobs_sum - mb_ref_logprobs_sum # KL per sequence

                        # --- RLOO Loss Calculation ---
                        # Loss uses advantages calculated *before* PPO loop (based on scores only)
                        pg_loss = (-mb_advantages.detach() * new_logprobs_sum).mean()
                        loss = pg_loss

                        # --- Backward Pass ---
                        # accelerator.backward handles scaling/syncing
                        accelerator.backward(loss)

                        # --- Optimizer Step (Handled outside accumulate context by Accelerator) ---
                        # Store stats for logging (within accumulate context, before potential optimizer step)
                        with torch.no_grad():
                            # Calculate approx KL between old (ref) and new policy logprobs for monitoring
                            # Note: This is KL(new || ref), not KL(new || old_policy) as in PPO
                            approxkl = (kl_sum).mean() # Monitor KL divergence from reference
                            # Calculate entropy
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy_token = -torch.sum(prob_dist * torch.log(prob_dist + 1e-9), dim=-1)
                            masked_entropy = entropy_token.masked_fill(mb_padding_mask, 0.0)
                            num_non_pad_tokens = (~mb_padding_mask).sum()
                            mean_entropy = masked_entropy.sum() / num_non_pad_tokens if num_non_pad_tokens > 0 else torch.tensor(0.0, device=device)

                            # Store stats for this minibatch update
                            if accelerator.is_main_process: # Store only on main to avoid large tensors on all ranks
                                approxkl_stats[ppo_epoch_idx, minibatch_idx] = approxkl.item()
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx] = pg_loss.item()
                                entropy_stats[ppo_epoch_idx, minibatch_idx] = mean_entropy.item()

                    # --- End Accumulate Context ---
                    # Accelerator manages optimizer step, zero_grad based on gradient_accumulation_steps

                    if accelerator.sync_gradients:
                         # This block executes only when optimizer.step() is called by Accelerator
                         self.state.global_step += 1 # Increment global step (PPO step)

                         # --- Logging, Checkpointing, Eval Checks ---
                         # Callbacks invoked AFTER optimizer step and scheduler step (handled by Trainer loop)
                         self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                         # Check if logging is due
                         if self.state.global_step % self.state.logging_steps == 0:
                             # Logging happens *after* the PPO loop finishes for the update/rollout
                             pass # Defer logging calculation

                         # Check if evaluation is due
                         if self.state.eval_steps and self.state.global_step % self.state.eval_steps == 0:
                             log_on_all_ranks(accelerator, f"--- Triggering Evaluation at PPO Step {self.state.global_step} ---")
                             self.evaluate(global_step=self.state.global_step)
                             model.train() # Ensure model is back in train mode

                         # Check if saving is due
                         if self.control.should_save: # Use TrainerControl flag
                              self._save_checkpoint() # Saves full state via Accelerator
                              self.control = self.callback_handler.on_save(self.args, self.state, self.control) # Call on_save callback


                    # --- Mini-batch Cleanup ---
                    del (mb_query_ids, mb_response_ids, mb_ref_logprobs_sum, mb_advantages, mb_seq_lengths,
                         mb_query_responses_ids, mb_query_mask, mb_resp_attn_mask, mb_query_responses_mask,
                         outputs, logits, new_logprobs_token, new_logprobs_token_masked, new_logprobs_sum,
                         kl_sum, pg_loss, loss, approxkl, prob_dist, entropy_token, masked_entropy, mean_entropy)
                    # if minibatch_idx % 8 == 0: # Less frequent
                    #     gc.collect()
                    #     if torch.cuda.is_available(): torch.cuda.empty_cache()

                # --- End Mini-batch Loop ---
                log_on_all_ranks(accelerator, f"    PPO Epoch {ppo_epoch_idx+1}/{args.num_ppo_epochs} finished in {time.time() - ppo_epoch_start_time:.2f}s.")

            # --- End PPO Epoch Loop ---
            log_on_all_ranks(accelerator, f"--- PPO Optimization Phase {update} finished in {time.time() - ppo_start_time:.2f}s ---")


            # --- Logging Phase (after PPO epochs for the update) ---
            log_start_time = time.time()
            # Gather necessary stats from the PPO loops and the rollout phase
            # We need means over the PPO epochs/minibatches for loss, KL, entropy
            # We also need metrics from the rollout (rewards, scores)

            # Calculate mean stats from the PPO loop (use main process stored stats)
            mean_approxkl = approxkl_stats.mean().item() if accelerator.is_main_process else 0.0
            mean_pg_loss = pg_loss_stats.mean().item() if accelerator.is_main_process else 0.0
            mean_entropy = entropy_stats.mean().item() if accelerator.is_main_process else 0.0

            # Gather rollout metrics (need to re-gather as they were potentially cleared)
            # Or better: calculate means locally and reduce
            rollout_metrics = {}
            with torch.no_grad():
                # Use the batch tensors that were kept on device
                mean_score = accelerator.reduce(batch_scores_device.mean(), reduction="mean").item()
                # Calculate RLHF reward = score - kl_coef * kl (Need KL for this)
                # KL was calculated inside PPO loop, need to recalculate or store it.
                # Let's just log the components separately for now.
                mean_adv = accelerator.reduce(batch_advantages_device.mean(), reduction="mean").item()
                mean_ref_logp = accelerator.reduce(batch_ref_logprobs_sum_device.mean(), reduction="mean").item()
                mean_seq_len = accelerator.reduce(batch_seq_lengths_device.float().mean(), reduction="mean").item()

                rollout_metrics = {
                    "rollout/scores_mean": mean_score,
                    "rollout/advantages_mean": mean_adv,
                    "rollout/ref_logprobs_sum_mean": mean_ref_logp,
                    "rollout/response_length_mean": mean_seq_len,
                }

            # Combine metrics for logging
            metrics = {}
            metrics["train/episode"] = update * args.batch_size # Track total episodes processed
            metrics["train/epoch"] = (update * args.batch_size) / self.train_dataset_len if self.train_dataset_len > 0 else 0
            metrics["train/loss_policy"] = mean_pg_loss
            metrics["train/policy_entropy"] = mean_entropy
            metrics["train/kl_policy_vs_ref"] = mean_approxkl # Approx KL(new_policy || ref_policy)
            metrics.update(rollout_metrics)
            metrics["train/lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["train/update"] = update # Log the outer loop step

            # Log the metrics
            if accelerator.is_main_process:
                 self.log(metrics) # Use Trainer's log method

            # --- LR Scheduler Step ---
            # Step the scheduler once per update/rollout
            self.lr_scheduler.step()

            # --- Save Adapter Checkpoint ---
            # Save the adapter state *after* the PPO optimization phase for this update.
            # This state will be loaded by vLLM for the *next* rollout.
            save_adapter_start_time = time.time()
            log_on_all_ranks(accelerator, f"--- Saving Adapter Checkpoint after Update {update} ---")
            try:
                unwrapped_model_save = accelerator.unwrap_model(model)
                peft_model_instance_save = self._find_peft_model(unwrapped_model_save)
                if peft_model_instance_save is None:
                    raise RuntimeError(f"Rank {accelerator.process_index}: Could not find PeftModel instance for saving after update {update}.")

                if accelerator.is_main_process:
                    os.makedirs(self.adapter_save_path, exist_ok=True)
                accelerator.wait_for_everyone()

                peft_model_instance_save.save_pretrained(
                    self.adapter_save_path,
                    selected_adapters=[args.lora_adapter_name],
                    safe_serialization=True,
                    is_main_process=accelerator.is_main_process
                )
                accelerator.wait_for_everyone() # Ensure save is complete
                log_on_all_ranks(accelerator, f"Adapter saved to {self.adapter_save_path} in {time.time() - save_adapter_start_time:.2f}s.")

            except Exception as e:
                log_on_all_ranks(accelerator, f"Error during adapter saving after update {update}: {e}")
                raise e

            # --- Update Cleanup ---
            del (batch_query_ids_device, batch_response_ids_device, batch_ref_logprobs_sum_device,
                 batch_advantages_device, batch_scores_device, batch_seq_lengths_device)
            # Clear PPO stats tensors
            del approxkl_stats, pg_loss_stats, entropy_stats
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            log_on_all_ranks(accelerator, f"--- Update {update}/{args.num_total_batches} finished in {time.time() - update_start_time:.2f}s ---")

            if self.control.should_training_stop:
                 log_on_all_ranks(accelerator, "Training stopped externally.")
                 break

        # --- End of Training Loop ---
        end_time = time.time()
        log_on_all_ranks(accelerator, f"Total training time: {end_time - start_time:.2f}s")

        # --- Final Evaluation ---
        if args.eval_steps > 0 and self.eval_dataloader is not None:
            self.accelerator.print("Running final evaluation after training...")
            self.evaluate(global_step=self.state.global_step) # Use final PPO step count
            self.accelerator.print("Final evaluation finished.")
        else:
            self.accelerator.print("Skipping final evaluation.")

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        accelerator.wait_for_everyone()
        log_on_all_ranks(accelerator, "=== Finished Training ===")

    def _save_checkpoint(self, trial=None, metrics=None):
        """Saves the training state using Accelerator for FSDP compatibility."""
        # Use the global_step from the Trainer state (tracks PPO steps)
        save_path = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")

        if self.is_world_process_zero():
            os.makedirs(save_path, exist_ok=True)
            log_on_all_ranks(self.accelerator, f"Rank 0: Preparing to save checkpoint to {save_path}")
        self.accelerator.wait_for_everyone()

        try:
            log_on_all_ranks(self.accelerator, f"Rank {self.accelerator.process_index}: Calling accelerator.save_state to {save_path}")
            # save_state saves model (PEFT included), optimizer, scheduler, RNG states
            self.accelerator.save_state(save_path)
            self.accelerator.wait_for_everyone()

            if self.is_world_process_zero():
                log_on_all_ranks(self.accelerator, f"Rank 0: Successfully saved checkpoint state: {save_path}")
                # Save tokenizer and trainer state separately
                self.processing_class.save_pretrained(save_path)
                self.state.save_to_json(os.path.join(save_path, "trainer_state.json"))
                # Save the adapter itself (redundant if save_state worked, but good practice)
                try:
                     unwrapped_model = self.accelerator.unwrap_model(self.model)
                     peft_model = self._find_peft_model(unwrapped_model)
                     if peft_model:
                          adapter_subdir = os.path.join(save_path, "adapter_model")
                          os.makedirs(adapter_subdir, exist_ok=True)
                          peft_model.save_pretrained(
                               adapter_subdir,
                               selected_adapters=[self.args.lora_adapter_name],
                               safe_serialization=True,
                               is_main_process=True # Only rank 0 writes files here
                          )
                          log_on_all_ranks(self.accelerator, f"Rank 0: Saved adapter state separately to {adapter_subdir}")
                except Exception as e:
                     log_on_all_ranks(self.accelerator, f"Rank 0: Warning - could not save adapter separately during checkpoint: {e}")


        except Exception as e:
            log_on_all_ranks(self.accelerator, f"Rank {self.accelerator.process_index}: Error during accelerator.save_state: {e}")
            raise e

        self.accelerator.wait_for_everyone()

    def evaluate(self, global_step: Optional[int] = None):
        """
        Run evaluation using the latest adapter state via vLLM.
        """
        if self.eval_dataloader is None:
            log_on_all_ranks(self.accelerator, "No evaluation dataloader found, skipping evaluation.")
            return {} # Return empty metrics dict

        args = self.args
        accelerator = self.accelerator
        model = self.model # Wrapped model

        log_on_all_ranks(accelerator, f"--- Starting Evaluation at Global Step {global_step} ---")
        eval_start_time = time.time()

        # Ensure model is in eval mode (important for FSDP/DDP sync)
        model.eval()

        # --- Adapter Handling ---
        adapter_path_for_eval = os.path.join(self.adapter_save_path, args.lora_adapter_name)
        adapter_loaded = False
        eval_metrics = {} # Store metrics here

        try:
            log_on_all_ranks(accelerator, f"Loading adapter '{self.vllm_adapter_name}' from {adapter_path_for_eval} for evaluation...")
            adapter_loaded = self._load_adapter_via_api(adapter_path_for_eval, self.vllm_adapter_name)
            status_list = [adapter_loaded]
            if accelerator.num_processes > 1:
                broadcast_object_list(status_list, from_process=0)
            adapter_loaded = status_list[0]

            if not adapter_loaded:
                log_on_all_ranks(accelerator, f"Warning: Failed to load adapter '{self.vllm_adapter_name}' for evaluation. Skipping.")
                # Restore model state and return
                model.train()
                accelerator.wait_for_everyone()
                return eval_metrics # Return empty

            log_on_all_ranks(accelerator, f"Adapter '{self.vllm_adapter_name}' loaded successfully for evaluation.")

            # Lists to store results locally before gathering
            local_prompts_text_list_cpu = []
            local_completions_text_list_cpu = []
            local_task_specific_data_list_cpu = []

            # --- Evaluation Loop (No Gradients) ---
            with torch.no_grad():
                for step, batch in enumerate(self.eval_dataloader):
                    local_prompts_text = batch['prompt']
                    local_task_specific_data = None
                    if self.task_type in ["math", "fewshot"]:
                        local_task_specific_data = batch['target']
                    elif self.task_type == "cpt":
                        local_task_specific_data = batch['questions']
                    else:
                        raise ValueError(f"Invalid task type '{self.task_type}' during evaluation.")

                    local_bs = len(local_prompts_text)
                    if local_bs == 0: continue

                    # --- Generate ONE completion per prompt via vLLM API ---
                    eval_sampling_params_dict = {
                        "n": 1,
                        "max_tokens": args.max_response_length, # Use response length
                        "temperature": args.temperature, # Can be 0.0 for deterministic eval if desired
                        "top_p": args.top_p, # Or 1.0 for deterministic
                        "stop": [self.processing_class.eos_token] if args.stop_token_id == self.processing_class.eos_token_id else [],
                    }

                    # Gather prompts across devices
                    all_prompts_text_gathered = gather_object(local_prompts_text)
                    flat_generated_responses_text_global = []
                    # Calculate global size for this eval step
                    global_num_prompts = len(all_prompts_text_gathered) # Actual gathered size
                    expected_global_completions = global_num_prompts * 1

                    if accelerator.is_main_process:
                         # No need to pad/truncate gathered list here, just use its length

                         flat_generated_responses_text_global = self._generate_via_vllm_api(
                             all_prompts_text_gathered,
                             self.vllm_adapter_name,
                             eval_sampling_params_dict
                         )
                         actual_len = len(flat_generated_responses_text_global)
                         if actual_len != expected_global_completions:
                             log_on_all_ranks(accelerator, f"Warning: Eval vLLM returned {actual_len} completions, expected {expected_global_completions}. Using returned data.")
                             # Adjust expectation if needed, or handle downstream
                             expected_global_completions = actual_len # Use actual length received
                    else:
                         # Size will be determined by broadcast
                         flat_generated_responses_text_global = None # Placeholder

                    # Broadcast generated text list (rank 0 sends, others receive)
                    object_list_to_broadcast = [flat_generated_responses_text_global]
                    broadcast_object_list(object_list_to_broadcast, from_process=0)
                    flat_generated_responses_text_global = object_list_to_broadcast[0]

                    # Distribute results back to local lists (handle potential size mismatch if broadcast changed size)
                    num_received = len(flat_generated_responses_text_global)
                    expected_local_count = local_bs
                    # Simple distribution - assumes order matches ranks, handle uneven division carefully
                    items_per_rank = num_received // accelerator.num_processes
                    remainder = num_received % accelerator.num_processes
                    start_idx = accelerator.process_index * items_per_rank + min(accelerator.process_index, remainder)
                    end_idx = start_idx + items_per_rank + (1 if accelerator.process_index < remainder else 0)

                    local_generated_responses_text = flat_generated_responses_text_global[start_idx:end_idx]

                    # Store data locally
                    local_prompts_text_list_cpu.extend(local_prompts_text[:len(local_generated_responses_text)]) # Match length if uneven
                    local_completions_text_list_cpu.extend(local_generated_responses_text)
                    local_task_specific_data_list_cpu.extend(local_task_specific_data[:len(local_generated_responses_text)])

                    if step % 10 == 0: # Log progress occasionally
                        log_on_all_ranks(accelerator, f"  Eval step {step}/{len(self.eval_dataloader)}")

            # --- End Evaluation Loop ---
            log_on_all_ranks(accelerator, "Evaluation generation finished. Aggregating data for reward.")

            # --- Reward Calculation Phase ---
            all_prompts_gathered = gather_object(local_prompts_text_list_cpu)
            all_completions_gathered = gather_object(local_completions_text_list_cpu)
            all_task_data_gathered = gather_object(local_task_specific_data_list_cpu)

            all_rewards_cpu = None
            all_baseline_rewards_cpu = None
            if accelerator.is_main_process:
                total_eval_samples = len(all_prompts_gathered)
                log_on_all_ranks(accelerator, f"Rank 0: Gathered {total_eval_samples} evaluation samples.")

                if total_eval_samples > 0:
                    # Reshape completions for reward fn (list of lists with k=1)
                    nested_completions_for_reward = [[comp] for comp in all_completions_gathered]

                    # Call reward function
                    nested_scores_list = self._call_zmq_reward_server(
                        prompts=all_prompts_gathered,
                        completions=nested_completions_for_reward,
                        task_type=self.task_type,
                        task_data=all_task_data_gathered,
                        kwargs=self.reward_fn_kwargs,
                        evaluate=True
                    )

                    # Flatten scores [[r1], [r2], ...] -> [r1, r2, ...]
                    flat_scores = []
                    flat_baseline_scores = []
                    for i, sublist in enumerate(nested_scores_list):
                         if not isinstance(sublist, list) or len(sublist) != 2:
                              log_on_all_ranks(accelerator, f"Warning: Eval reward sublist for sample {i} has length {len(sublist)}, expected 2. Using NaN.")
                              flat_scores.append(float('nan'))
                              flat_baseline_scores.append(float('nan'))
                              continue
                         try:
                              # Handle potential errors during generation
                              if "ERROR: FAILED GENERATION" in all_completions_gathered[i]:
                                   flat_scores.append(-10.0) # Assign penalty score directly
                                   flat_baseline_scores.append(float('nan'))
                              else:
                                   flat_scores.append(float(sublist[0]))
                                   flat_baseline_scores.append(float(sublist[1]))
                         except (ValueError, TypeError):
                              log_on_all_ranks(accelerator, f"Warning: Eval reward score for sample {i} is not float: {sublist[0]}. Using NaN.")
                              flat_scores.append(float('nan'))
                              flat_baseline_scores.append(float('nan'))

                    all_rewards_cpu = torch.tensor(flat_scores, dtype=torch.float)
                    all_baseline_rewards_cpu = torch.tensor(flat_baseline_scores, dtype=torch.float)
                    log_on_all_ranks(accelerator, f"Calculated {all_rewards_cpu.numel()} rewards for evaluation.")
                else:
                    all_rewards_cpu = torch.tensor([], dtype=torch.float) # Empty tensor
                    all_baseline_rewards_cpu = torch.tensor([], dtype=torch.float) # Empty tensor

            # Broadcast rewards tensor to all processes
            rewards_to_broadcast = [all_rewards_cpu, all_baseline_rewards_cpu]
            if accelerator.num_processes > 1:
                broadcast_object_list(rewards_to_broadcast, from_process=0)
            all_rewards_cpu, all_baseline_rewards_cpu = rewards_to_broadcast[0], rewards_to_broadcast[1]

            # --- Log Metrics (on main process) ---
            if accelerator.is_main_process:
                eval_metrics = {}
                if all_rewards_cpu is not None and all_rewards_cpu.numel() > 0:
                    valid_rewards = all_rewards_cpu[~torch.isnan(all_rewards_cpu)]
                    if valid_rewards.numel() > 0:
                        mean_reward = valid_rewards.mean().item()
                        std_reward = valid_rewards.std().item() if valid_rewards.numel() > 1 else 0.0
                        eval_metrics = {
                            f"eval/reward_mean": mean_reward,
                            f"eval/reward_std": std_reward,
                            f"eval/samples": valid_rewards.numel(),
                            f"eval/total_samples": all_rewards_cpu.numel(),
                        }
                    else:
                         log_on_all_ranks(accelerator, "No valid rewards calculated for evaluation metrics.")
                else:
                    log_on_all_ranks(accelerator, "No rewards tensor available for evaluation metrics.")

                if all_baseline_rewards_cpu is not None and all_baseline_rewards_cpu.numel() > 0:
                    valid_baseline_rewards = all_baseline_rewards_cpu[~torch.isnan(all_baseline_rewards_cpu)]
                    if valid_baseline_rewards.numel() > 0:
                        mean_baseline_reward = valid_baseline_rewards.mean().item()
                        std_baseline_reward = valid_baseline_rewards.std().item() if valid_baseline_rewards.numel() > 1 else 0.0
                        # Add baseline metrics to the existing dict
                        eval_metrics.update({
                            f"eval/baseline_reward_mean": mean_baseline_reward,
                            f"eval/baseline_reward_std": std_baseline_reward,
                            f"eval/baseline_samples": valid_baseline_rewards.numel(),
                        })
                    else:
                        # Add NaN baseline metrics if none are valid
                        eval_metrics.update({
                            f"eval/baseline_reward_mean": float('nan'),
                            f"eval/baseline_reward_std": float('nan'),
                            f"eval/baseline_samples": 0,
                        })
                        log_on_all_ranks(accelerator, "No valid baseline rewards calculated for evaluation metrics.")
                else:
                    log_on_all_ranks(accelerator, "No baseline rewards tensor available for evaluation metrics.")

                # --- Log Combined Metrics (Only if the dictionary is not empty) ---
                if eval_metrics:
                     log_on_all_ranks(accelerator, f"Evaluation Metrics: {eval_metrics}")
                     self.log(eval_metrics)
                else:
                     log_on_all_ranks(accelerator, "No evaluation metrics were calculated.")

                # --- Optional: Log examples to WandB Table ---
                if is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
                    try:
                        log_on_all_ranks(accelerator, "Logging evaluation examples to WandB table...")
                        max_table_rows = 50 # Limit number of rows
                        num_table_rows = min(len(all_prompts_gathered), max_table_rows)
                        eval_table = wandb.Table(columns=["global_step", "prompt", "completion", "reward", "baseline_reward"])
                        indices_to_log = torch.randperm(len(all_prompts_gathered))[:num_table_rows] # Log random samples

                        for i in indices_to_log:
                            prompt_txt = all_prompts_gathered[i]
                            comp_txt = all_completions_gathered[i]
                            reward_val = all_rewards_cpu[i].item()
                            baseline_reward_val = all_baseline_rewards_cpu[i].item()
                            eval_table.add_data(global_step, prompt_txt, comp_txt, reward_val, baseline_reward_val)

                        # Log the table (uses self.log implicitly if WandbCallback is active)
                        self.log({"eval/examples": eval_table})
                        log_on_all_ranks(accelerator, f"Logged {num_table_rows} evaluation examples to WandB.")
                    except Exception as wb_err:
                        log_on_all_ranks(accelerator, f"Warning: Failed to log WandB evaluation table: {wb_err}")


        except Exception as e:
            log_on_all_ranks(accelerator, f"An error occurred during evaluation: {e}")
            import traceback
            log_on_all_ranks(accelerator, traceback.format_exc())

        finally:
            # --- Unload Adapter ---
            if adapter_loaded and accelerator.is_main_process:
                log_on_all_ranks(accelerator, f"Unloading adapter '{self.vllm_adapter_name}' after evaluation...")
                unload_success = self._unload_adapter_via_api(self.vllm_adapter_name)
                if not unload_success:
                    log_on_all_ranks(accelerator, f"Warning: Failed to unload adapter '{self.vllm_adapter_name}' after evaluation.")

            # Restore model to train mode AFTER evaluation logic finishes
            model.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            eval_end_time = time.time()
            log_on_all_ranks(accelerator, f"--- Finished Evaluation in {eval_end_time - eval_start_time:.2f}s ---")
            accelerator.wait_for_everyone() # Sync at the end

        return eval_metrics # Return calculated metrics


    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """Creates a draft model card for the trained PEFT adapter."""
        if not self.is_world_process_zero():
            return

        try:
            # Find the base model name correctly, even if wrapped
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            peft_model_instance = self._find_peft_model(unwrapped_model)

            if peft_model_instance and hasattr(peft_model_instance, 'base_model') and hasattr(peft_model_instance.base_model, 'config') and hasattr(peft_model_instance.base_model.config, '_name_or_path'):
                 base_model_name = peft_model_instance.base_model.config._name_or_path
            else:
                 # Fallback or warning if base model name cannot be determined
                 base_model_name = "unknown_base_model"
                 print("Warning: Could not automatically determine base model name for model card.")

            tags = tags or []
            if isinstance(tags, str):
                tags = [tags]
            tags.extend(self._tag_names) # Add trainer tags

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

            # Attempt to get WandB URL if available and callback is used
            wandb_url = None
            if is_wandb_available() and WandbCallback in self.callback_handler.callbacks:
                 # Check if wandb.run exists (might not if init failed or run finished)
                 if wandb.run:
                      wandb_url = wandb.run.get_url()

            # Generate model card using the utility function
            model_card = generate_model_card(
                base_model=base_model_name,
                model_name=model_name or f"{base_model_name}-{self.args.lora_adapter_name}-RLOO-Indirect-PPO",
                hub_model_id=getattr(self, 'hub_model_id', None), # Trainer doesn't set this by default
                dataset_name=dataset_name,
                tags=tags,
                wandb_url=wandb_url,
                trainer_name="RLOOIndirectTrainer (PPO-Style)",
                trainer_citation=citation,
                paper_title="Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs",
                paper_id="2402.14740",
                # peft_config=peft_model_instance.peft_config[self.args.lora_adapter_name].to_dict() if peft_model_instance else None # Add PEFT config if found
            )

            # Save the model card
            output_path = os.path.join(self.args.output_dir, "README.md")
            model_card.save(output_path)
            print(f"Model card saved to {output_path}")

        except Exception as e:
            print(f"Error generating model card: {e}")

