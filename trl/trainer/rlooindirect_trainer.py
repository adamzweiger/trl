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

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    _is_vllm_available = True
except ImportError as e:
    _is_vllm_available = False
    import sys
    import os
    print(f"!!! DEBUG vLLM IMPORT FAILED IN PROCESS !!!")
    print(f"!!! Python Path (sys.path): {sys.path}")
    print(f"!!! PYTHONPATH Env Var: {os.environ.get('PYTHONPATH')}")
    print(f"!!! Import Error: {e}")
    warnings.warn("vLLM not installed. RLOOIndirectTrainer requires vLLM. `pip install vllm`")

if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0
LORA_ADAPTER_NAME = "outer_lora" # Fixed name for the adapter we train


# Type hint for the external reward function
RewardFnType = Callable[[LLM, str, List[str], Any, Dict[str, Any]], List[float]]

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
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ) -> None:
        if not _is_vllm_available:
            raise ImportError("vLLM is required for RLOOIndirectTrainer but is not installed.")
        if not is_peft_available():
             raise ImportError("PEFT is required for RLOOIndirectTrainer but is not installed.")

        self.args = config # Use self.args consistently like Trainer base class
        args = config # Local alias for convenience

        if processing_class.padding_side != "left":
            raise ValueError("Tokenizer padding side must be 'left' for RLOOIndirectTrainer.")
        self.processing_class = processing_class

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
            data_collator = DataCollatorWithPadding(self.processing_class)
        self.data_collator = data_collator

        # --- Dataset Handling ---
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
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
        self.accelerator = Accelerator()
        if hasattr(args, 'gradient_accumulation_steps') and self.accelerator.gradient_accumulation_steps != args.gradient_accumulation_steps:
            warnings.warn(
                f"Configuration mismatch: RLOOIndirectConfig specified gradient_accumulation_steps={args.gradient_accumulation_steps}, "
                f"but Accelerator was initialized with gradient_accumulation_steps={self.accelerator.gradient_accumulation_steps}. "
                f"Using the Accelerator's value.",
                UserWarning
            )

        args.world_size = self.accelerator.num_processes

        # Batch sizes need careful calculation based on dataloader batch size and k
        args.local_batch_size_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.effective_batch_size = args.local_dataloader_batch_size * args.world_size
        args.total_batch_size_per_update = args.local_batch_size_per_step * args.world_size

        # Calculate total training steps
        if args.max_steps > 0:
            num_training_steps = args.max_steps
            num_train_epochs = (args.max_steps * args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size) / (len(self.train_dataset) // args.rloo_k)
        else:
            steps_per_epoch = math.ceil( (len(self.train_dataset) // args.rloo_k) / (args.local_dataloader_batch_size * args.world_size) )
            num_training_steps = math.ceil(args.num_train_epochs * steps_per_epoch)
            num_train_epochs = args.num_train_epochs
        args.num_training_steps = num_training_steps
        args.num_train_epochs = num_train_epochs

        time_tensor = torch.tensor(int(time.time()), device=self.accelerator.device)
        time_int = broadcast(time_tensor, 0).item()
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + self.accelerator.process_index * 100003
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, num_training_steps // args.num_sample_generations)

        # --- Optimizer and Scheduler ---
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
             # Create optimizer only for trainable PEFT parameters
             self.create_optimizer_and_scheduler(num_training_steps=num_training_steps)

        # --- Trainer Internals (adapted from Trainer/RLOOTrainer) ---
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Disable dropout in policy model
        disable_dropout_in_model(self.model)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id

        # --- vLLM Initialization ---
        # Done lazily in train() or evaluate() on the main process to avoid device conflicts
        self.llm = None
        self.sampling_params = None
        self.adapter_save_path = os.path.join(args.output_dir, f"{LORA_ADAPTER_NAME}_tmp")

        # --- Reward Function Loading ---
        self.reward_fn = self._load_reward_function(args.reward_fn_path)
        self.reward_fn_kwargs = args.reward_fn_kwargs or {}

        # --- Trainer Base Class Setup ---
        # We need to call super().__init__ but many arguments are derived differently here
        # We manually set up components like dataloaders, optimizer, accelerator preparation
        # Re-implement necessary parts of Trainer.__init__ after Accelerator setup

        # Dataloader setup
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader(eval_dataset) if eval_dataset else None

        # Prepare model, optimizer, dataloaders with Accelerator
        self.model, self.optimizer, self.lr_scheduler, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, train_dataloader, eval_dataloader
        )
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # We need state and control objects for callbacks, logging etc.
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            max_steps = num_training_steps // args.gradient_accumulation_steps, # Max steps for state/callbacks
        )
        self.control = TrainerControl()

        # Callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Add PEFT model tags
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # Other Trainer attributes needed
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.current_flos = 0 # Required by base class, not calculated here
        self.hp_search_backend = None # Required by base class
        self._signature_columns = ['input_ids', 'attention_mask', 'labels'] # Base needs this, may need adjustment

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(args.output_dir, exist_ok=True)
            # Save PEFT config
            self.model.save_pretrained(args.output_dir) # Saves adapter_config.json etc.

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

    def _init_vllm(self):
        """Initializes the vLLM engine on the main process."""
        if not self.accelerator.is_main_process:
            return # Only main process initializes vLLM

        if self.llm is not None:
            return # Already initialized

        args = self.args
        device_type = PartialState().default_device.type
        device_module = getattr(torch, device_type)
        vllm_device = args.vllm_device

        if vllm_device == "auto":
            if device_module.device_count() == 1:
                vllm_device = f"{device_type}:0"
            else:
                 # Find first unused device or use the last device if all are used by Accelerate
                used_devices = {f"{device_type}:{idx}" for idx in range(self.accelerator.num_processes)}
                available_devices = {f"{device_type}:{idx}" for idx in range(device_module.device_count())}
                free_devices = list(available_devices - used_devices)
                if free_devices:
                    vllm_device = free_devices[0]
                else:
                     vllm_device = f"{device_type}:{self.accelerator.num_processes - 1}" # Fallback to last training GPU
                     warnings.warn(
                        f"All GPUs seem occupied by Accelerate. Placing vLLM on {vllm_device}. "
                        "This might lead to OOM errors. Consider reducing num_processes for training."
                    )
        elif vllm_device in {f"{device_type}:{idx}" for idx in range(self.accelerator.num_processes)}:
             warnings.warn(
                 f"vLLM device {vllm_device} is also used for training. Ensure sufficient VRAM or use a dedicated device."
             )

        print(f"[RLOOIndirectTrainer] Initializing vLLM on device: {vllm_device}")
        self.llm = LLM(
            model=self.model.base_model.config._name_or_path, # Use base model path
            device=vllm_device.split(':')[-1], # vLLM expects device index or list
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            dtype=args.vllm_dtype,
            max_model_len=args.vllm_max_model_len,
            # Crucial for LoRA:
            enable_lora=True,
            max_lora_rank=args.max_lora_rank,
            trust_remote_code=getattr(args, "trust_remote_code", False),
            download_dir=os.environ.get("HF_HOME", None)
        )

        self.sampling_params = SamplingParams(
            n=1, # Generate 1 completion per request (we repeat requests k times)
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k is not None else -1,
            # stop_token_ids=[self.processing_class.eos_token_id] # Add EOS stop token
            # Add other sampling params if needed
        )
        if self.processing_class.eos_token_id is not None:
             self.sampling_params.stop_token_ids = [self.processing_class.eos_token_id]

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

        # Initialize vLLM on main process, wait for others
        self._init_vllm()
        accelerator.wait_for_everyone()

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
        total_train_batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil( (len(self.train_dataset) // args.rloo_k) / (args.local_dataloader_batch_size * args.world_size * args.gradient_accumulation_steps) )
        max_steps = args.num_training_steps

        print(f"  Num examples = {len(self.train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs:.2f}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, dist. & accum.) = {total_train_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_steps}")

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # --- Training Loop ---
        for step in range(max_steps):
            # Collective gathering dictionaries
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
            pg_clipfrac_stats_accum = [] # Not used by RLOO but keep structure if needed later
            pg_loss_stats_accum = []
            vf_clipfrac_stats_accum = [] # Not used by RLOO
            entropy_stats_accum = []
            ratio_stats_accum = []

            # --- Experience Generation Phase ---
            # Accumulate gradients for args.gradient_accumulation_steps
            for _ in range(args.gradient_accumulation_steps):
                with torch.no_grad():
                    # Get batch: keys are 'input_ids', 'attention_mask', 'target'
                    # Batch size is local_dataloader_batch_size
                    raw_batch = next(iter_dataloader)
                    prompts_data = raw_batch['input_ids'].to(device)
                    targets = raw_batch['target'] # Keep targets on CPU or move later as needed

                    # Repeat prompts k times for RLOO
                    queries = prompts_data.repeat_interleave(args.rloo_k, dim=0)
                    repeated_targets = [t for t in targets for _ in range(args.rloo_k)]
                    context_length = queries.shape[1]

                    # --- vLLM Generation ---
                    # 1. Save current adapter state (only main process needs to save)
                    if accelerator.is_main_process:
                         # Need to handle potential races if saving takes time? Unlikely with small adapters.
                         # Ensure model is not in FSDP-wrapped state if saving parameters directly
                         # self.model.save_pretrained(self.adapter_save_path) # PeftModel save
                        
                         # Use accelerator.save_state for better handling of distributed states
                         # Saving only the adapter weights might require custom logic or using peft's save
                         # For simplicity/correctness with Accelerator, save full state, then potentially load only adapter later if needed
                         # Let's stick to peft's save for now, assuming it works correctly even if model is prepared by accelerator
                        unwrapped_model = accelerator.unwrap_model(self.model)
                        unwrapped_model.save_pretrained(self.adapter_save_path)

                    accelerator.wait_for_everyone() # Ensure save completes before others proceed

                    # 2. Prepare vLLM request
                    lora_request = LoRARequest(
                        lora_name=LORA_ADAPTER_NAME,
                        lora_int_id=step, # Use step or unique ID for vLLM internal cache key maybe?
                        lora_local_path=self.adapter_save_path,
                    )

                    # 3. Decode queries for vLLM
                    queries_text = self.processing_class.batch_decode(queries, skip_special_tokens=True)

                    # 4. Generate using vLLM (only main process interacts with LLM)
                    generated_responses_text = [None] * len(queries_text) # Placeholder on non-main processes
                    if accelerator.is_main_process:
                        vllm_outputs = self.llm.generate(
                            queries_text,
                            self.sampling_params,
                            lora_request=lora_request,
                            use_tqdm=False # Disable vLLM TQDM
                        )
                        # Extract generated text
                        generated_responses_text = [output.outputs[0].text for output in vllm_outputs]

                    # 5. Broadcast results
                    generated_responses_text = broadcast(generated_responses_text)

                    # 6. Tokenize responses
                    responses_tokenized = self.processing_class(
                        generated_responses_text,
                        padding='longest', # Pad responses to the same length within the k*batch_size group
                        truncation=True,
                        max_length=args.max_completion_length,
                        return_tensors="pt",
                    ).to(device)
                    responses_ids = responses_tokenized.input_ids
                    responses_mask = responses_tokenized.attention_mask # Use this? Or calculate based on padding later?

                    # Ensure responses_ids doesn't start with BOS if tokenizer adds it
                    if self.processing_class.bos_token_id is not None and responses_ids.shape[1] > 0:
                        if (responses_ids[:, 0] == self.processing_class.bos_token_id).all():
                             responses_ids = responses_ids[:, 1:]
                             responses_mask = responses_mask[:, 1:]

                    # Truncate responses at stop token if specified
                    processed_responses_ids = responses_ids
                    if args.stop_token_id is not None:
                        processed_responses_ids = truncate_response(
                            args.stop_token_id, self.processing_class.pad_token_id, responses_ids
                        )

                    # --- Log Prob Calculation ---
                    # Construct full input sequences (query + response)
                    query_responses_ids = torch.cat([queries, processed_responses_ids], dim=1)
                    # Need attention mask for the combined sequence
                    query_mask = torch.ones_like(queries) # Assuming queries are not padded internally
                    # Response mask needs to account for padding *within* the response batch
                    resp_padding_mask = (processed_responses_ids == self.processing_class.pad_token_id)
                    resp_attn_mask = ~resp_padding_mask
                    query_responses_mask = torch.cat([query_mask, resp_attn_mask], dim=1)


                    # Policy logprobs (adapter enabled)
                    with accelerator.unwrap_model(self.model).enable_adapter(): # Ensure adapter is active
                        outputs = forward(self.model, query_responses_ids, query_responses_mask) # Check if forward needs specific mask handling
                        # Logits shape: (k*batch, seq_len, vocab_size)
                        # We need logits corresponding to the *response* tokens only
                        # Shift logits left by 1: logit[i] predicts token[i+1]
                        logits = outputs.logits[:, context_length - 1 : -1] # Get logits for response tokens
                        logits /= args.temperature + 1e-7
                        logprobs = selective_log_softmax(logits, processed_responses_ids)
                        del outputs, logits
                        torch.cuda.empty_cache()

                    # Reference logprobs (adapter disabled)
                    with accelerator.unwrap_model(self.model).disable_adapter():
                        ref_outputs = forward(self.model, query_responses_ids, query_responses_mask)
                        ref_logits = ref_outputs.logits[:, context_length - 1 : -1]
                        ref_logits /= args.temperature + 1e-7
                        ref_logprobs = selective_log_softmax(ref_logits, processed_responses_ids)
                        del ref_outputs, ref_logits
                        torch.cuda.empty_cache()


                    # --- Reward Calculation ---
                    # Decode processed responses for reward function
                    processed_responses_text = self.processing_class.batch_decode(processed_responses_ids, skip_special_tokens=True)

                    # Group texts and targets for the reward function
                    # queries_text_repeated = self.processing_class.batch_decode(queries, skip_special_tokens=True) # Already have this
                    scores = torch.zeros(len(queries_text), device=device, dtype=torch.float)

                    # Call reward function for each original prompt (grouping k responses)
                    # This might be slow if called serially. Consider batching if reward_fn supports it.
                    # Assuming reward_fn takes one prompt's data at a time for now.
                    current_idx = 0
                    original_prompts_text = self.processing_class.batch_decode(prompts_data, skip_special_tokens=True)
                    for i in range(args.local_dataloader_batch_size):
                         prompt_text = original_prompts_text[i]
                         target = targets[i]
                         k_completions = processed_responses_text[current_idx : current_idx + args.rloo_k]

                         # Call external reward function
                         # Needs the vLLM engine instance - only available on main process?
                         # This design requires reward_fn to either:
                         # a) Not need the llm object OR
                         # b) Be callable only on main process, requiring gathering/scattering results OR
                         # c) Have llm initialized everywhere (less ideal)
                         # Let's assume reward_fn is called everywhere but might only use llm on main process if needed.
                         # If llm is needed by reward_fn, it must handle the main_process check internally or we broadcast results.
                         # For simplicity now, assume llm is passed but might be None on non-main processes if reward_fn handles it.
                         llm_for_reward = self.llm if accelerator.is_main_process else None
                         # Handle potential case where reward_fn doesn't need llm
                         import inspect
                         sig = inspect.signature(self.reward_fn)
                         reward_kwargs = self.reward_fn_kwargs.copy()
                         if 'llm' in sig.parameters:
                             reward_kwargs['llm'] = llm_for_reward
                         if 'target' in sig.parameters:
                            reward_kwargs['target'] = target

                         k_scores = self.reward_fn(
                             prompt_text=prompt_text,
                             completions_text=k_completions,
                             **reward_kwargs # Pass llm, target, and other kwargs
                         )

                         if not isinstance(k_scores, list) or len(k_scores) != args.rloo_k:
                             raise ValueError(f"Reward function must return a list of {args.rloo_k} floats.")

                         scores[current_idx : current_idx + args.rloo_k] = torch.tensor(k_scores, device=device, dtype=torch.float)
                         current_idx += args.rloo_k

                    # Post-process scores (EOS penalty, normalization, clipping)
                    contain_eos_token = torch.any(processed_responses_ids == self.processing_class.eos_token_id, dim=-1)
                    if args.missing_eos_penalty is not None:
                        scores[~contain_eos_token] -= args.missing_eos_penalty

                    # Create padding mask for logprobs based on processed responses
                    sequence_lengths = first_true_indices(processed_responses_ids == self.processing_class.pad_token_id) - 1
                    response_idxs = torch.arange(processed_responses_ids.shape[1], device=device).repeat(processed_responses_ids.shape[0], 1)
                    padding_mask = response_idxs > sequence_lengths.unsqueeze(1) # True where padded

                    logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                    ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                    # --- RLOO Advantage Calculation ---
                    kl = logprobs - ref_logprobs # Shape: (k*batch, resp_len)

                    # Normalize raw scores (rewards from function) if needed
                    if args.normalize_reward:
                        # Normalize across the entire generation batch (k * local_dataloader_batch_size)
                        gathered_scores = accelerator.gather(scores)
                        mean_score = gathered_scores.mean()
                        std_score = gathered_scores.std()
                        scores = (scores - mean_score) / (std_score + 1e-8)
                        scores = torch.clamp(scores, -args.reward_clip_range, args.reward_clip_range)

                    # Combine score and KL penalty
                    if args.token_level_kl:
                         # Apply KL penalty per token, add score at the end
                         kl_reward = -args.kl_coef * kl # Shape: (k*batch, resp_len)

                         eos_indices = padding_mask.size(1) - 1 - padding_mask.long().fliplr().argmax(dim=1, keepdim=True)
                         last_reward_tensor = torch.zeros_like(kl_reward)
                         scores_shaped = scores.reshape(-1, 1).to(kl_reward.dtype)
                         last_reward_tensor.scatter_(dim=1, index=eos_indices, src=scores_shaped)

                         non_score_reward_per_step = kl_reward.sum(1) # Sum KL rewards per sequence (for logging)
                         combined_reward_per_step = last_reward_tensor + kl_reward # Shape: (k*batch, resp_len)
                         rlhf_reward = combined_reward_per_step.sum(1) # Total reward per sequence
                    else:
                         # Apply KL penalty at sequence level
                         sequence_kl = kl.sum(1) # Shape: (k*batch)
                         non_score_reward_per_step = -args.kl_coef * sequence_kl
                         rlhf_reward = non_score_reward_per_step + scores # Total reward per sequence

                    # Calculate RLOO baseline and advantages
                    # Reshape rewards to (k, local_dataloader_batch_size)
                    rlhf_reward_grouped = rlhf_reward.reshape(args.rloo_k, -1)
                    baseline = (rlhf_reward_grouped.sum(0, keepdim=True) - rlhf_reward_grouped) / (args.rloo_k - 1)
                    advantages = rlhf_reward_grouped - baseline # Shape: (k, local_dataloader_batch_size)
                    advantages = advantages.flatten() # Back to (k * local_dataloader_batch_size)

                    # Normalize advantages if needed
                    if args.normalize_advantage:
                         # Normalize across the entire generation batch
                         gathered_advantages = accelerator.gather(advantages)
                         mean_adv = gathered_advantages.mean()
                         std_adv = gathered_advantages.std()
                         advantages = (advantages - mean_adv) / (std_adv + 1e-8)

                    # --- Store batch data for optimization phase ---
                    # Only store tensors needed for loss calculation
                    all_query_ids_list.append(queries.cpu()) # Move to CPU if accumulating many steps
                    all_response_ids_list.append(processed_responses_ids.cpu())
                    all_logprobs_list.append(logprobs.sum(1).cpu()) # Store sum of logprobs
                    all_advantages_list.append(advantages.cpu())
                    # Optional: store for detailed logging/debugging
                    all_sequence_lengths_list.append(sequence_lengths.cpu())
                    all_ref_logprobs_list.append(ref_logprobs.sum(1).cpu())
                    all_scores_list.append(scores.cpu())
                    all_kl_list.append(kl.sum(1).cpu())
                    all_rlhf_rewards_list.append(rlhf_reward.cpu())
                    all_non_score_rewards_list.append(non_score_reward_per_step.cpu())

                # --- End of Experience Generation (within gradient accumulation loop) ---
                # Free up memory
                del (queries, responses_ids, processed_responses_ids, query_responses_ids, query_responses_mask,
                     logprobs, ref_logprobs, kl, kl_reward, non_score_reward_per_step, rlhf_reward,
                     rlhf_reward_grouped, baseline, advantages, scores, padding_mask, sequence_lengths)
                if 'k_scores' in locals(): del k_scores
                if 'logits' in locals(): del logits
                if 'ref_logits' in locals(): del ref_logits
                if 'outputs' in locals(): del outputs
                if 'ref_outputs' in locals(): del ref_outputs
                torch.cuda.empty_cache()
                gc.collect()

            # --- End of Gradient Accumulation Loop ---

            # --- Optimization Phase ---
            # Collate accumulated data
            if not all_advantages_list: continue # Skip update if no data collected

            batch_query_ids = torch.cat(all_query_ids_list, dim=0).to(device)
            batch_response_ids = torch.cat(all_response_ids_list, dim=0).to(device)
            batch_old_logprobs_sum = torch.cat(all_logprobs_list, dim=0).to(device)
            batch_advantages = torch.cat(all_advantages_list, dim=0).to(device)
            local_accumulation_batch_size = len(batch_advantages) # Total samples in this update step on this device

            # RLOO typically updates once per batch of experience (like REINFORCE)
            # The original RLOOTrainer had num_ppo_epochs, let's respect that structure
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                # Shuffle indices for minibatches within the accumulated batch
                b_inds = torch.randperm(local_accumulation_batch_size, device=device)
                minibatch_idx = 0
                # Loop over minibatches
                for mini_batch_start in range(0, local_accumulation_batch_size, args.local_batch_size_per_step): # Use local_batch_size_per_step as minibatch size? Check calculation
                    mini_batch_end = mini_batch_start + args.local_batch_size_per_step
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]

                    # We perform the backward pass inside the accumulate context
                    # This seems different from original RLOO trainer?
                    # Original RLOO accumulated gradients *within* the minibatch loop of PPO epochs
                    # Here, we've already generated all data. Let's adjust.
                    # We should iterate through minibatches and do forward/backward for each.

                    # Get minibatch data
                    mb_query_ids = batch_query_ids[mini_batch_inds]
                    mb_response_ids = batch_response_ids[mini_batch_inds]
                    mb_old_logprobs_sum = batch_old_logprobs_sum[mini_batch_inds]
                    mb_advantages = batch_advantages[mini_batch_inds]

                    # Recompute logprobs for the current policy
                    mb_query_responses_ids = torch.cat([mb_query_ids, mb_response_ids], dim=1)
                    # Need mask. Recreate or store? Let's recreate mask
                    mb_query_mask = torch.ones_like(mb_query_ids)
                    mb_resp_padding_mask = (mb_response_ids == self.processing_class.pad_token_id)
                    mb_resp_attn_mask = ~mb_resp_padding_mask
                    mb_query_responses_mask = torch.cat([mb_query_mask, mb_resp_attn_mask], dim=1)
                    mb_context_length = mb_query_ids.shape[1]

                    # Forward pass with gradient enabled
                    with accelerator.accumulate(self.model): # Accumulate gradients here
                        output = forward(self.model, mb_query_responses_ids, mb_query_responses_mask)
                        logits = output.logits[:, mb_context_length - 1 : -1]
                        logits /= args.temperature + 1e-7

                        # Compute new logprobs (token level)
                        new_logprobs_token = selective_log_softmax(logits, mb_response_ids)
                        # Apply padding mask (True where padded)
                        mb_padding_mask = response_idxs = torch.arange(mb_response_ids.shape[1], device=device).repeat(mb_response_ids.shape[0], 1) > \
                                            (first_true_indices(mb_response_ids == self.processing_class.pad_token_id) - 1).unsqueeze(1)
                        new_logprobs_token = torch.masked_fill(new_logprobs_token, mb_padding_mask, INVALID_LOGPROB)
                        new_logprobs_sum = new_logprobs_token.sum(1) # Sum logprobs for sequence

                        # --- RLOO Loss Calculation ---
                        logprobs_diff = new_logprobs_sum - mb_old_logprobs_sum # mb_old_logprobs_sum is already detached (from CPU)
                        ratio = torch.exp(logprobs_diff)
                        pg_loss = -mb_advantages * ratio
                        pg_loss = pg_loss.mean() # Average loss over minibatch

                        # KL penalty (sequence level if not token level)
                        loss = pg_loss
                        if not args.token_level_kl:
                            # Recompute KL on the fly for this minibatch? Need ref_logprobs for minibatch.
                            # This is inefficient. Let's assume KL penalty is handled via reward if seq level.
                            # Revisit this: Original RLOO included KL in reward even for seq level.
                            # Sticking to reward-based KL for now. Loss is just policy gradient loss.
                            pass # KL handled in reward calculation


                        # Backward pass (managed by accelerator.accumulate)
                        accelerator.backward(loss)

                        # --- Log Stats (inside accumulation context) ---
                        with torch.no_grad():
                             # Approx KL between old and new policy (different from KL vs ref policy)
                             approxkl = 0.5 * (logprobs_diff**2).mean().item()
                             policy_loss = pg_loss.item()
                             # Entropy calculation
                             prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                             entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1) # Per token
                             # Mask entropy based on padding mask, then average over non-padded tokens
                             masked_entropy = entropy.masked_fill(mb_padding_mask, 0.0)
                             mean_entropy = masked_entropy.sum() / (~mb_padding_mask).sum()
                             mean_entropy = mean_entropy.item()

                             approxkl_stats_accum.append(approxkl)
                             pg_loss_stats_accum.append(policy_loss)
                             entropy_stats_accum.append(mean_entropy)
                             ratio_stats_accum.append(ratio.mean().item()) # Log mean ratio

                    # End accumulate context
                    minibatch_idx += 1

                    if accelerator.sync_gradients:
                         # Accumulation finished, perform optimizer step
                         # Clip gradients if needed
                         if args.max_grad_norm is not None:
                             accelerator.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                         self.optimizer.step()
                         self.optimizer.zero_grad()


                # --- End Minibatch Loop ---
            # --- End PPO Epoch Loop ---

            # --- LR Scheduling and Logging ---
            if accelerator.sync_gradients:
                self.lr_scheduler.step() # Step scheduler after optimizer step
                self.state.global_step += 1

                # Gather accumulated stats across all GPUs for logging
                if accelerator.is_main_process:
                     # Collate data from the generation phase for logging metrics
                     log_scores = torch.cat(all_scores_list, dim=0).float()
                     log_advantages = torch.cat(all_advantages_list, dim=0).float()
                     log_rlhf_rewards = torch.cat(all_rlhf_rewards_list, dim=0).float()
                     log_non_score_rewards = torch.cat(all_non_score_rewards_list, dim=0).float()
                     log_kl_sum = torch.cat(all_kl_list, dim=0).float()
                     log_old_logprobs_sum = torch.cat(all_logprobs_list, dim=0).float()
                     log_seq_lengths = torch.cat(all_sequence_lengths_list, dim=0)

                     # Aggregate stats from optimization phase
                     mean_approxkl = torch.tensor(np.mean(approxkl_stats_accum), device=device) if approxkl_stats_accum else torch.tensor(0.0, device=device)
                     mean_pg_loss = torch.tensor(np.mean(pg_loss_stats_accum), device=device) if pg_loss_stats_accum else torch.tensor(0.0, device=device)
                     mean_entropy = torch.tensor(np.mean(entropy_stats_accum), device=device) if entropy_stats_accum else torch.tensor(0.0, device=device)
                     mean_ratio = torch.tensor(np.mean(ratio_stats_accum), device=device) if ratio_stats_accum else torch.tensor(1.0, device=device)

                     # Reduce metrics across processes
                     mean_score_red = reduce(log_scores.mean(), reduction='mean')
                     mean_adv_red = reduce(log_advantages.mean(), reduction='mean')
                     std_adv_red = reduce(log_advantages.std(), reduction='mean')
                     mean_rlhf_reward_red = reduce(log_rlhf_rewards.mean(), reduction='mean')
                     mean_non_score_reward_red = reduce(log_non_score_rewards.mean(), reduction='mean')
                     mean_kl_red = reduce(log_kl_sum.mean(), reduction='mean')
                     mean_seq_len_red = reduce(log_seq_lengths.float().mean(), reduction='mean')
                     mean_approxkl_red = reduce(mean_approxkl, reduction='mean')
                     mean_pg_loss_red = reduce(mean_pg_loss, reduction='mean')
                     mean_entropy_red = reduce(mean_entropy, reduction='mean')
                     mean_ratio_red = reduce(mean_ratio, reduction='mean')


                     metrics = {}
                     metrics["train/episode"] = self.state.global_step * total_train_batch_size
                     metrics["train/reward_score"] = mean_score_red.item()
                     metrics["train/reward_rlhf"] = mean_rlhf_reward_red.item()
                     metrics["train/reward_non_score"] = mean_non_score_reward_red.item()
                     metrics["train/advantage_mean"] = mean_adv_red.item()
                     metrics["train/advantage_std"] = std_adv_red.item()
                     metrics["train/kl_ref_policy"] = mean_kl_red.item() # KL vs reference policy
                     metrics["train/policy_entropy"] = mean_entropy_red.item()
                     metrics["train/loss_policy"] = mean_pg_loss_red.item()
                     metrics["train/kl_approx"] = mean_approxkl_red.item() # KL vs old policy
                     metrics["train/ratio"] = mean_ratio_red.item()
                     metrics["train/seq_length"] = mean_seq_len_red.item()
                     metrics["train/lr"] = self.lr_scheduler.get_last_lr()[0]
                     metrics["train/epoch"] = self.state.epoch

                     self.log(metrics)

            # --- Callback Handling ---
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            # --- Checkpointing ---
            if self.control.should_save:
                 self._save_checkpoint()
                 self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            # --- Evaluation ---
            if self.control.should_evaluate:
                 # Implement evaluation logic if needed
                 # metrics = self.evaluate()
                 # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
                 pass # Placeholder for evaluation


            # --- Sample Generation Logging ---
            if accelerator.is_main_process and args.num_sample_generations > 0 and self.state.global_step > 0 and \
               (self.state.global_step % self.sample_generations_freq == 0 or self.control.should_training_stop):
                 self.generate_completions(sampling=True)


            # Check if training should stop
            if self.control.should_training_stop:
                 break

            # Update epoch state
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
         """Generates completions for evaluation or logging using vLLM."""
         if not self.is_world_process_zero(): # Only main process generates
             return

         args = self.args
         if self.llm is None:
             print("Warning: vLLM not initialized, skipping completion generation.")
             return

         eval_dataloader = dataloader if dataloader else self.get_eval_dataloader()
         if eval_dataloader is None:
             print("Warning: No evaluation dataloader found, skipping completion generation.")
             return

         print(f"\n=== Generating Completions at Step {self.state.global_step} ===")

         # Use a sampling config potentially different from training
         eval_sampling_params = SamplingParams(
             n=1,
             max_tokens=args.max_completion_length,
             temperature=0.1 if not sampling else args.temperature, # Low temp for greedy eval
             top_p=1.0,
             top_k=-1,
             stop_token_ids=self.sampling_params.stop_token_ids,
         )

         # Prepare LoRA request with current adapter
         # Make sure the tmp path reflects the *latest saved* adapter if called mid-training
         # If called after _save_checkpoint, adapter_save_path might not be the final checkpoint path
         current_adapter_path = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
         if not os.path.exists(os.path.join(current_adapter_path, "adapter_model.bin")): # Check if checkpoint exists
             current_adapter_path = self.adapter_save_path # Fallback to tmp path if checkpoint not saved yet

         if not os.path.exists(os.path.join(current_adapter_path, "adapter_model.bin")):
             print(f"Warning: Adapter not found at {current_adapter_path}, generating with base model.")
             eval_lora_request = None
         else:
             eval_lora_request = LoRARequest(
                 lora_name=f"{LORA_ADAPTER_NAME}_eval",
                 lora_int_id=self.state.global_step + 1000, # Unique ID
                 lora_local_path=current_adapter_path,
             )

         table = defaultdict(list)
         for batch in eval_dataloader:
             prompts_ids = batch["input_ids"].to(self.llm.device) # Move prompts to vLLM device if needed
             targets = batch["target"]
             prompts_text = self.processing_class.batch_decode(prompts_ids, skip_special_tokens=True)

             # Generate
             vllm_outputs = self.llm.generate(
                 prompts_text,
                 eval_sampling_params,
                 lora_request=eval_lora_request,
                 use_tqdm=True
             )
             completions_text = [output.outputs[0].text for output in vllm_outputs]

             # Calculate rewards using the external function
             scores = []
             for i in range(len(prompts_text)):
                 # Prepare kwargs for reward function
                 import inspect
                 sig = inspect.signature(self.reward_fn)
                 reward_kwargs = self.reward_fn_kwargs.copy()
                 if 'llm' in sig.parameters: reward_kwargs['llm'] = self.llm
                 if 'target' in sig.parameters: reward_kwargs['target'] = targets[i]

                 # Note: Passing a list containing a single completion
                 score = self.reward_fn(
                     prompt_text=prompts_text[i],
                     completions_text=[completions_text[i]], # Pass as list
                     **reward_kwargs
                 )
                 scores.append(score[0] if isinstance(score, list) else score) # Take first element if list returned


             table["prompt"].extend(prompts_text)
             table["target"].extend([str(t) for t in targets]) # Store string representation of target
             table["model_response"].extend(completions_text)
             table["score"].extend(scores)

             if sampling: # Only generate one batch if sampling for logs
                 break

         df = pd.DataFrame(table)

         if self.is_world_process_zero(): # Ensure only main process logs
             print_rich_table(df.head())
             log_file_path = os.path.join(args.output_dir, f"completions_step_{self.state.global_step}.csv")
             df.to_csv(log_file_path, index=False)
             print(f"Saved completions log to {log_file_path}")

             if "wandb" in args.report_to and wandb.run is not None:
                 try:
                     wandb.log({f"eval/completions_step_{self.state.global_step}": wandb.Table(dataframe=df)})
                 except Exception as e:
                     print(f"Warning: Failed to log table to wandb: {e}")

             if "comet_ml" in args.report_to:
                 try:
                     log_table_to_comet_experiment(
                         name=f"completions_step_{self.state.global_step}.csv",
                         table=df,
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
