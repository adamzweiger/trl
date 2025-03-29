#!/usr/bin/env python3
"""
external/trl/trl/trainer/rloo_self_edit_trainer.py

This trainer implements a self-editing RL training loop.
Instead of using standard generation and reward functions, this trainer calls an inner loop
(from src/trainRL/math_training.py) that:
  - Samples multiple generations (using vLLM with the current RL_lora adapter) for a given prompt,
  - Returns the generated texts and their rewards (currently computed randomly).

The trainer then uses a differentiable forward pass through the model (a HuggingFace forward pass)
to compute the log probabilities of the generated tokens. These log probabilities (computed from
the full prompt plus the generation) are multiplied by the rewards to form a policy-gradient loss,
which is then backpropagated to update the RL_lora adapter.
"""

import gc
import os
import time
from typing import Optional, Callable

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, Trainer, TrainerCallback, is_wandb_available

if is_wandb_available():
    import wandb

from vllm import SamplingParams, LLM
from src.trainRL.math_training import rl_inner_loop_iteration

def get_lora_request_from_policy(policy):
    """
    Create a LoRARequest from a PEFT model (policy).
    In a proper implementation, this function would extract the current adapter parameters from the policy.
    Here we simply construct a LoRARequest with an empty lora_path to indicate in-memory parameters.
    """
    from vllm.lora.request import LoRARequest
    lora_path = "tmp_lora_path"
    policy.save_pretrained(lora_path)
    return LoRARequest(
        lora_name="rl_lora",
        lora_int_id=0,
        lora_path=lora_path
    )

class RLOOSelfEditTrainer(Trainer):
    _tag_names = ["trl", "rloo", "self_edit"]

    def __init__(
        self,
        config,
        processing_class: Optional[PreTrainedTokenizerBase],
        policy: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[Callable] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        # Inner loop hyperparameters:
        num_generations: int = 5,
        inner_temperature: float = 0.7,
        inner_top_p: float = 0.95,
        inner_max_tokens: int = 4096,
        gradient_accumulation_steps: int = 1,
        gpu_memory_utilization: float = 0.7,  # Lower memory utilization for vLLM
    ) -> None:
        super().__init__(
            model=policy,
            args=config,
            train_dataset=train_dataset,
            data_collator=data_collator,
            optimizers=optimizers,
            callbacks=callbacks,
        )
        self.processing_class = processing_class
        self.policy = policy  # This is our RL_lora adapter (a PEFT model).
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        torch.manual_seed(config.seed)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        self.inner_loop_sampling_params = SamplingParams(
            temperature=inner_temperature,
            top_p=inner_top_p,
            max_tokens=inner_max_tokens,
            stop_token_ids=[],
            n=num_generations
        )
        self.model_name = config.model_name

        # Initialize a separate vLLM model for the inner loop generations.
        self.vllm_model = LLM(
            model=self.model_name,
            enable_lora=True,
            max_model_len=config.max_model_len,
            max_lora_rank=config.max_lora_rank,
            trust_remote_code=True,
            tensor_parallel_size=2,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=os.environ.get("HF_HOME", None)
        )

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        model.train()
        device = accelerator.device

        total_loss = 0.0
        total_reward = 0.0
        total_samples = 0

        print(f"Training started with {len(self.dataloader)} batches.")

        for step, batch in enumerate(self.dataloader):
            batch_loss = 0.0
            batch_reward = 0.0
            batch_count = 0

            # Expect each sample to contain a "prompt" key.
            if isinstance(batch, dict) and "prompt" in batch:
                samples = [{"prompt": p} for p in batch["prompt"]]
            else:
                samples = batch

            for sample in samples:
                prompt = sample.get("prompt", "")
                print(f"Processing prompt: {prompt}")
                # Call the inner loop using the prompt, the vLLM model, and the current LoRA adapter.
                result = rl_inner_loop_iteration(prompt, self.vllm_model, get_lora_request_from_policy(self.policy), self.inner_loop_sampling_params)
                print(f"Generated {len(result['generations'])} generations for prompt: {prompt}")
                sample_losses = []
                for gen_text, reward in zip(result["generations"], result["rewards"]):
                    print(f"Generated text: {gen_text}, Reward: {reward}")
                    full_text = prompt + gen_text
                    tokenized = self.processing_class(full_text, return_tensors="pt", truncation=True).to(device)
                    outputs = self.model(**tokenized)
                    logits = outputs.logits
                    # Compute log probabilities over generated tokens.
                    prompt_tokenized = self.processing_class(prompt, return_tensors="pt").to(device)
                    prompt_length = prompt_tokenized["input_ids"].size(1)
                    generated_token_ids = tokenized["input_ids"][:, prompt_length:]
                    generated_logits = logits[:, prompt_length:]
                    generated_log_probs = torch.log_softmax(generated_logits, dim=-1)
                    selected_log_probs = generated_log_probs.gather(dim=-1, index=generated_token_ids.unsqueeze(-1)).squeeze(-1)
                    total_log_prob = selected_log_probs.sum()
                    sample_losses.append(- reward * total_log_prob)

                    print(f"Sample loss for generation: {sample_losses[-1].item():.4f}, Reward: {reward:.4f}")
                    
                    # Clean up intermediate tensors
                    del tokenized, outputs, logits, generated_logits, generated_log_probs, selected_log_probs
                    torch.cuda.empty_cache()
                    
                # Clean up prompt tokenization
                del prompt_tokenized, generated_token_ids
                
                if sample_losses:
                    sample_loss = sum(sample_losses) / len(sample_losses)
                else:
                    sample_loss = torch.tensor(0.0, device=device)
                batch_loss += sample_loss
                avg_reward = sum(result["rewards"]) / len(result["rewards"]) if result["rewards"] else 0.0
                batch_reward += avg_reward
                batch_count += 1
                total_samples += 1

                print(f"Batch loss: {batch_loss.item():.4f}, Batch reward: {batch_reward:.4f}, Total samples: {total_samples}")
                print(f"Average reward for this batch: {avg_reward:.4f}")
                
                # Clean up result after each sample
                del result, sample_losses
                if 'sample_loss' in locals():
                    del sample_loss

            if batch_count > 0:
                batch_loss = batch_loss / batch_count
                batch_reward = batch_reward / batch_count
            else:
                batch_loss = torch.tensor(0.0, device=device)

            print(f"Batch loss after averaging: {batch_loss.item():.4f}, Batch reward after averaging: {batch_reward:.4f}")

            optimizer.zero_grad()
            accelerator.backward(batch_loss)
            optimizer.step()

            print(f"Step {step}: Backpropagation completed. Loss: {batch_loss.item():.4f}, Reward: {batch_reward:.4f}")

            total_loss += batch_loss.item()
            total_reward += batch_reward

            if step % args.logging_steps == 0:
                print(f"Step {step}: Loss {batch_loss.item():.4f}, Avg Reward {batch_reward:.4f}")
            if args.save_steps is not None and step % args.save_steps == 0:
                self._save_checkpoint(model)

            # Clean up batch variables
            del batch, batch_loss
            torch.cuda.empty_cache()
            gc.collect()

        avg_loss = total_loss / (step + 1)
        avg_reward = total_reward / (step + 1)
        print(f"Training completed: Avg Loss {avg_loss:.4f}, Avg Reward {avg_reward:.4f}")
        return model

    def _save_checkpoint(self, model):
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, f"checkpoint-step-{int(time.time())}")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
