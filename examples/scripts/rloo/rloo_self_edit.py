#!/usr/bin/env python3
"""
rloo_self_edit.py

This script loads a JSON file of prompts and targets, converts it into a dataset,
and trains an RL_lora adapter using a self-editing RL inner loop.
The inner loop (defined in src/trainRL/math_training.py) is called by the trainer
to sample multiple generations for each prompt, and then a policy-gradient loss is computed
(using differentiable log probabilities) to update the adapter.

Usage:
  python rloo_self_edit.py \
    --json_file logs/prompts_targets.json \
    --output_dir models/self_edit_rl \
    --model_name EleutherAI/pythia-1b-deduped \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_steps 1000 \
    --learning_rate 5e-5 \
    --seed 42 \
    --num_generations 5 \
    --inner_temperature 0.7 \
    --inner_top_p 0.95 \
    --inner_max_tokens 256 \
    --outer_lora_rank 8 \
    --outer_lora_alpha 16 \
    --outer_lora_dropout 0.1
"""

import argparse
import json
import os
import shutil
import torch

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import PartialState
from peft import LoraConfig, get_peft_model

# Import the RLOOSelfEditTrainer
from trl import RLOOSelfEditTrainer

def simple_data_collator(features):
    """
    A simple data collator that groups raw dictionary samples into lists.
    Each sample is expected to have keys such as "prompt" and "target".
    """
    collated = {}
    for key in features[0]:
        collated[key] = [f[key] for f in features]
    return collated

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RLOO Self-Edit Trainer")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file with prompts and targets")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints and model saving")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name for the base model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--total_steps", type=int, default=1000,
                        help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    # Inner loop hyperparameters
    parser.add_argument("--num_generations", type=int, default=5,
                        help="Number of generations to sample per prompt in the inner loop")
    parser.add_argument("--inner_temperature", type=float, default=0.7,
                        help="Sampling temperature for the inner loop")
    parser.add_argument("--inner_top_p", type=float, default=0.95,
                        help="Top-p sampling value for the inner loop")
    parser.add_argument("--inner_max_tokens", type=int, default=256,
                        help="Maximum tokens to generate in the inner loop")
    # Outer LoRA hyperparameters
    parser.add_argument("--outer_lora_rank", type=int, default=8,
                        help="LoRA rank for the outer adapter")
    parser.add_argument("--outer_lora_alpha", type=int, default=16,
                        help="LoRA alpha for the outer adapter")
    parser.add_argument("--outer_lora_dropout", type=float, default=0.1,
                        help="LoRA dropout for the outer adapter")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoints steps")
    parser.add_argument("--max_model_len", type=int, default=12800,
                        help="Maximum model length")
    args = parser.parse_args()
    
    # Remove output directory if it exists
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load JSON file of prompts and targets
    with open(args.json_file, "r") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = {
            "prompt": item.get("prompt", ""),
            "target": item.get("target", "")
        }
        samples.append(sample)

    dataset = Dataset.from_list(samples)
    train_dataset = dataset
    print(f"Loaded {len(train_dataset)} samples from {args.json_file}")
    # Load tokenizer and setup padding
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        padding_side="left", 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if not hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = "{prompt}\n"
    print(f"Tokenizer loaded with pad token: {tokenizer.pad_token}")
    # Load the base model and wrap it as a PEFT model using LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=os.environ.get("HF_HOME", None)
    )
    print(f"Base model loaded: {args.model_name}")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.outer_lora_rank,
        lora_alpha=args.outer_lora_alpha,
        lora_dropout=args.outer_lora_dropout,
        bias="none"
    )
    policy = get_peft_model(base_model, peft_config)

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.total_steps,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        num_train_epochs=1
    )

    # Add custom attributes needed by the trainer
    training_args.model_name = args.model_name
    training_args.max_model_len = args.max_model_len
    training_args.max_lora_rank = args.outer_lora_rank

    # Initialize the RL trainer
    trainer = RLOOSelfEditTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        train_dataset=train_dataset,
        data_collator=simple_data_collator,
        optimizers=(None, None),
        callbacks=None,
        num_generations=args.num_generations,
        inner_temperature=args.inner_temperature,
        inner_top_p=args.inner_top_p,
        inner_max_tokens=args.inner_max_tokens,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Train the model
    with PartialState().local_main_process_first():
        print("Starting training...")
        trainer.train()
        print("Training finished.")

    # Save the final model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")
