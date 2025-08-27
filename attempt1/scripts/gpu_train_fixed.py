#!/usr/bin/env python3
"""
Fixed training script for specialized asset models
"""

import sys
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer
import json
from datetime import datetime
import time

# Get asset from command line
if len(sys.argv) != 2 or sys.argv[1] not in ["BTC", "ETH", "SOL", "HYPE"]:
    print("Usage: python gpu_train_fixed.py [BTC|ETH|SOL|HYPE]")
    sys.exit(1)

ASSET = sys.argv[1]

print("="*80)
print(f"{ASSET} SPECIALIZED MODEL TRAINING")
print(f"Started: {datetime.now()}")
print("="*80)

start_time = time.time()

# Configuration
model_name = "Qwen/Qwen2.5-7B"
output_dir = f"./models/dapo_{ASSET.lower()}_specialized"

# Training files
train_file = f'data/training/{ASSET}_100K_train.jsonl'
val_file = f'data/training/{ASSET}_100K_val.jsonl'

print(f"\n🎯 Configuration:")
print(f"  Asset: {ASSET}")
print(f"  Model: {model_name}")
print(f"  Output: {output_dir}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
print(f"Loading Qwen2.5-7B...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

print("Applying LoRA...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load dataset
print(f"\nLoading {ASSET} dataset...")
train_dataset = load_dataset('json', data_files=train_file, split='train')
eval_dataset = load_dataset('json', data_files=val_file, split='train')

print(f"Training examples: {len(train_dataset):,}")
print(f"Validation examples: {len(eval_dataset):,}")

# Format for DPO
def format_example(example):
    return {
        "prompt": example.get('prompt', ''),
        "chosen": example.get('chosen', ''),
        "rejected": example.get('rejected', '')
    }

train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_example, remove_columns=eval_dataset.column_names)

# Asset-specific parameters
batch_sizes = {"BTC": 12, "ETH": 10, "SOL": 8, "HYPE": 8}
learning_rates = {"BTC": 2e-5, "ETH": 2.5e-5, "SOL": 3e-5, "HYPE": 3e-5}

batch_size = batch_sizes.get(ASSET, 10)
learning_rate = learning_rates.get(ASSET, 2e-5)

print(f"\n⚙️ Training parameters:")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Epochs: 3")

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=250,
    eval_strategy="steps",
    eval_steps=250,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    run_name=f"dapo_{ASSET.lower()}",
    logging_dir=f"./logs/{ASSET.lower()}",
    remove_unused_columns=False,
)

# Calculate steps
steps_per_epoch = len(train_dataset) // (batch_size * 4)
total_steps = steps_per_epoch * 3
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Est. time: {total_steps * 2 / 3600:.1f} hours")

# Initialize trainer
print(f"\nInitializing DAPO trainer...")
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will use model copy as reference
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # New API uses processing_class
    beta=0.1,
)

# DAPO with asymmetric clipping
class DAPOTrainerFixed(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs)
        if not return_outputs and isinstance(loss, torch.Tensor):
            # Asymmetric clipping
            if loss > 0.28:
                loss = loss * 0.5 + 0.14
            elif loss < -0.20:
                loss = loss * 0.5 - 0.10
        return loss

trainer.__class__ = DAPOTrainerFixed

# Start training
print("\n" + "="*80)
print(f"Starting {ASSET} training...")
print("="*80 + "\n")

trainer.train()

# Save model
print(f"\nSaving {ASSET} model...")
trainer.save_model()
tokenizer.save_pretrained(output_dir)

# Training complete
training_time = time.time() - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)

print("\n" + "="*80)
print(f"✅ {ASSET} TRAINING COMPLETE!")
print(f"Time: {hours}h {minutes}m")
print(f"Model saved to: {output_dir}")
print("="*80)