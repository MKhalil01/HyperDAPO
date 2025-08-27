#!/usr/bin/env python3
"""
Quick Training Script for 100K Test Dataset
For rapid iteration and validation
Expected training time: 3-4 hours
"""

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

print("="*80)
print(f"100K TEST DATASET TRAINING")
print(f"Started: {datetime.now()}")
print("Purpose: Quick iteration and validation")
print("="*80)

start_time = time.time()

# Configuration
model_name = "Qwen/Qwen2.5-7B"
output_dir = "./models/dapo_test_100k"

# Training files
train_file = 'data/training/TEST_100K_train.jsonl'
val_file = 'data/training/TEST_100K_val.jsonl'

print(f"\nTraining configuration:")
print(f"  Dataset: 100K test dataset")
print(f"  Train: 80K examples")
print(f"  Val: 10K examples")
print(f"  Batch size: 16 (larger for faster training)")
print(f"  Epochs: 2 (less epochs for quick test)")

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
print("Loading Qwen2.5-7B with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# LoRA configuration - slightly smaller for faster training
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Smaller rank for faster training
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

print("Applying LoRA...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load dataset
print(f"\nLoading 100K test dataset...")
train_dataset = load_dataset('json', data_files=train_file, split='train')
eval_dataset = load_dataset('json', data_files=val_file, split='train')

print(f"Training examples: {len(train_dataset):,}")
print(f"Validation examples: {len(eval_dataset):,}")

# Quick analysis
def quick_analysis(dataset):
    sample = dataset[:100]
    buy_count = sum(1 for ex in sample if 'buy' in ex['chosen'].lower())
    sell_count = sum(1 for ex in sample if 'sell' in ex['chosen'].lower())
    print(f"  Sample (n=100): Buy={buy_count}%, Sell={sell_count}%")

print("\nDataset composition:")
quick_analysis(train_dataset)

# Format examples
def format_example(example):
    return {
        "prompt": example['prompt'],
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

train_dataset = train_dataset.map(format_example)
eval_dataset = eval_dataset.map(format_example)

# Training arguments optimized for speed
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,  # Just 2 epochs for quick test
    per_device_train_batch_size=16,  # Larger batch for speed
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Less accumulation
    learning_rate=3e-5,  # Slightly higher LR
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,  # Less warmup
    logging_steps=10,
    save_steps=200,  # Save more frequently
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    run_name="dapo_test_100k",
    logging_dir="./logs/test_100k",
    remove_unused_columns=False,
)

# Calculate expected training time
steps_per_epoch = len(train_dataset) // (16 * 2)  # batch_size * gradient_accumulation
total_steps = steps_per_epoch * 2  # 2 epochs
estimated_time = total_steps * 1.5 / 3600  # ~1.5 seconds per step

print(f"\nTraining steps:")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Estimated time: {estimated_time:.1f} hours")

# Initialize trainer
print("\nInitializing DAPO trainer...")
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    beta=0.1,
    loss_type="sigmoid",
    max_prompt_length=384,
    max_length=512,
)

# Enhanced DAPO with asymmetric clipping
class QuickDAPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs)
        
        # DAPO asymmetric clipping
        if not return_outputs:
            if loss > 0.28:
                loss = loss * 0.5 + 0.14
            elif loss < -0.20:
                loss = loss * 0.5 - 0.10
        
        return loss

trainer.__class__ = QuickDAPOTrainer

# Training
print("\n" + "="*80)
print("Starting 100K TEST TRAINING...")
print("Features:")
print("  ✓ Buy and Sell orders")
print("  ✓ 3 user objectives")
print("  ✓ Market microstructure")
print("  ✓ Real prices")
print("="*80 + "\n")

# Train
trainer.train()

# Save model
print("\nSaving model...")
trainer.save_model()
tokenizer.save_pretrained(output_dir)

# Calculate actual training time
training_time = time.time() - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)

# Save training summary
summary = {
    "dataset": "100K test dataset",
    "training_examples": len(train_dataset),
    "validation_examples": len(eval_dataset),
    "total_steps": total_steps,
    "training_time": f"{hours}h {minutes}m",
    "model_base": model_name,
    "completed": str(datetime.now()),
    "purpose": "Quick iteration and validation",
    "next_steps": [
        "Evaluate on test set",
        "Run inference examples",
        "Check loss curves",
        "Decide on full training"
    ]
}

with open(f"{output_dir}/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ 100K TEST TRAINING COMPLETE!")
print(f"Training time: {hours}h {minutes}m")
print(f"Model saved to: {output_dir}")
print("\nNext steps:")
print("  1. Evaluate model performance")
print("  2. Test inference with sample prompts")
print("  3. If results good, proceed with larger dataset")
print("="*80)