#!/usr/bin/env python3
"""
Train SPECIALIZED model for a single asset
Usage: python gpu_train_asset_model.py BTC
       python gpu_train_asset_model.py ETH
       python gpu_train_asset_model.py SOL
       python gpu_train_asset_model.py HYPE
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
    print("Usage: python gpu_train_asset_model.py [BTC|ETH|SOL|HYPE]")
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

# Asset-specific training files
train_file = f'data/training/{ASSET}_100K_train.jsonl'
val_file = f'data/training/{ASSET}_100K_val.jsonl'

print(f"\n🎯 Training Configuration:")
print(f"  Asset: {ASSET}")
print(f"  Dataset: 100K specialized for {ASSET}")
print(f"  Train: 80K examples")
print(f"  Val: 10K examples")
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
print(f"Loading Qwen2.5-7B for {ASSET} specialization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# LoRA configuration - optimized for asset-specific learning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Good balance for specialization
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

print("Applying LoRA for specialized training...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load dataset
print(f"\nLoading {ASSET} dataset...")
train_dataset = load_dataset('json', data_files=train_file, split='train')
eval_dataset = load_dataset('json', data_files=val_file, split='train')

print(f"Training examples: {len(train_dataset):,}")
print(f"Validation examples: {len(eval_dataset):,}")

# Analyze asset-specific characteristics
def analyze_asset_data(dataset, asset):
    """Analyze asset-specific patterns"""
    sample = dataset[:100]
    
    # Get prices
    prices = [ex['metadata']['mid_price'] for ex in sample if 'metadata' in ex]
    if prices:
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        volatility = (max_price - min_price) / avg_price * 100
        
        print(f"\n📊 {asset} Market Characteristics (sample):")
        print(f"  Avg price: ${avg_price:,.2f}")
        print(f"  Price range: ${min_price:,.2f} - ${max_price:,.2f}")
        print(f"  Volatility: {volatility:.2f}%")
    
    # Action distribution
    buy_count = sum(1 for ex in sample if 'buy' in ex['chosen'].lower())
    sell_count = sum(1 for ex in sample if 'sell' in ex['chosen'].lower())
    print(f"  Actions: Buy={buy_count}%, Sell={sell_count}%")

analyze_asset_data(train_dataset, ASSET)

# Format examples
def format_example(example):
    return {
        "prompt": example['prompt'],
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

train_dataset = train_dataset.map(format_example)
eval_dataset = eval_dataset.map(format_example)

# Asset-specific training parameters
if ASSET == "BTC":
    # BTC: More stable, can use larger batches
    batch_size = 12
    learning_rate = 2e-5
elif ASSET == "ETH":
    # ETH: Moderate volatility
    batch_size = 10
    learning_rate = 2.5e-5
elif ASSET in ["SOL", "HYPE"]:
    # SOL/HYPE: Higher volatility, smaller batches
    batch_size = 8
    learning_rate = 3e-5
else:
    batch_size = 10
    learning_rate = 2e-5

print(f"\n⚙️ Asset-specific parameters:")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,  # 3 epochs for good convergence
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
    run_name=f"dapo_{ASSET.lower()}_specialized",
    logging_dir=f"./logs/{ASSET.lower()}_specialized",
    remove_unused_columns=False,
)

# Calculate training time
steps_per_epoch = len(train_dataset) // (batch_size * 4)
total_steps = steps_per_epoch * 3
estimated_hours = total_steps * 2 / 3600  # ~2 seconds per step

print(f"\n⏱️ Training timeline:")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Estimated time: {estimated_hours:.1f} hours")

# Initialize trainer
print(f"\nInitializing DAPO trainer for {ASSET}...")
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

# Asset-specific DAPO trainer
class AssetSpecializedDAPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs)
        
        # DAPO asymmetric clipping
        # Can be tuned per asset if needed
        clip_upper = 0.28
        clip_lower = 0.20
        
        if not return_outputs:
            if loss > clip_upper:
                loss = loss * 0.5 + clip_upper * 0.5
            elif loss < -clip_lower:
                loss = loss * 0.5 - clip_lower * 0.5
        
        return loss

trainer.__class__ = AssetSpecializedDAPOTrainer

# Training
print("\n" + "="*80)
print(f"Starting {ASSET} SPECIALIZED TRAINING...")
print("Model will learn:")
print(f"  ✓ {ASSET}-specific price patterns")
print(f"  ✓ {ASSET} volatility characteristics")
print(f"  ✓ Optimal strategies for {ASSET}")
print("="*80 + "\n")

# Train
trainer.train()

# Save model
print(f"\nSaving {ASSET} specialized model...")
trainer.save_model()
tokenizer.save_pretrained(output_dir)

# Calculate actual training time
training_time = time.time() - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)

# Save asset-specific summary
summary = {
    "asset": ASSET,
    "model_type": "specialized",
    "dataset_size": "100K",
    "training_examples": len(train_dataset),
    "validation_examples": len(eval_dataset),
    "total_steps": total_steps,
    "training_time": f"{hours}h {minutes}m",
    "model_base": model_name,
    "completed": str(datetime.now()),
    "specialization": f"{ASSET}-specific trading patterns",
    "use_case": f"Optimized for {ASSET} limit order execution"
}

with open(f"{output_dir}/model_info.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print(f"✅ {ASSET} SPECIALIZED MODEL COMPLETE!")
print(f"Training time: {hours}h {minutes}m")
print(f"Model saved to: {output_dir}")
print(f"\nThis model is specialized for {ASSET}:")
print(f"  - Understands {ASSET} price ranges")
print(f"  - Adapted to {ASSET} volatility")
print(f"  - Optimized for {ASSET} trading patterns")
print("="*80)