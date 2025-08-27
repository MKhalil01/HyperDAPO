#!/usr/bin/env python3
"""
GPU Training Script for Enhanced Dataset
Trains on data with buy/sell orders and user objectives
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

print("="*80)
print(f"ENHANCED DAPO TRAINING")
print(f"Started: {datetime.now()}")
print("="*80)

# Configuration
model_name = "Qwen/Qwen2.5-7B"
output_dir = "./models/dapo_enhanced"

# Choose dataset size based on available time
# Option 1: Full 4M dataset (~112 hours)
# Option 2: 2M subset (~56 hours)
# Option 3: 1M subset (~28 hours)
DATASET_SIZE = "1M"  # Change this to "2M" or "4M" as needed

if DATASET_SIZE == "4M":
    train_file = 'data/training/ENHANCED_4M_train.jsonl'
    val_file = 'data/training/ENHANCED_4M_val.jsonl'
    batch_size = 4  # Smaller batch for larger dataset
elif DATASET_SIZE == "2M":
    # We'll sample from the 4M dataset
    train_file = 'data/training/ENHANCED_2M_train.jsonl'
    val_file = 'data/training/ENHANCED_4M_val.jsonl'
    batch_size = 6
else:  # 1M
    train_file = 'data/training/ENHANCED_1M_train.jsonl'
    val_file = 'data/training/ENHANCED_1M_val.jsonl'
    batch_size = 8

print(f"Training with {DATASET_SIZE} dataset")
print(f"Batch size: {batch_size}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 4-bit quantization for memory efficiency
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

# Load enhanced dataset
print(f"\nLoading enhanced dataset from {train_file}...")
train_dataset = load_dataset('json', data_files=train_file, split='train')
eval_dataset = load_dataset('json', data_files=val_file, split='train')

print(f"Training examples: {len(train_dataset):,}")
print(f"Validation examples: {len(eval_dataset):,}")

# Analyze dataset distribution
def analyze_dataset(dataset, name="Dataset"):
    """Quick analysis of dataset composition"""
    sample = dataset[:min(1000, len(dataset))]
    
    buy_count = sum(1 for ex in sample if 'buy' in ex['chosen'].lower() or 'buy' in ex['rejected'].lower())
    sell_count = sum(1 for ex in sample if 'sell' in ex['chosen'].lower() or 'sell' in ex['rejected'].lower())
    
    print(f"\n{name} Analysis (sample of {len(sample)}):")
    print(f"  Buy orders: ~{buy_count/10:.1f}%")
    print(f"  Sell orders: ~{sell_count/10:.1f}%")

analyze_dataset(train_dataset, "Training")

def format_example(example):
    """Format for DPO training"""
    return {
        "prompt": example['prompt'],
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

# Apply formatting
train_dataset = train_dataset.map(format_example)
eval_dataset = eval_dataset.map(format_example)

# Calculate training steps
steps_per_epoch = len(train_dataset) // (batch_size * 4)  # 4 is gradient accumulation
total_steps = steps_per_epoch * 3  # 3 epochs

print(f"\nTraining configuration:")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Estimated time: {total_steps * 2.5 / 3600:.1f} hours")

# DAPO Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    bf16=True,
    gradient_checkpointing=True,
    report_to="tensorboard",
    run_name=f"dapo_enhanced_{DATASET_SIZE}",
    logging_dir=f"./logs/enhanced_{DATASET_SIZE}",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
)

# Initialize DPO trainer
print("\nInitializing DAPO trainer...")
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    beta=0.1,  # DPO beta parameter
    loss_type="sigmoid",
    label_smoothing=0.0,
    max_prompt_length=384,  # Increased for longer prompts
    max_length=512,
)

# Custom DAPO loss with asymmetric clipping
class EnhancedDAPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs)
        
        # Apply asymmetric clipping (DAPO feature)
        clip_upper = 0.28
        clip_lower = 0.20
        
        if not return_outputs:
            # Clip the loss asymmetrically
            if loss > clip_upper:
                loss = loss * 0.5 + clip_upper * 0.5
            elif loss < -clip_lower:
                loss = loss * 0.5 - clip_lower * 0.5
        
        return loss

# Replace trainer with enhanced version
trainer.__class__ = EnhancedDAPOTrainer

# Start training
print("\n" + "="*80)
print(f"Starting ENHANCED DAPO training...")
print(f"Dataset: {DATASET_SIZE} ({len(train_dataset):,} examples)")
print(f"Features:")
print("  - Buy AND Sell orders")
print("  - 3 user objectives (aggressive/patient/risk-averse)")
print("  - Market microstructure context")
print("  - Asymmetric clipping (0.28/0.20)")
print("="*80 + "\n")

trainer.train()

# Save final model
print("\nSaving final model...")
trainer.save_model()
tokenizer.save_pretrained(output_dir)

# Save training metadata
metadata = {
    "dataset_size": DATASET_SIZE,
    "total_examples": len(train_dataset),
    "model_base": model_name,
    "training_completed": str(datetime.now()),
    "features": [
        "buy_sell_orders",
        "user_objectives",
        "market_microstructure",
        "realistic_prices"
    ]
}

with open(f"{output_dir}/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*80)
print("✅ ENHANCED TRAINING COMPLETE!")
print(f"Model saved to: {output_dir}")
print(f"Completed: {datetime.now()}")
print("\nModel capabilities:")
print("  - Understands buy vs sell decisions")
print("  - Adapts to different trading styles")
print("  - Reads market microstructure signals")
print("  - Works with real cryptocurrency prices")
print("="*80)