#!/usr/bin/env python3
"""
DAPO training script for GPU 1 with FULL dataset
Parallel experiment with complete 4-month, 4-asset data
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np
from datetime import datetime

# Force GPU 1 usage
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class DAPOConfig:
    """DAPO configuration - identical to GPU 0 for fair comparison"""
    model_name = "Qwen/Qwen2.5-7B"
    
    # DAPO parameters (same as GPU 0)
    clip_higher = 0.28
    clip_lower = 0.2
    dynamic_sampling = True
    token_level_pg = True
    
    # Training (same hyperparameters)
    learning_rate = 2e-5
    batch_size = 8
    gradient_accumulation_steps = 4
    num_epochs = 3
    warmup_steps = 100
    max_length = 512
    
    # LoRA (same configuration)
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    
    # Paths (different data file)
    train_data = "data/training/medium_train.jsonl"  # Full dataset
    output_dir = "models/dapo_full"
    checkpoint_dir = "models/checkpoints_full"
    log_file = "training_gpu1_output.log"

def log_message(message, config):
    """Log messages to file and console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    with open(config.log_file, 'a') as f:
        f.write(log_entry + '\n')

def main():
    config = DAPOConfig()
    
    log_message("="*60, config)
    log_message("DAPO FULL DATASET TRAINING - GPU 1", config)
    log_message("="*60, config)
    log_message(f"Model: {config.model_name}", config)
    log_message(f"Dataset: Full 4-month, 4-asset data", config)
    log_message(f"GPU: NVIDIA A100 40GB (GPU 1)", config)
    log_message(f"Experiment: Parallel comparison with GPU 0 test run", config)
    log_message("="*60, config)
    
    # Load tokenizer
    log_message("Loading tokenizer...", config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    log_message("Loading model on GPU 1...", config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply LoRA
    log_message("Applying LoRA configuration...", config)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    log_message("Loading FULL training dataset...", config)
    dataset = load_dataset('json', data_files=config.train_data, split='train')
    
    # Split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    log_message(f"Train examples: {len(train_dataset):,}", config)
    log_message(f"Validation examples: {len(val_dataset):,}", config)
    log_message(f"This is {len(train_dataset) / 159704:.1f}x larger than GPU 0 dataset", config)
    
    # Tokenize
    def tokenize_function(examples):
        if 'prompt' in examples and 'chosen' in examples:
            texts = [p + " " + c for p, c in zip(examples['prompt'], examples['chosen'])]
        else:
            texts = examples['text'] if 'text' in examples else examples['content']
        
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=config.max_length
        )
    
    log_message("Tokenizing datasets...", config)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_dir='logs_gpu1',
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,  # Less frequent for larger dataset
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Log comparison info
    log_message("\n" + "="*60, config)
    log_message("EXPERIMENT COMPARISON", config)
    log_message("="*60, config)
    log_message("GPU 0 (Test Run):", config)
    log_message("  - 2 days BTC data only", config)
    log_message("  - 159,704 examples", config)
    log_message("  - 97MB dataset", config)
    log_message("", config)
    log_message("GPU 1 (Full Dataset):", config)
    log_message("  - 4 months data", config)
    log_message("  - BTC, ETH, SOL, HYPE", config)
    log_message(f"  - {len(train_dataset):,} examples", config)
    log_message("  - ~5GB dataset", config)
    log_message("="*60, config)
    
    # Train
    log_message(f"\nStarting DAPO training at {datetime.now()}", config)
    trainer.train()
    
    # Save final model
    log_message("\nSaving final model...", config)
    trainer.save_model(config.output_dir + "/final")
    tokenizer.save_pretrained(config.output_dir + "/final")
    
    # Final summary
    log_message("\n" + "="*60, config)
    log_message("TRAINING COMPLETE!", config)
    log_message(f"Model saved to: {config.output_dir}/final", config)
    log_message(f"Time: {datetime.now()}", config)
    log_message("="*60, config)
    
    # Save metrics for comparison
    metrics = {
        'gpu': 1,
        'dataset': 'full',
        'examples': len(train_dataset),
        'final_loss': trainer.state.best_metric,
        'total_steps': trainer.state.global_step,
        'best_checkpoint': trainer.state.best_model_checkpoint
    }
    
    import json
    with open('gpu1_final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    log_message(f"Final metrics saved to gpu1_final_metrics.json", config)

if __name__ == "__main__":
    main()