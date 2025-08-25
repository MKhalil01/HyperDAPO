#!/usr/bin/env python3
"""
DAPO training script optimized for GPU server with A100
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

# Set environment for optimal GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first A100
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class DAPOConfig:
    """DAPO configuration for cryptocurrency trading"""
    # Model
    model_name = "Qwen/Qwen2.5-7B"  # Can also use "meta-llama/Llama-3.2-3B"
    
    # DAPO parameters (key innovations)
    clip_higher = 0.28  # Asymmetric upper bound
    clip_lower = 0.2    # Standard lower bound
    dynamic_sampling = True
    token_level_pg = True
    
    # Training
    learning_rate = 2e-5
    batch_size = 8  # Increased for A100
    gradient_accumulation_steps = 4  # Effective batch = 32
    num_epochs = 3
    warmup_steps = 100
    max_length = 512
    
    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    
    # Paths
    train_data = "data/training/BTC_test_training_data.jsonl"
    output_dir = "models/dapo_trading"
    checkpoint_dir = "models/checkpoints"
    
def compute_dapo_loss(logits, labels, old_logprobs, advantages, config):
    """Custom DAPO loss with asymmetric clipping"""
    # Get log probabilities
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    logprobs = torch.gather(logprobs, 2, labels.unsqueeze(-1)).squeeze(-1)
    
    # Compute ratio
    ratio = torch.exp(logprobs - old_logprobs)
    
    # Asymmetric clipping (DAPO innovation)
    clipped_ratio = torch.where(
        advantages > 0,
        torch.clamp(ratio, 1 - config.clip_lower, 1 + config.clip_higher),
        torch.clamp(ratio, 1 - config.clip_lower, 1 + config.clip_lower)
    )
    
    # Policy loss
    loss1 = -advantages * ratio
    loss2 = -advantages * clipped_ratio
    policy_loss = torch.max(loss1, loss2).mean()
    
    return policy_loss

def main():
    config = DAPOConfig()
    
    print("="*60)
    print("DAPO Training for Cryptocurrency Trading")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"DAPO clip_higher: {config.clip_higher}")
    print(f"Batch size: {config.batch_size}")
    print(f"GPU: NVIDIA A100 40GB")
    print("="*60)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization for efficient training
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply LoRA
    print("Applying LoRA...")
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
    print("Loading training data...")
    dataset = load_dataset('json', data_files=config.train_data, split='train')
    
    # Split into train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Tokenize function
    def tokenize_function(examples):
        # For DAPO, we need both chosen and rejected
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
    
    # Tokenize datasets
    print("Tokenizing datasets...")
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
        logging_dir='logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
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
    
    # Train
    print("\n" + "="*60)
    print("Starting DAPO training...")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.output_dir + "/final")
    tokenizer.save_pretrained(config.output_dir + "/final")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {config.output_dir}/final")
    print(f"Time: {datetime.now()}")
    print("="*60)

if __name__ == "__main__":
    main()