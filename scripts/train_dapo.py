#!/usr/bin/env python3
"""
DAPO training script for cryptocurrency trading LLM
Run this on GPU cloud (Vast.ai, RunPod, etc.)
"""

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
from train_config import TrainingConfig, get_training_args, DAPOTrainer

def main():
    # Initialize configuration
    config = TrainingConfig()
    
    print(f"Loading model: {config.model.model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        load_in_4bit=config.model.load_in_4bit,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map=config.model.device_map
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print("Loading training data...")
    train_dataset = load_dataset('json', data_files=config.train_data_path, split='train')
    val_dataset = load_dataset('json', data_files=config.val_data_path, split='train')
    
    # Tokenize datasets
    def tokenize_function(examples):
        # Combine prompt and chosen response
        texts = [p + " " + c for p, c in zip(examples['prompt'], examples['chosen'])]
        return tokenizer(texts, truncation=True, padding='max_length', max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Custom DAPO trainer
    dapo_trainer = DAPOTrainer(config)
    
    # Training arguments
    training_args = get_training_args(config)
    
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
    print("Starting DAPO training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(config.output_dir + "/final")
    tokenizer.save_pretrained(config.output_dir + "/final")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
