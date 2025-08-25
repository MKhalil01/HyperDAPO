"""
DAPO training configuration for LLM-based trading
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for base LLM model"""
    model_name: str = "Qwen/Qwen2.5-7B"  # Primary choice
    # model_name: str = "meta-llama/Llama-3.2-3B"  # Alternative
    use_flash_attention: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = True  # For training on limited VRAM
    device_map: str = "auto"
    torch_dtype: str = "float16"
    max_length: int = 1024
    

@dataclass 
class DAPOConfig:
    """DAPO-specific hyperparameters"""
    # Core DAPO parameters (from paper)
    clip_higher: float = 0.28  # Increased from standard 0.2
    clip_lower: float = 0.2    # Standard PPO clip
    dynamic_sampling: bool = True  # Filter prompts with acc=0 or acc=1
    token_level_pg: bool = True  # Token-level policy gradient
    overlong_reward_shaping: bool = True  # For long outputs
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 4  # Small for GPU memory
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # DAPO-specific
    kl_penalty: float = 0.01  # KL divergence penalty
    entropy_coef: float = 0.01  # Entropy bonus
    value_loss_coef: float = 0.5  # Value function loss weight
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    

@dataclass
class LoRAConfig:
    """LoRA configuration for efficient fine-tuning"""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Will be set based on model
    modules_to_save: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default for Qwen models
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
            

@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Model settings
    model: ModelConfig = None
    dapo: DAPOConfig = None
    lora: LoRAConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.dapo is None:
            self.dapo = DAPOConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
    
    # Data settings
    train_data_path: str = "data/training/combined_train.jsonl"
    val_data_path: str = "data/training/combined_validation.jsonl"
    test_data_path: str = "data/training/combined_test.jsonl"
    
    # Training settings
    output_dir: str = "models/dapo_trading"
    checkpoint_dir: str = "models/checkpoints"
    logging_dir: str = "logs/training"
    
    # Hardware settings
    num_gpus: int = 1
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Evaluation settings
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_reward"
    greater_is_better: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    

class DAPOTrainer:
    """Custom trainer for DAPO optimization"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        """Compute GAE advantages for DAPO"""
        advantages = []
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            last_advantage = delta + gamma * lam * last_advantage
            advantages.insert(0, last_advantage)
        
        return torch.tensor(advantages)
    
    def compute_policy_loss(self, logprobs, old_logprobs, advantages, clip_higher, clip_lower):
        """DAPO policy loss with asymmetric clipping"""
        ratio = torch.exp(logprobs - old_logprobs)
        
        # Asymmetric clipping (DAPO innovation)
        clipped_ratio = torch.where(
            advantages > 0,
            torch.clamp(ratio, 1 - clip_lower, 1 + clip_higher),  # Higher upper bound
            torch.clamp(ratio, 1 - clip_lower, 1 + clip_lower)   # Standard for negative
        )
        
        loss1 = -advantages * ratio
        loss2 = -advantages * clipped_ratio
        
        return torch.max(loss1, loss2).mean()
    
    def dynamic_sampling_filter(self, batch_data):
        """Filter out prompts with accuracy = 0 or 1 (DAPO dynamic sampling)"""
        filtered_data = []
        
        for item in batch_data:
            # Check if this prompt has moderate difficulty
            if 0.1 < item.get('accuracy', 0.5) < 0.9:
                filtered_data.append(item)
        
        return filtered_data if filtered_data else batch_data  # Return original if all filtered
    

def get_training_args(config: TrainingConfig):
    """Get HuggingFace TrainingArguments for DAPO"""
    from transformers import TrainingArguments
    
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.dapo.num_epochs,
        per_device_train_batch_size=config.dapo.batch_size,
        per_device_eval_batch_size=config.dapo.batch_size,
        gradient_accumulation_steps=config.dapo.gradient_accumulation_steps,
        warmup_steps=config.dapo.warmup_steps,
        weight_decay=config.dapo.weight_decay,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        evaluation_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.dapo.learning_rate,
        max_grad_norm=config.dapo.max_grad_norm,
        report_to=["tensorboard"],
        push_to_hub=False,
    )


def create_training_script():
    """Generate the training script for cloud execution"""
    script = '''#!/usr/bin/env python3
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
'''
    
    return script


if __name__ == "__main__":
    # Test configuration
    config = TrainingConfig()
    
    print("DAPO Trading Configuration")
    print("="*50)
    print(f"Model: {config.model.model_name}")
    print(f"DAPO clip_higher: {config.dapo.clip_higher}")
    print(f"Dynamic sampling: {config.dapo.dynamic_sampling}")
    print(f"Batch size: {config.dapo.batch_size}")
    print(f"Learning rate: {config.dapo.learning_rate}")
    print(f"LoRA rank: {config.lora.r}")
    
    # Save training script
    script = create_training_script()
    with open("scripts/train_dapo.py", "w") as f:
        f.write(script)
    print("\nTraining script saved to scripts/train_dapo.py")
    
    print("\nNext steps:")
    print("1. Upload training data to GPU cloud")
    print("2. Install dependencies: transformers, peft, accelerate, datasets")
    print("3. Run: python train_dapo.py")
    print("4. Monitor tensorboard for training progress")