#!/usr/bin/env python3
"""
Train models with 1M examples for thesis comparison
Compare performance vs 100K models
"""

import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
from trl import DPOTrainer, DPOConfig
from datetime import datetime

def load_1m_dataset(asset):
    """Load 1M example dataset"""
    print(f"\n📊 Loading 1M {asset} dataset...")
    
    # Load JSON files
    with open(f'data/training/{asset}_1M_train.json', 'r') as f:
        train_data = json.load(f)
    
    with open(f'data/training/{asset}_1M_val.json', 'r') as f:
        val_data = json.load(f)
    
    # Convert to Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"  Training examples: {len(train_dataset):,}")
    print(f"  Validation examples: {len(val_dataset):,}")
    
    return train_dataset, val_dataset

def train_1m_model(asset):
    """Train model with 1M examples"""
    
    print(f"""
{'='*80}
Training {asset} Model with 1M Examples
{'='*80}
    """)
    
    # Load datasets
    train_dataset, val_dataset = load_1m_dataset(asset)
    
    # Load tokenizer
    print("\n🤖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    print("\n🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    # LoRA configuration - same as 100K for fair comparison
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Training arguments - adjusted for larger dataset
    training_args = DPOConfig(
        output_dir=f"./models/{asset.lower()}_1m_model",
        num_train_epochs=1,  # Keep 1 epoch for fair comparison
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=100,
        save_steps=5000,  # Save less frequently due to larger dataset
        eval_strategy="steps",  # Use eval_strategy instead
        eval_steps=5000,
        bf16=True,
        push_to_hub=False,
        report_to="none",
        max_grad_norm=0.3,
        remove_unused_columns=False,
        beta=0.1,  # DPO beta
        # DAPO settings
        loss_type="sigmoid",
    )
    
    # Initialize DPO trainer
    print("\n🚀 Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
    
    # Start training
    print(f"\n⏰ Starting training at {datetime.now()}")
    print(f"   Expected duration: ~25-30 hours for 1M examples")
    print(f"   This is 10x the data of the 100K model")
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    # Save the model
    print(f"\n💾 Saving {asset} 1M model...")
    trainer.save_model(f"./models/{asset.lower()}_1m_model")
    
    # Calculate training time
    training_duration = (end_time - start_time).total_seconds() / 3600
    
    # Save training metrics
    metrics = {
        "asset": asset,
        "dataset_size": "1M",
        "training_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "training_duration_hours": training_duration,
        "final_loss": trainer.state.log_history[-1].get('loss', 'N/A'),
        "timestamp": str(datetime.now())
    }
    
    with open(f"{asset.lower()}_1m_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ {asset} 1M model training complete!")
    print(f"   Training time: {training_duration:.2f} hours")
    print(f"   Model saved to: ./models/{asset.lower()}_1m_model")
    
    return metrics

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_1m_models.py [BTC|ETH]")
        sys.exit(1)
    
    asset = sys.argv[1].upper()
    if asset not in ["BTC", "ETH"]:
        print("Asset must be BTC or ETH")
        sys.exit(1)
    
    # Train the model
    metrics = train_1m_model(asset)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Asset: {asset}")
    print(f"Dataset: 1M examples")
    print(f"Duration: {metrics['training_duration_hours']:.2f} hours")
    print(f"Final Loss: {metrics['final_loss']}")
    print("\nNext steps:")
    print("1. Train the other asset model")
    print("2. Compare with 100K model performance")
    print("3. Run backtesting on both")

if __name__ == "__main__":
    main()