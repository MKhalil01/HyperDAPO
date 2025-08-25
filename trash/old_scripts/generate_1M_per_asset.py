#!/usr/bin/env python3
"""
Generate 1M examples per asset class
Total: 4M examples across BTC, ETH, SOL, HYPE
Can be sampled down if needed for training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.data_generator import TrainingDataGenerator
import numpy as np
from datetime import datetime
import random
import gc

print("="*60)
print("1M PER ASSET DATASET GENERATION")
print("="*60)
print(f"Start time: {datetime.now()}")
print("Target: 1M examples per asset (4M total)")
print("="*60)

generator = TrainingDataGenerator()

assets = ["BTC", "ETH", "SOL", "HYPE"]
target_per_asset = 1_000_000

# Process each asset independently
for asset in assets:
    print(f"\n{'='*60}")
    print(f"Processing {asset} - Target: {target_per_asset:,} examples")
    print('='*60)
    
    asset_examples = []
    
    # Use all available data
    date_ranges = [
        ("20250420", "20250430"),  # April
        ("20250501", "20250531"),  # May
        ("20250601", "20250630"),  # June
        ("20250701", "20250731"),  # July
        ("20250801", "20250819"),  # August
    ]
    
    for start_date, end_date in date_ranges:
        print(f"\n  Generating {asset}: {start_date} to {end_date}")
        
        try:
            examples = generator.generate_training_examples(
                symbol=asset,
                start_date=start_date,
                end_date=end_date,
                lookforward_minutes=5
            )
            
            if examples:
                asset_examples.extend(examples)
                print(f"    Generated {len(examples):,} examples")
                print(f"    Asset total so far: {len(asset_examples):,}")
                
                # If we already have enough, stop
                if len(asset_examples) >= target_per_asset:
                    print(f"    ✓ Reached target of {target_per_asset:,}")
                    break
                    
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            continue
    
    # Sample down to exactly 1M if we have more
    if len(asset_examples) > target_per_asset:
        print(f"\n  Sampling exactly {target_per_asset:,} from {len(asset_examples):,}")
        asset_examples = random.sample(asset_examples, target_per_asset)
    
    print(f"\n  Final {asset} dataset: {len(asset_examples):,} examples")
    
    # Calculate statistics
    if asset_examples:
        sample = random.sample(asset_examples, min(1000, len(asset_examples)))
        chosen_rewards = [e['chosen_reward'] for e in sample]
        rejected_rewards = [e['rejected_reward'] for e in sample]
        
        print(f"  Quality metrics:")
        print(f"    Avg chosen reward: {np.mean(chosen_rewards):.3f}")
        print(f"    Avg rejected reward: {np.mean(rejected_rewards):.3f}")
        print(f"    Avg reward diff: {np.mean([c-r for c,r in zip(chosen_rewards, rejected_rewards)]):.3f}")
    
    # Create splits for this asset
    print(f"\n  Creating train/val/test splits for {asset}...")
    splits = generator.create_train_val_test_split(asset_examples)
    
    # Save splits
    for split_name, split_data in splits.items():
        output_file = f"/Volumes/Extreme Pro/aaa/{asset}_1M_{split_name}.json"
        print(f"  💾 Saving {asset} {split_name}: {len(split_data):,} examples")
        generator.save_training_data(split_data, output_file)
    
    # Clear memory before processing next asset
    del asset_examples
    gc.collect()
    
    print(f"\n  ✅ {asset} complete!")

print("\n" + "="*60)
print("COMBINING ALL ASSETS")
print("="*60)

# Now create combined datasets
print("\nCombining all assets into unified dataset...")

all_train = []
all_val = []
all_test = []

for asset in assets:
    print(f"\nLoading {asset} splits...")
    
    # Load train
    train_file = f"/Volumes/Extreme Pro/aaa/{asset}_1M_train.json"
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            import json
            data = json.load(f)
            all_train.extend(data)
            print(f"  Loaded {len(data):,} training examples")
    
    # Load val
    val_file = f"/Volumes/Extreme Pro/aaa/{asset}_1M_val.json"
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            data = json.load(f)
            all_val.extend(data)
            print(f"  Loaded {len(data):,} validation examples")
    
    # Load test
    test_file = f"/Volumes/Extreme Pro/aaa/{asset}_1M_test.json"
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            data = json.load(f)
            all_test.extend(data)
            print(f"  Loaded {len(data):,} test examples")

# Shuffle combined datasets
print("\nShuffling combined datasets...")
random.shuffle(all_train)
random.shuffle(all_val)
random.shuffle(all_test)

# Save combined datasets
print("\nSaving combined datasets...")
generator.save_training_data(all_train, "/Volumes/Extreme Pro/aaa/COMBINED_4M_train.json")
generator.save_training_data(all_val, "/Volumes/Extreme Pro/aaa/COMBINED_4M_val.json")
generator.save_training_data(all_test, "/Volumes/Extreme Pro/aaa/COMBINED_4M_test.json")

print("\n" + "="*60)
print("✅ DATASET GENERATION COMPLETE!")
print("="*60)
print(f"Total training examples: {len(all_train):,}")
print(f"Total validation examples: {len(all_val):,}")
print(f"Total test examples: {len(all_test):,}")
print(f"Grand total: {len(all_train) + len(all_val) + len(all_test):,}")
print(f"\nEstimated training time for full 4M: ~112 hours (4.7 days)")
print(f"Estimated training time if sampled to 2M: ~56 hours (2.3 days)")
print(f"\nEnd time: {datetime.now()}")
print("\nDatasets saved to: /Volumes/Extreme Pro/aaa/")
print("\nYou can now:")
print("1. Use full 4M dataset for maximum data")
print("2. Sample down to 2M for 2-day training")
print("3. Use individual asset datasets for specialized models")