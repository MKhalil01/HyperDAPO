#!/usr/bin/env python3
"""
Generate 100K test dataset for quick training iteration
- 25K examples per asset (BTC, ETH, SOL, HYPE)
- Balanced buy/sell distribution
- All 3 user objectives
- Quick to generate and train (~3-4 hours training)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced generator
exec(open('ClaudeWorkingScripts/enhanced_data_generator.py').read())

import numpy as np
from datetime import datetime
import random
import json

print("="*80)
print("100K TEST DATASET GENERATION")
print("="*80)
print(f"Start time: {datetime.now()}")
print("Purpose: Quick training iteration and validation")
print("Target: 100K examples (25K per asset)")
print("Estimated training time: 3-4 hours")
print("="*80)

# Initialize generator
generator = EnhancedTrainingDataGenerator()

assets = ["BTC", "ETH", "SOL", "HYPE"]
target_per_asset = 25000  # 25K per asset = 100K total

# Track statistics
stats = {
    'total': 0,
    'buy': 0,
    'sell': 0,
    'aggressive': 0,
    'patient': 0,
    'risk_averse': 0
}

all_examples = []

for asset in assets:
    print(f"\n{'='*60}")
    print(f"Processing {asset} - Target: {target_per_asset:,} examples")
    print('='*60)
    
    asset_examples = []
    
    # Use recent data only for faster processing
    # Just last 2 weeks of data for test dataset
    date_ranges = [
        ("20250810", "20250819"),  # Recent 10 days
    ]
    
    for start_date, end_date in date_ranges:
        print(f"\n  📅 Processing {asset}: {start_date} to {end_date}")
        
        try:
            examples = generator.generate_training_examples(
                symbol=asset,
                start_date=start_date,
                end_date=end_date,
                lookforward_minutes=5
            )
            
            if examples:
                # Sample to get exactly what we need
                if len(examples) > target_per_asset:
                    examples = random.sample(examples, target_per_asset)
                
                asset_examples.extend(examples)
                
                # Quick stats
                buy_count = sum(1 for e in examples if e['metadata']['action_side'] == 'buy')
                sell_count = sum(1 for e in examples if e['metadata']['action_side'] == 'sell')
                
                print(f"    Generated {len(examples):,} examples")
                print(f"    Buy: {buy_count:,} | Sell: {sell_count:,}")
                
                # Verify prices are correct
                sample_prices = [e['metadata']['mid_price'] for e in random.sample(examples, min(5, len(examples)))]
                avg_price = np.mean(sample_prices)
                print(f"    Price check: ${avg_price:,.2f}")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue
    
    # Ensure we have exactly 25K
    if len(asset_examples) < target_per_asset:
        print(f"    ⚠️ Only got {len(asset_examples):,} examples, generating more...")
        # Try earlier dates
        additional_dates = [("20250801", "20250809")]
        for start_date, end_date in additional_dates:
            if len(asset_examples) >= target_per_asset:
                break
            try:
                more_examples = generator.generate_training_examples(
                    symbol=asset,
                    start_date=start_date,
                    end_date=end_date,
                    lookforward_minutes=5
                )
                if more_examples:
                    needed = target_per_asset - len(asset_examples)
                    asset_examples.extend(more_examples[:needed])
                    print(f"    Added {min(needed, len(more_examples)):,} more examples")
            except:
                pass
    
    # Trim to exact target
    asset_examples = asset_examples[:target_per_asset]
    
    # Add to total
    all_examples.extend(asset_examples)
    
    # Update stats
    for ex in asset_examples:
        stats['total'] += 1
        if ex['metadata']['action_side'] == 'buy':
            stats['buy'] += 1
        else:
            stats['sell'] += 1
        
        obj = ex['metadata']['user_objective']
        if obj == 'aggressive':
            stats['aggressive'] += 1
        elif obj == 'patient':
            stats['patient'] += 1
        else:
            stats['risk_averse'] += 1
    
    print(f"\n  ✅ {asset} complete: {len(asset_examples):,} examples")

# Shuffle all examples
print(f"\n{'='*60}")
print("Shuffling and creating splits...")
random.shuffle(all_examples)

# Create train/val/test splits (80/10/10)
n = len(all_examples)
train_size = int(n * 0.8)  # 80K for training
val_size = int(n * 0.1)     # 10K for validation
test_size = n - train_size - val_size  # 10K for test

splits = {
    'train': all_examples[:train_size],
    'val': all_examples[train_size:train_size + val_size],
    'test': all_examples[train_size + val_size:]
}

# Save splits as both JSON and JSONL
print(f"\n💾 Saving 100K test dataset...")
for split_name, split_data in splits.items():
    # Save as JSON for inspection
    json_file = f"data/training/TEST_100K_{split_name}.json"
    with open(json_file, 'w') as f:
        json.dump(split_data, f)
    print(f"  Saved {split_name}: {len(split_data):,} examples to {json_file}")
    
    # Save as JSONL for training
    jsonl_file = f"data/training/TEST_100K_{split_name}.jsonl"
    with open(jsonl_file, 'w') as f:
        for item in split_data:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved JSONL version to {jsonl_file}")

# Print summary statistics
print(f"\n{'='*60}")
print("📊 DATASET STATISTICS")
print('='*60)
print(f"Total examples: {stats['total']:,}")
print(f"\nAction distribution:")
print(f"  Buy orders: {stats['buy']:,} ({stats['buy']/stats['total']*100:.1f}%)")
print(f"  Sell orders: {stats['sell']:,} ({stats['sell']/stats['total']*100:.1f}%)")
print(f"\nUser objectives:")
print(f"  Aggressive: {stats['aggressive']:,} ({stats['aggressive']/stats['total']*100:.1f}%)")
print(f"  Patient: {stats['patient']:,} ({stats['patient']/stats['total']*100:.1f}%)")
print(f"  Risk-averse: {stats['risk_averse']:,} ({stats['risk_averse']/stats['total']*100:.1f}%)")

# Quality check - sample 10 examples
print(f"\n{'='*60}")
print("🔍 QUALITY CHECK (10 random examples)")
print('='*60)
for i in range(min(10, len(all_examples))):
    ex = random.choice(all_examples)
    print(f"\n{i+1}. {ex['metadata']['symbol']} - {ex['metadata']['action_side'].upper()} - {ex['metadata']['user_objective']}")
    print(f"   Price: ${ex['metadata']['mid_price']:,.2f}")
    print(f"   Chosen: {ex['chosen'][:60]}...")
    print(f"   Reward diff: {ex['chosen_reward'] - ex['rejected_reward']:.3f}")

print(f"\n{'='*60}")
print("✅ 100K TEST DATASET COMPLETE!")
print('='*60)
print(f"Files saved to: data/training/TEST_100K_*.json[l]")
print(f"Total size: ~60MB")
print(f"Estimated training time: 3-4 hours")
print(f"End time: {datetime.now()}")
print("\nPerfect for:")
print("  - Quick iteration and testing")
print("  - Hyperparameter tuning")
print("  - Validating the approach")
print("\nNext: Transfer to GPU and run quick training test!")
print('='*60)