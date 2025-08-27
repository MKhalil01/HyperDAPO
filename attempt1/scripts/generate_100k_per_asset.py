#!/usr/bin/env python3
"""
Generate 100K examples PER ASSET CLASS
For training specialized models for each cryptocurrency
- 100K examples each for BTC, ETH, SOL, HYPE
- Each will train a separate specialized model
- ~3-4 hours training per model
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
print("100K PER ASSET DATASET GENERATION")
print("="*80)
print(f"Start time: {datetime.now()}")
print("Purpose: Train SPECIALIZED models for each asset")
print("Target: 100K examples per asset (4 separate datasets)")
print("Estimated training time: 3-4 hours per model")
print("="*80)

# Initialize generator
generator = EnhancedTrainingDataGenerator()

assets = ["BTC", "ETH", "SOL", "HYPE"]
target_per_asset = 100000  # 100K per asset

for asset in assets:
    print(f"\n{'='*80}")
    print(f"GENERATING {asset} DATASET - 100K EXAMPLES")
    print('='*80)
    
    asset_examples = []
    asset_stats = {
        'buy': 0,
        'sell': 0,
        'aggressive': 0,
        'patient': 0,
        'risk_averse': 0
    }
    
    # Use last month of data for each asset
    date_ranges = [
        ("20250801", "20250810"),  # First 10 days of August
        ("20250811", "20250819"),  # Last 9 days of August
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
                asset_examples.extend(examples)
                
                # Stats
                buy_count = sum(1 for e in examples if e['metadata']['action_side'] == 'buy')
                sell_count = sum(1 for e in examples if e['metadata']['action_side'] == 'sell')
                
                print(f"    Generated {len(examples):,} examples")
                print(f"    Buy: {buy_count:,} | Sell: {sell_count:,}")
                print(f"    Total so far: {len(asset_examples):,}")
                
                # Price verification
                sample_prices = [e['metadata']['mid_price'] for e in random.sample(examples, min(5, len(examples)))]
                avg_price = np.mean(sample_prices)
                
                # Asset-specific price validation
                if asset == "BTC" and avg_price < 50000:
                    print(f"    ⚠️ WARNING: Low BTC prices! Avg: ${avg_price:,.2f}")
                elif asset == "ETH" and avg_price < 1000:
                    print(f"    ⚠️ WARNING: Low ETH prices! Avg: ${avg_price:,.2f}")
                elif asset == "SOL" and avg_price < 50:
                    print(f"    ⚠️ WARNING: Low SOL prices! Avg: ${avg_price:,.2f}")
                else:
                    print(f"    ✓ Price check passed: ${avg_price:,.2f}")
                
                # Stop if we have enough
                if len(asset_examples) >= target_per_asset:
                    print(f"    ✓ Reached target of {target_per_asset:,}")
                    break
                    
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue
    
    # If we need more data, go back further
    if len(asset_examples) < target_per_asset:
        print(f"\n  Need more data (have {len(asset_examples):,}, need {target_per_asset:,})")
        additional_ranges = [
            ("20250720", "20250731"),  # Late July
            ("20250701", "20250719"),  # Early July
        ]
        
        for start_date, end_date in additional_ranges:
            if len(asset_examples) >= target_per_asset:
                break
                
            print(f"  📅 Adding {asset}: {start_date} to {end_date}")
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
                    print(f"    Added {min(needed, len(more_examples)):,} examples")
            except Exception as e:
                print(f"    Error: {e}")
    
    # Trim to exactly 100K
    if len(asset_examples) > target_per_asset:
        print(f"\n  Sampling exactly {target_per_asset:,} from {len(asset_examples):,}")
        asset_examples = random.sample(asset_examples, target_per_asset)
    
    # Calculate statistics
    for ex in asset_examples:
        if ex['metadata']['action_side'] == 'buy':
            asset_stats['buy'] += 1
        else:
            asset_stats['sell'] += 1
        
        obj = ex['metadata']['user_objective']
        if obj == 'aggressive':
            asset_stats['aggressive'] += 1
        elif obj == 'patient':
            asset_stats['patient'] += 1
        else:
            asset_stats['risk_averse'] += 1
    
    print(f"\n  📊 {asset} Dataset Statistics:")
    print(f"    Total: {len(asset_examples):,}")
    print(f"    Buy: {asset_stats['buy']:,} ({asset_stats['buy']/len(asset_examples)*100:.1f}%)")
    print(f"    Sell: {asset_stats['sell']:,} ({asset_stats['sell']/len(asset_examples)*100:.1f}%)")
    print(f"    Aggressive: {asset_stats['aggressive']:,} ({asset_stats['aggressive']/len(asset_examples)*100:.1f}%)")
    print(f"    Patient: {asset_stats['patient']:,} ({asset_stats['patient']/len(asset_examples)*100:.1f}%)")
    print(f"    Risk-averse: {asset_stats['risk_averse']:,} ({asset_stats['risk_averse']/len(asset_examples)*100:.1f}%)")
    
    # Shuffle and split
    print(f"\n  Creating train/val/test splits for {asset}...")
    random.shuffle(asset_examples)
    
    n = len(asset_examples)
    train_size = int(n * 0.8)  # 80K training
    val_size = int(n * 0.1)     # 10K validation
    
    splits = {
        'train': asset_examples[:train_size],
        'val': asset_examples[train_size:train_size + val_size],
        'test': asset_examples[train_size + val_size:]
    }
    
    # Save splits for this asset
    print(f"\n  💾 Saving {asset} dataset...")
    for split_name, split_data in splits.items():
        # Save as JSON for inspection
        json_file = f"data/training/{asset}_100K_{split_name}.json"
        with open(json_file, 'w') as f:
            json.dump(split_data, f)
        print(f"    {split_name}: {len(split_data):,} examples → {json_file}")
        
        # Save as JSONL for training
        jsonl_file = f"data/training/{asset}_100K_{split_name}.jsonl"
        with open(jsonl_file, 'w') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        print(f"    JSONL: → {jsonl_file}")
    
    # Quality check - show 3 random examples
    print(f"\n  🔍 Quality Check - 3 Random {asset} Examples:")
    for i in range(3):
        ex = random.choice(asset_examples)
        print(f"\n    {i+1}. {ex['metadata']['action_side'].upper()} - {ex['metadata']['user_objective']}")
        print(f"       Price: ${ex['metadata']['mid_price']:,.2f}")
        print(f"       Chosen: {ex['chosen'][:60]}...")
        print(f"       Reward diff: {ex['chosen_reward'] - ex['rejected_reward']:.3f}")
    
    print(f"\n  ✅ {asset} dataset complete!")

# Summary
print(f"\n{'='*80}")
print("✅ ALL ASSET DATASETS COMPLETE!")
print('='*80)
print("\n📊 Summary:")
for asset in assets:
    print(f"  {asset}: 100K examples (80K train, 10K val, 10K test)")

print("\n📁 Files created:")
print("  Per asset: [ASSET]_100K_train.jsonl, [ASSET]_100K_val.jsonl, [ASSET]_100K_test.json")

print("\n🚀 Training approach:")
print("  1. Train 4 separate specialized models")
print("  2. Each model learns asset-specific patterns")
print("  3. ~3-4 hours training per model")
print("  4. Can run in parallel on multiple GPUs")

print("\n💡 Advantages of specialized models:")
print("  - Better price prediction (each asset has different ranges)")
print("  - Asset-specific volatility patterns")
print("  - Tailored to each market's microstructure")
print("  - Higher accuracy than general model")

print(f"\nEnd time: {datetime.now()}")
print('='*80)