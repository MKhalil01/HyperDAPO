#!/usr/bin/env python3
"""
Verify quality of generated DAPO training data
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd
from collections import Counter

def analyze_training_data(file_path):
    """Analyze training data quality and statistics"""
    
    print(f"Analyzing: {file_path}")
    print("="*60)
    
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    
    # Analyze rewards
    chosen_rewards = [e['chosen_reward'] for e in data]
    rejected_rewards = [e['rejected_reward'] for e in data]
    reward_diffs = [c-r for c,r in zip(chosen_rewards, rejected_rewards)]
    
    print("\nReward Statistics:")
    print(f"  Chosen rewards   - Mean: {np.mean(chosen_rewards):.3f}, Std: {np.std(chosen_rewards):.3f}")
    print(f"  Rejected rewards - Mean: {np.mean(rejected_rewards):.3f}, Std: {np.std(rejected_rewards):.3f}")
    print(f"  Reward difference - Mean: {np.mean(reward_diffs):.3f}, Std: {np.std(reward_diffs):.3f}")
    print(f"  Positive diffs: {sum(1 for d in reward_diffs if d > 0)}/{len(reward_diffs)} ({100*sum(1 for d in reward_diffs if d > 0)/len(reward_diffs):.1f}%)")
    
    # Analyze actions
    chosen_actions = []
    rejected_actions = []
    
    for e in data[:1000]:  # Sample first 1000 for action analysis
        chosen = e['chosen'].lower()
        rejected = e['rejected'].lower()
        
        if 'hold' in chosen:
            chosen_actions.append('hold')
        elif 'buy' in chosen:
            chosen_actions.append('buy')
        elif 'sell' in chosen:
            chosen_actions.append('sell')
        else:
            chosen_actions.append('other')
            
        if 'hold' in rejected:
            rejected_actions.append('hold')
        elif 'buy' in rejected:
            rejected_actions.append('buy')
        elif 'sell' in rejected:
            rejected_actions.append('sell')
        else:
            rejected_actions.append('other')
    
    print("\nAction Distribution (first 1000 examples):")
    print("  Chosen actions:", dict(Counter(chosen_actions)))
    print("  Rejected actions:", dict(Counter(rejected_actions)))
    
    # Check prompt lengths
    prompt_lengths = [len(e['prompt']) for e in data[:1000]]
    print(f"\nPrompt lengths - Mean: {np.mean(prompt_lengths):.0f}, Min: {min(prompt_lengths)}, Max: {max(prompt_lengths)}")
    
    # Check for data quality issues
    print("\nData Quality Checks:")
    
    # Check for identical chosen/rejected
    identical = sum(1 for e in data if e['chosen'] == e['rejected'])
    print(f"  Identical chosen/rejected: {identical}")
    
    # Check for missing rewards
    missing_rewards = sum(1 for e in data if e.get('chosen_reward') is None or e.get('rejected_reward') is None)
    print(f"  Missing rewards: {missing_rewards}")
    
    # Check reward range
    out_of_range = sum(1 for e in data if abs(e['chosen_reward']) > 1 or abs(e['rejected_reward']) > 1)
    print(f"  Rewards out of [-1, 1] range: {out_of_range}")
    
    # Sample examples
    print("\n" + "="*60)
    print("Sample Examples:")
    print("="*60)
    
    for i in range(min(3, len(data))):
        e = data[i]
        print(f"\nExample {i+1}:")
        print(f"  Prompt: {e['prompt'][:100]}...")
        print(f"  Chosen: {e['chosen'][:100]}")
        print(f"  Rejected: {e['rejected'][:100]}")
        print(f"  Rewards: Chosen={e['chosen_reward']:.3f}, Rejected={e['rejected_reward']:.3f}, Diff={e['chosen_reward']-e['rejected_reward']:.3f}")
    
    return {
        'total_examples': len(data),
        'mean_reward_diff': np.mean(reward_diffs),
        'positive_diffs_pct': 100*sum(1 for d in reward_diffs if d > 0)/len(reward_diffs)
    }

def main():
    """Analyze all training data files"""
    
    print("="*60)
    print("DAPO Training Data Quality Verification")
    print("="*60)
    
    training_dir = Path("data/training")
    
    # Find all JSON training files
    json_files = list(training_dir.glob("*.json"))
    
    if not json_files:
        print("No training data files found!")
        return
    
    results = {}
    for file_path in json_files:
        if 'jsonl' not in str(file_path):  # Skip JSONL files
            print(f"\n{'='*60}")
            results[file_path.stem] = analyze_training_data(file_path)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  Examples: {stats['total_examples']}")
        print(f"  Mean reward diff: {stats['mean_reward_diff']:.3f}")
        print(f"  Positive diffs: {stats['positive_diffs_pct']:.1f}%")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    all_good = True
    
    for name, stats in results.items():
        if stats['mean_reward_diff'] < 0.1:
            print(f"⚠️ {name}: Low reward difference, may need to adjust reward calculation")
            all_good = False
        if stats['positive_diffs_pct'] < 95:
            print(f"⚠️ {name}: Some negative reward differences, check preference pair generation")
            all_good = False
        if stats['total_examples'] < 1000:
            print(f"⚠️ {name}: Low number of examples, consider generating more data")
            all_good = False
    
    if all_good:
        print("✅ All training data looks good for DAPO training!")
        print("Ready to proceed with GPU setup and model training")
    
    print("\nNext steps:")
    print("1. Generate full dataset for all assets (if not done)")
    print("2. Set up GPU environment on Vast.ai or RunPod")
    print("3. Upload training data and start DAPO training")

if __name__ == "__main__":
    main()