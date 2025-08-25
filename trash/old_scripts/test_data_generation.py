#!/usr/bin/env python3
"""
Test training data generation with a small sample
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from llm.data_generator import TrainingDataGenerator
import json

def main():
    print("Testing data generation with small sample...")
    
    generator = TrainingDataGenerator()
    
    # Generate just 3 days of data for BTC as a test
    examples = generator.generate_training_examples(
        symbol="BTC",
        start_date="20250818",  # Just last 3 days
        end_date="20250820",
        lookforward_minutes=5
    )
    
    print(f"\nGenerated {len(examples)} training examples")
    
    if examples:
        # Show first example
        print("\n" + "="*60)
        print("SAMPLE TRAINING EXAMPLE")
        print("="*60)
        
        sample = examples[0]
        
        print("\nPROMPT:")
        print("-"*40)
        print(sample['prompt'][:500] + "..." if len(sample['prompt']) > 500 else sample['prompt'])
        
        print("\nCHOSEN ACTION:")
        print("-"*40)
        print(sample['chosen'])
        
        print("\nREJECTED ACTION:")
        print("-"*40)
        print(sample['rejected'])
        
        print("\nREWARDS:")
        print("-"*40)
        print(f"Chosen reward: {sample['chosen_reward']:.3f}")
        print(f"Rejected reward: {sample['rejected_reward']:.3f}")
        print(f"Reward difference: {sample['chosen_reward'] - sample['rejected_reward']:.3f}")
        
        # Save sample to file
        output_dir = Path("data/training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "sample_training_data.json", "w") as f:
            json.dump(examples[:10], f, indent=2, default=str)
        
        print(f"\nSaved first 10 examples to data/training/sample_training_data.json")
        
        # Statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        
        import numpy as np
        chosen_rewards = [e['chosen_reward'] for e in examples]
        rejected_rewards = [e['rejected_reward'] for e in examples]
        
        print(f"Average chosen reward: {np.mean(chosen_rewards):.3f}")
        print(f"Average rejected reward: {np.mean(rejected_rewards):.3f}")
        print(f"Chosen win rate: {sum(1 for c, r in zip(chosen_rewards, rejected_rewards) if c > r) / len(examples):.1%}")
        
        # Action distribution
        chosen_actions = [e['chosen'].split('.')[0] for e in examples]
        hold_count = sum(1 for a in chosen_actions if 'Hold' in a)
        buy_count = sum(1 for a in chosen_actions if 'buy' in a.lower())
        sell_count = sum(1 for a in chosen_actions if 'sell' in a.lower())
        
        print(f"\nAction distribution:")
        print(f"  Buy orders: {buy_count} ({buy_count/len(examples):.1%})")
        print(f"  Sell orders: {sell_count} ({sell_count/len(examples):.1%})")
        print(f"  Hold: {hold_count} ({hold_count/len(examples):.1%})")
    
    else:
        print("No examples generated - check data files")

if __name__ == "__main__":
    main()