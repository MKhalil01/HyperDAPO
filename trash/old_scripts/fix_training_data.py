#!/usr/bin/env python3
"""
Fix training data format for DAPO training
"""

import json
import sys

def convert_json_to_jsonl(input_file, output_file):
    """Convert JSON array to JSONL format"""
    print(f"Converting {input_file} to JSONL format...")
    
    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Write as JSONL
    with open(output_file, 'w') as f:
        for item in data:
            # Ensure each item has required fields
            processed_item = {
                'prompt': item.get('prompt', ''),
                'chosen': item.get('chosen', ''),
                'rejected': item.get('rejected', ''),
                'chosen_reward': item.get('chosen_reward', 0),
                'rejected_reward': item.get('rejected_reward', 0)
            }
            f.write(json.dumps(processed_item) + '\n')
    
    print(f"Converted {len(data)} examples to JSONL")
    print(f"Output saved to: {output_file}")
    
    # Show sample
    if data:
        print("\nSample entry:")
        print(f"Prompt: {data[0]['prompt'][:100]}...")
        print(f"Chosen: {data[0]['chosen'][:100]}")

if __name__ == "__main__":
    # Convert the test training data
    convert_json_to_jsonl(
        'data/training/BTC_test_training_data.json',
        'data/training/BTC_training_fixed.jsonl'
    )