"""
Generate training data for DAPO-based LLM trading model
"""

import pandas as pd
import numpy as np
import json
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm.text_encoder import MarketStateEncoder, TradingContextEncoder
from llm.prompts import TradingPrompts
from data.features import FeatureEngineering


class RewardCalculator:
    """Calculate rewards for trading actions based on outcomes"""
    
    def __init__(self, 
                 spread_cost_weight: float = 0.3,
                 price_improvement_weight: float = 0.4,
                 fill_probability_weight: float = 0.3):
        """
        Initialize reward calculator
        
        Args:
            spread_cost_weight: Weight for spread cost component
            price_improvement_weight: Weight for price improvement
            fill_probability_weight: Weight for fill probability
        """
        self.spread_cost_weight = spread_cost_weight
        self.price_improvement_weight = price_improvement_weight
        self.fill_probability_weight = fill_probability_weight
    
    def calculate_limit_order_reward(self,
                                    order_price: float,
                                    mid_price: float,
                                    future_prices: np.ndarray,
                                    spread: float,
                                    side: str = 'buy') -> Tuple[float, Dict]:
        """
        Calculate reward for a limit order based on future price movement
        
        Args:
            order_price: Limit order price
            mid_price: Current mid price
            future_prices: Array of future mid prices (e.g., next 5 minutes)
            spread: Current bid-ask spread
            side: 'buy' or 'sell'
            
        Returns:
            (reward, metrics_dict)
        """
        metrics = {}
        
        # Calculate if order would have filled
        if side == 'buy':
            filled = np.any(future_prices <= order_price)
            fill_price = future_prices[future_prices <= order_price][0] if filled else None
        else:
            filled = np.any(future_prices >= order_price)
            fill_price = future_prices[future_prices >= order_price][0] if filled else None
        
        metrics['filled'] = filled
        
        if not filled:
            # Penalty for not filling
            metrics['fill_probability'] = 0.0
            metrics['price_improvement'] = 0.0
            metrics['spread_cost'] = 0.0
            return -0.1, metrics  # Small penalty for no fill
        
        # Calculate fill probability based on how quickly it filled
        if filled:
            fill_index = np.where(future_prices <= order_price if side == 'buy' else future_prices >= order_price)[0][0]
            fill_probability = 1.0 - (fill_index / len(future_prices))
            metrics['fill_probability'] = fill_probability
        else:
            fill_probability = 0.0
            metrics['fill_probability'] = 0.0
        
        # Calculate price improvement vs market order
        if mid_price > 0:
            if side == 'buy':
                market_price = mid_price + spread/2
                price_improvement = (market_price - order_price) / mid_price
            else:
                market_price = mid_price - spread/2
                price_improvement = (order_price - market_price) / mid_price
        else:
            price_improvement = 0.0
        
        metrics['price_improvement'] = price_improvement
        
        # Calculate spread cost saved
        spread_cost_saved = spread / (2 * mid_price) if (filled and mid_price > 0) else 0
        metrics['spread_cost'] = spread_cost_saved
        
        # Calculate final PnL if we held position
        final_price = future_prices[-1] if len(future_prices) > 0 else mid_price
        if order_price > 0:
            if side == 'buy':
                pnl = (final_price - order_price) / order_price
            else:
                pnl = (order_price - final_price) / order_price
        else:
            pnl = 0.0
        
        metrics['pnl'] = pnl
        
        # Combine into final reward
        reward = (
            self.fill_probability_weight * fill_probability +
            self.price_improvement_weight * max(0, price_improvement * 100) +  # Scale up
            self.spread_cost_weight * spread_cost_saved * 100 +  # Scale up
            0.2 * max(0, pnl * 100)  # Add PnL component
        )
        
        # Normalize reward to [-1, 1] range
        reward = np.clip(reward, -1, 1)
        
        return reward, metrics
    
    def calculate_hold_reward(self, 
                            mid_price: float,
                            future_prices: np.ndarray,
                            volatility: float) -> Tuple[float, Dict]:
        """
        Calculate reward for holding (not trading)
        
        Args:
            mid_price: Current mid price
            future_prices: Future price movements
            volatility: Current volatility
            
        Returns:
            (reward, metrics)
        """
        metrics = {}
        
        # Calculate opportunity cost
        if mid_price > 0 and len(future_prices) > 0:
            price_change = (future_prices[-1] - mid_price) / mid_price
        else:
            price_change = 0.0
        metrics['price_change'] = price_change
        
        # Reward for holding during high volatility
        if volatility > 2.0:  # High volatility threshold
            volatility_bonus = 0.2
        else:
            volatility_bonus = 0
        
        metrics['volatility'] = volatility
        
        # Small positive reward for avoiding bad trades
        if abs(price_change) < 0.001:  # Price didn't move much
            reward = 0.1 + volatility_bonus
        else:
            # Penalty for missing opportunity
            reward = -abs(price_change) * 10
        
        return np.clip(reward, -1, 1), metrics


class TrainingDataGenerator:
    """Generate training data for DAPO model"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.encoder = MarketStateEncoder()
        self.context_encoder = TradingContextEncoder()
        self.prompts = TradingPrompts()
        self.reward_calc = RewardCalculator()
        self.feature_eng = FeatureEngineering()
    
    def generate_training_examples(self,
                                  symbol: str = "BTC",
                                  start_date: str = "20250420",
                                  end_date: str = "20250520",
                                  lookforward_minutes: int = 5) -> List[Dict]:
        """
        Generate training examples from historical data
        
        Args:
            symbol: Asset symbol
            start_date: Start date YYYYMMDD
            end_date: End date YYYYMMDD
            lookforward_minutes: Minutes to look forward for rewards
            
        Returns:
            List of training examples
        """
        print(f"Generating training data for {symbol} from {start_date} to {end_date}")
        
        # Load and process data for date range
        all_examples = []
        
        # Get list of dates
        dates = pd.date_range(
            start=pd.to_datetime(start_date, format='%Y%m%d'),
            end=pd.to_datetime(end_date, format='%Y%m%d'),
            freq='D'
        )
        
        for date in dates:
            date_str = date.strftime('%Y%m%d')
            
            # Load data for this date
            trades_file = self.data_dir / f"hyperliquid_{symbol}_trades_{date_str}.csv.gz"
            ob_file = self.data_dir / f"hyperliquid_{symbol}_book_snapshot_5_{date_str}.csv.gz"
            
            if not trades_file.exists() or not ob_file.exists():
                continue
            
            print(f"Processing {date_str}...")
            
            # Load data
            df_trades = pd.read_csv(trades_file, compression='gzip')
            df_ob = pd.read_csv(ob_file, compression='gzip')
            
            # Compute features
            df_features = self.feature_eng.compute_combined_features(df_trades, df_ob)
            
            if df_features.empty:
                continue
            
            # Generate examples from this day
            examples = self._generate_examples_from_features(
                df_features, 
                symbol,
                lookforward_minutes
            )
            
            all_examples.extend(examples)
            print(f"  Generated {len(examples)} examples")
        
        print(f"\nTotal examples generated: {len(all_examples)}")
        return all_examples
    
    def _generate_examples_from_features(self,
                                        df: pd.DataFrame,
                                        symbol: str,
                                        lookforward_minutes: int) -> List[Dict]:
        """Generate examples from feature DataFrame"""
        examples = []
        
        # Sample every 30 seconds to avoid too much correlation
        sample_indices = range(0, len(df) - lookforward_minutes * 2, 2)
        
        for i in sample_indices:
            if i + lookforward_minutes * 2 >= len(df):
                break
            
            # Current state
            current_features = df.iloc[i].to_dict()
            
            # Future prices for reward calculation
            future_prices = df.iloc[i+1:i+lookforward_minutes*2]['mid_price'].values
            
            if len(future_prices) < lookforward_minutes:
                continue
            
            # Generate market description
            market_state = self.encoder.encode_market_state(
                current_features, 
                symbol
            )
            
            # Generate different actions and their rewards
            mid_price = current_features.get('mid_price', 0)
            if mid_price <= 0:
                # Skip if no valid mid price
                continue
            spread = current_features.get('spread', 0.0001) * mid_price
            
            # Action 1: Aggressive buy (at mid - small offset)
            aggressive_buy_price = mid_price - spread * 0.1
            aggressive_reward, aggressive_metrics = self.reward_calc.calculate_limit_order_reward(
                aggressive_buy_price, mid_price, future_prices, spread, 'buy'
            )
            
            # Action 2: Passive buy (at mid - larger offset)
            passive_buy_price = mid_price - spread * 0.5
            passive_reward, passive_metrics = self.reward_calc.calculate_limit_order_reward(
                passive_buy_price, mid_price, future_prices, spread, 'buy'
            )
            
            # Action 3: Hold
            hold_reward, hold_metrics = self.reward_calc.calculate_hold_reward(
                mid_price, future_prices, current_features.get('volatility_1h', 1.0)
            )
            
            # Create preference pairs based on rewards
            if aggressive_reward > passive_reward and aggressive_reward > hold_reward:
                chosen_action = f"Place buy limit order at ${aggressive_buy_price:,.2f}"
                chosen_reason = "Aggressive entry near mid price for quick fill"
                rejected_action = f"Place buy limit order at ${passive_buy_price:,.2f}"
                rejected_reason = "Too passive, might miss the move"
            elif passive_reward > hold_reward:
                chosen_action = f"Place buy limit order at ${passive_buy_price:,.2f}"
                chosen_reason = "Patient entry for better price"
                rejected_action = "Hold position"
                rejected_reason = "Missing opportunity in trending market"
            else:
                chosen_action = "Hold position"
                chosen_reason = "Market conditions unfavorable for entry"
                rejected_action = f"Place buy limit order at ${aggressive_buy_price:,.2f}"
                rejected_reason = "Too risky in current conditions"
            
            # Create training example
            example = {
                'prompt': self.prompts.create_trading_prompt(market_state),
                'chosen': f"{chosen_action}. {chosen_reason}",
                'rejected': f"{rejected_action}. {rejected_reason}",
                'chosen_reward': max(aggressive_reward, passive_reward, hold_reward),
                'rejected_reward': min(aggressive_reward, passive_reward, hold_reward),
                'metadata': {
                    'timestamp': df.index[i] if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    'symbol': symbol,
                    'mid_price': mid_price,
                    'spread': spread,
                    'features': current_features
                }
            }
            
            examples.append(example)
        
        return examples
    
    def save_training_data(self, examples: List[Dict], output_file: str):
        """Save training examples to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2, default=str)
        
        print(f"Saved {len(examples)} examples to {output_file}")
        
        # Also save in JSONL format for easier streaming
        jsonl_file = output_path.with_suffix('.jsonl')
        with open(jsonl_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example, default=str) + '\n')
        
        print(f"Also saved in JSONL format: {jsonl_file}")
    
    def create_train_val_test_split(self, 
                                   examples: List[Dict],
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        """Split examples into train/validation/test sets"""
        n = len(examples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Shuffle examples
        import random
        random.seed(42)
        random.shuffle(examples)
        
        splits = {
            'train': examples[:n_train],
            'validation': examples[n_train:n_train + n_val],
            'test': examples[n_train + n_val:]
        }
        
        print(f"Split sizes - Train: {len(splits['train'])}, "
              f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        
        return splits


def main():
    """Generate training data for all assets"""
    generator = TrainingDataGenerator()
    
    # Generate examples for each asset
    assets = ["BTC", "ETH", "SOL", "HYPE"]
    all_examples = []
    
    for asset in assets:
        print(f"\n{'='*60}")
        print(f"Processing {asset}")
        print('='*60)
        
        # Generate one month of data per asset (to keep manageable)
        examples = generator.generate_training_examples(
            symbol=asset,
            start_date="20250720",  # Last month of data
            end_date="20250820",
            lookforward_minutes=5
        )
        
        all_examples.extend(examples)
        
        # Save asset-specific data
        generator.save_training_data(
            examples,
            f"data/training/{asset}_training_data.json"
        )
    
    # Create combined dataset
    print(f"\n{'='*60}")
    print(f"Creating combined dataset")
    print('='*60)
    
    # Split into train/val/test
    splits = generator.create_train_val_test_split(all_examples)
    
    # Save splits
    for split_name, split_data in splits.items():
        generator.save_training_data(
            split_data,
            f"data/training/combined_{split_name}.json"
        )
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Training Data Statistics")
    print('='*60)
    print(f"Total examples: {len(all_examples)}")
    print(f"Average chosen reward: {np.mean([e['chosen_reward'] for e in all_examples]):.3f}")
    print(f"Average rejected reward: {np.mean([e['rejected_reward'] for e in all_examples]):.3f}")
    
    # Sample example
    if all_examples:
        print("\nSample training example:")
        print("-" * 40)
        sample = all_examples[0]
        print(f"Prompt: {sample['prompt'][:200]}...")
        print(f"Chosen: {sample['chosen']}")
        print(f"Rejected: {sample['rejected']}")
        print(f"Reward diff: {sample['chosen_reward'] - sample['rejected_reward']:.3f}")


if __name__ == "__main__":
    main()