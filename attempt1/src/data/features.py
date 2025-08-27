#!/usr/bin/env python3
"""
Feature engineering pipeline for DAPO training
Combines trades and order book data to create rich features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """Extract features from trades and order book data"""
    
    def __init__(self, 
                 lookback_windows: List[int] = [10, 30, 60, 300],
                 orderbook_levels: int = 5):
        """
        Initialize feature engineering
        
        Args:
            lookback_windows: Time windows in seconds for rolling features
            orderbook_levels: Number of order book levels to use
        """
        self.lookback_windows = lookback_windows
        self.orderbook_levels = orderbook_levels
        
    def compute_orderbook_features(self, df_ob: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from order book snapshots
        
        Args:
            df_ob: Order book dataframe with bid/ask levels
            
        Returns:
            DataFrame with order book features
        """
        features = pd.DataFrame(index=df_ob.index)
        
        # Basic spread and mid price
        features['spread'] = df_ob['asks[0].price'] - df_ob['bids[0].price']
        features['spread_bps'] = (features['spread'] / df_ob['bids[0].price']) * 10000
        features['mid_price'] = (df_ob['asks[0].price'] + df_ob['bids[0].price']) / 2
        
        # Order book imbalance at different levels
        for level in [1, 3, 5]:
            bid_volume = 0
            ask_volume = 0
            
            for i in range(min(level, self.orderbook_levels)):
                bid_volume += df_ob[f'bids[{i}].amount']
                ask_volume += df_ob[f'asks[{i}].amount']
            
            features[f'imbalance_l{level}'] = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
            features[f'total_volume_l{level}'] = bid_volume + ask_volume
        
        # Weighted mid price (using volume)
        weighted_bid = 0
        weighted_ask = 0
        total_bid_vol = 0
        total_ask_vol = 0
        
        for i in range(self.orderbook_levels):
            weighted_bid += df_ob[f'bids[{i}].price'] * df_ob[f'bids[{i}].amount']
            weighted_ask += df_ob[f'asks[{i}].price'] * df_ob[f'asks[{i}].amount']
            total_bid_vol += df_ob[f'bids[{i}].amount']
            total_ask_vol += df_ob[f'asks[{i}].amount']
        
        features['weighted_mid_price'] = (
            (weighted_bid / (total_bid_vol + 1e-10) + 
             weighted_ask / (total_ask_vol + 1e-10)) / 2
        )
        
        # Book pressure (volume weighted by distance from mid)
        features['book_pressure'] = 0
        for i in range(self.orderbook_levels):
            weight = 1 / (i + 1)
            features['book_pressure'] += (
                df_ob[f'bids[{i}].amount'] * weight - 
                df_ob[f'asks[{i}].amount'] * weight
            )
        features['book_pressure'] = features['book_pressure'] / (features['book_pressure'].abs() + 1e-10)
        
        # Depth metrics
        features['bid_depth_weighted'] = sum(
            df_ob[f'bids[{i}].amount'] / (i + 1) 
            for i in range(self.orderbook_levels)
        )
        features['ask_depth_weighted'] = sum(
            df_ob[f'asks[{i}].amount'] / (i + 1) 
            for i in range(self.orderbook_levels)
        )
        
        # Price level distances
        features['bid_ask_spread_l5'] = (
            df_ob['asks[4].price'] - df_ob['bids[4].price']
        ) if self.orderbook_levels >= 5 else features['spread']
        
        return features
    
    def compute_trade_features(self, df_trades: pd.DataFrame, 
                              window_seconds: int = 60) -> pd.DataFrame:
        """
        Extract features from trades data
        
        Args:
            df_trades: Trades dataframe
            window_seconds: Rolling window in seconds
            
        Returns:
            DataFrame with trade features
        """
        # Make a copy to avoid modifying original
        df_trades = df_trades.copy()
        
        # Trade flow imbalance
        df_trades['signed_volume'] = df_trades['amount'] * df_trades['side'].map({'buy': 1, 'sell': -1})
        
        # Set timestamp as index for time-based operations
        df_trades_ts = df_trades.set_index('timestamp')
        
        # Initialize features dataframe
        features = pd.DataFrame(index=df_trades_ts.index)
        
        # Rolling features
        for window in self.lookback_windows:
            window_str = f'{window}s'
            rolling = df_trades_ts.rolling(window_str)
            
            # Volume metrics
            features[f'volume_{window}s'] = rolling['amount'].sum().values
            features[f'trade_count_{window}s'] = rolling['amount'].count().values
            features[f'avg_trade_size_{window}s'] = rolling['amount'].mean().values
            
            # Trade flow
            features[f'trade_flow_{window}s'] = rolling['signed_volume'].sum().values
            features[f'trade_flow_ratio_{window}s'] = (
                features[f'trade_flow_{window}s'] / (features[f'volume_{window}s'] + 1e-10)
            )
            
            # Price metrics
            features[f'price_std_{window}s'] = rolling['price'].std().values
            
            # VWAP (simplified)
            price_volume = df_trades_ts['price'] * df_trades_ts['amount']
            features[f'vwap_{window}s'] = (
                rolling.apply(lambda x: x.sum(), raw=False)[price_volume.columns[0]] /
                rolling['amount'].sum()
            ).values if len(price_volume.shape) > 1 else (
                price_volume.rolling(window_str).sum() / 
                rolling['amount'].sum()
            ).values
        
        # Microstructure features
        features['tick_direction'] = np.sign(df_trades_ts['price'].diff()).values
        features['tick_size'] = df_trades_ts['price'].diff().abs().values
        
        # Reset index to match original
        features = features.reset_index(drop=False)
        
        return features
    
    def compute_combined_features(self, 
                                 df_trades: pd.DataFrame,
                                 df_ob: pd.DataFrame) -> pd.DataFrame:
        """
        Combine trades and order book data to create full feature set
        
        Args:
            df_trades: Trades dataframe
            df_ob: Order book dataframe
            
        Returns:
            Combined feature dataframe
        """
        # Ensure timestamps are datetime
        df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
        df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp'])
        
        # Get order book features
        ob_features = self.compute_orderbook_features(df_ob)
        ob_features['timestamp'] = df_ob['timestamp']
        
        # Get trade features
        trade_features = self.compute_trade_features(df_trades)
        # Make sure timestamp is properly aligned
        if 'timestamp' in trade_features.columns:
            # timestamp already in features from compute_trade_features
            pass
        else:
            trade_features['timestamp'] = df_trades['timestamp'].values
        
        # Merge on nearest timestamp
        df_combined = pd.merge_asof(
            trade_features.sort_values('timestamp'),
            ob_features.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('1s')
        )
        
        # Add additional cross features
        df_combined['trade_price'] = df_trades['price']
        df_combined['trade_size'] = df_trades['amount']
        df_combined['trade_side'] = df_trades['side'].map({'buy': 1, 'sell': -1})
        
        # Distance from mid price
        df_combined['price_vs_mid'] = (
            (df_combined['trade_price'] - df_combined['mid_price']) / 
            df_combined['mid_price'] * 10000  # in basis points
        )
        
        # Trade through spread
        df_combined['aggressive_trade'] = (
            ((df_combined['trade_side'] == 1) & 
             (df_combined['trade_price'] >= df_combined['mid_price'])) |
            ((df_combined['trade_side'] == -1) & 
             (df_combined['trade_price'] <= df_combined['mid_price']))
        ).astype(int)
        
        # Normalize features
        df_combined = self.normalize_features(df_combined)
        
        return df_combined
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features for model training
        
        Args:
            df: Feature dataframe
            
        Returns:
            Normalized dataframe
        """
        # List of columns to normalize
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        volume_cols = [col for col in df.columns if 'volume' in col or 'amount' in col]
        
        # Z-score normalization for most features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['timestamp', 'trade_side', 'aggressive_trade']:
                # Use robust scaling (median and IQR)
                median = df[col].median()
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr = q75 - q25
                
                if iqr > 0:
                    df[col] = (df[col] - median) / iqr
                else:
                    df[col] = 0
        
        # Clip outliers
        df[numeric_cols] = df[numeric_cols].clip(-10, 10)
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def create_training_features(self, 
                                df: pd.DataFrame,
                                target_horizon: int = 60) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features and targets for DAPO training
        
        Args:
            df: Combined feature dataframe
            target_horizon: Seconds ahead for target prediction
            
        Returns:
            Features dataframe and target series
        """
        # Select feature columns
        feature_cols = [
            'spread_bps', 'imbalance_l1', 'imbalance_l3', 'imbalance_l5',
            'book_pressure', 'total_volume_l1', 'total_volume_l5',
            'volume_60s', 'trade_flow_60s', 'trade_flow_ratio_60s',
            'price_return_60s', 'vwap_60s', 'price_vs_mid',
            'aggressive_trade', 'bid_depth_weighted', 'ask_depth_weighted'
        ]
        
        # Filter available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        
        # Create target: future price movement
        # For limit order execution, we want to predict if our order will fill
        # and at what price improvement
        df['future_mid'] = df['mid_price'].shift(-target_horizon)
        df['future_return'] = (df['future_mid'] / df['mid_price'] - 1) * 10000  # bps
        
        # Target: price improvement opportunity
        # Positive = good for buying, Negative = good for selling
        y = df['future_return']
        
        # Remove rows with NaN targets
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y


def main():
    """Example usage"""
    print("🔧 Feature Engineering Pipeline")
    print("=" * 60)
    
    # Initialize feature engineering
    fe = FeatureEngineering()
    
    # Load sample data
    print("\n📊 Loading data...")
    df_trades = pd.read_csv('data/raw/hyperliquid_BTC_trades_20250819.csv.gz', nrows=10000)
    df_ob = pd.read_csv('data/raw/hyperliquid_BTC_book_snapshot_5_20250819.csv.gz', nrows=10000)
    
    # Convert timestamps
    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='us')
    df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp'], unit='us')
    
    print(f"Loaded {len(df_trades)} trades and {len(df_ob)} order book snapshots")
    
    # Compute features
    print("\n⚙️ Computing features...")
    df_features = fe.compute_combined_features(df_trades, df_ob)
    
    print(f"Created {len(df_features)} samples with {df_features.shape[1]} features")
    
    # Create training data
    X, y = fe.create_training_features(df_features)
    
    print(f"\n✅ Training data ready:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Feature columns: {list(X.columns)}")
    
    # Save processed data
    output_path = Path('data/processed')
    output_path.mkdir(exist_ok=True)
    
    X.to_csv(output_path / 'features_sample.csv', index=False)
    y.to_csv(output_path / 'targets_sample.csv', index=False)
    
    print(f"\n💾 Saved to data/processed/")


if __name__ == "__main__":
    main()