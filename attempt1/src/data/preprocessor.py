#!/usr/bin/env python3
"""
Data preprocessing pipeline for DAPO training
Handles data loading, cleaning, splitting, and batching
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from datetime import datetime, timedelta

class DataPreprocessor:
    """Preprocess market data for DAPO training"""
    
    def __init__(self, 
                 data_dir: str = "data/raw",
                 sequence_length: int = 60,
                 prediction_horizon: int = 30,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15):
        """
        Initialize preprocessor
        
        Args:
            data_dir: Directory containing raw data
            sequence_length: Number of time steps for input sequences
            prediction_horizon: Time steps ahead to predict
            train_ratio: Portion of data for training
            val_ratio: Portion of data for validation
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        
        self.scaler = RobustScaler()
        self.feature_columns = None
        
    def load_data(self, 
                  symbol: str = "BTC",
                  start_date: str = "2025-08-17",
                  end_date: str = "2025-08-19") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load trades and order book data
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Trades and order book dataframes
        """
        print(f"📊 Loading {symbol} data from {start_date} to {end_date}")
        
        # Convert dates to datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        trades_list = []
        ob_list = []
        
        # Load data for each day
        current = start
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            
            # Load trades
            trades_file = self.data_dir / f"hyperliquid_{symbol}_trades_{date_str}.csv.gz"
            if trades_file.exists():
                df_trades = pd.read_csv(trades_file)
                df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='us')
                trades_list.append(df_trades)
                print(f"  ✓ Loaded {len(df_trades)} trades for {current.date()}")
            
            # Load order book
            ob_file = self.data_dir / f"hyperliquid_{symbol}_book_snapshot_5_{date_str}.csv.gz"
            if ob_file.exists():
                df_ob = pd.read_csv(ob_file)
                df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp'], unit='us')
                ob_list.append(df_ob)
                print(f"  ✓ Loaded {len(df_ob)} order book snapshots for {current.date()}")
            
            current += timedelta(days=1)
        
        # Combine all data
        df_trades = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
        df_ob = pd.concat(ob_list, ignore_index=True) if ob_list else pd.DataFrame()
        
        print(f"\n📈 Total loaded: {len(df_trades)} trades, {len(df_ob)} order book snapshots")
        
        return df_trades, df_ob
    
    def create_sequences(self, 
                        features: np.ndarray,
                        targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            features: Feature array
            targets: Target array
            
        Returns:
            Sequences and corresponding targets
        """
        n_samples = len(features) - self.sequence_length - self.prediction_horizon + 1
        
        if n_samples <= 0:
            raise ValueError("Not enough data for the specified sequence length and prediction horizon")
        
        X = np.zeros((n_samples, self.sequence_length, features.shape[1]))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            X[i] = features[i:i + self.sequence_length]
            y[i] = targets[i + self.sequence_length + self.prediction_horizon - 1]
        
        return X, y
    
    def prepare_training_data(self, 
                            df_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df_features: Feature dataframe
            
        Returns:
            Dictionary with train, val, test splits
        """
        # Remove NaN values
        df_features = df_features.dropna()
        
        # Sort by timestamp if available
        if 'timestamp' in df_features.columns:
            df_features = df_features.sort_values('timestamp')
            df_features = df_features.drop('timestamp', axis=1)
        
        # Separate features and target
        target_col = 'future_return' if 'future_return' in df_features.columns else None
        
        if target_col:
            X = df_features.drop(target_col, axis=1)
            y = df_features[target_col].values
        else:
            # If no target, create a dummy one (for unsupervised learning)
            X = df_features
            y = np.zeros(len(X))
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        # Split data (time-based split for time series)
        n_samples = len(X_seq)
        train_end = int(n_samples * self.train_ratio)
        val_end = train_end + int(n_samples * self.val_ratio)
        
        data_splits = {
            'X_train': X_seq[:train_end],
            'y_train': y_seq[:train_end],
            'X_val': X_seq[train_end:val_end],
            'y_val': y_seq[train_end:val_end],
            'X_test': X_seq[val_end:],
            'y_test': y_seq[val_end:]
        }
        
        print(f"\n📊 Data splits:")
        print(f"  Training:   {len(data_splits['X_train'])} sequences")
        print(f"  Validation: {len(data_splits['X_val'])} sequences")
        print(f"  Testing:    {len(data_splits['X_test'])} sequences")
        print(f"  Sequence shape: {data_splits['X_train'].shape}")
        
        return data_splits
    
    def create_limit_order_labels(self, 
                                 df_trades: pd.DataFrame,
                                 df_ob: pd.DataFrame,
                                 limit_offset_bps: List[float] = [0, 5, 10, 20]) -> pd.DataFrame:
        """
        Create labels for limit order placement
        
        Args:
            df_trades: Trades dataframe
            df_ob: Order book dataframe
            limit_offset_bps: Offset from mid price in basis points
            
        Returns:
            DataFrame with limit order labels
        """
        # Merge trades with order book on nearest timestamp
        df_trades = df_trades.sort_values('timestamp')
        df_ob = df_ob.sort_values('timestamp')
        
        df_merged = pd.merge_asof(
            df_trades,
            df_ob[['timestamp', 'bids[0].price', 'asks[0].price']],
            on='timestamp',
            direction='nearest'
        )
        
        # Calculate mid price
        df_merged['mid_price'] = (df_merged['bids[0].price'] + df_merged['asks[0].price']) / 2
        
        # Create labels for different limit order strategies
        labels = pd.DataFrame(index=df_merged.index)
        
        for offset in limit_offset_bps:
            # Buy limit orders
            buy_limit_price = df_merged['mid_price'] * (1 - offset / 10000)
            labels[f'buy_limit_{offset}bps_filled'] = (
                df_merged['price'] <= buy_limit_price
            ).astype(int)
            
            # Sell limit orders
            sell_limit_price = df_merged['mid_price'] * (1 + offset / 10000)
            labels[f'sell_limit_{offset}bps_filled'] = (
                df_merged['price'] >= sell_limit_price
            ).astype(int)
        
        # Add execution quality metrics
        labels['spread_captured'] = (
            (df_merged['asks[0].price'] - df_merged['bids[0].price']) / 
            df_merged['mid_price'] * 10000
        )
        
        return labels
    
    def save_preprocessed_data(self, 
                              data_splits: Dict[str, np.ndarray],
                              output_dir: str = "data/processed"):
        """
        Save preprocessed data to disk
        
        Args:
            data_splits: Dictionary with train/val/test splits
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save numpy arrays
        for key, value in data_splits.items():
            np.save(output_path / f"{key}.npy", value)
        
        # Save scaler
        joblib.dump(self.scaler, output_path / "scaler.pkl")
        
        # Save feature columns
        with open(output_path / "feature_columns.txt", 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        print(f"\n💾 Saved preprocessed data to {output_path}")


def main():
    """Example usage"""
    print("🔧 Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df_trades, df_ob = preprocessor.load_data(
        symbol="BTC",
        start_date="2025-08-17",
        end_date="2025-08-19"
    )
    
    # Create limit order labels
    print("\n🎯 Creating limit order labels...")
    labels = preprocessor.create_limit_order_labels(df_trades[:10000], df_ob[:10000])
    
    print("Label statistics:")
    for col in labels.columns:
        if 'filled' in col:
            fill_rate = labels[col].mean() * 100
            print(f"  {col}: {fill_rate:.2f}% fill rate")
    
    # Load pre-computed features (from features.py)
    features_file = Path("data/processed/features_sample.csv")
    if features_file.exists():
        print("\n📊 Loading pre-computed features...")
        df_features = pd.read_csv(features_file)
        
        # Prepare training data
        data_splits = preprocessor.prepare_training_data(df_features)
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(data_splits)
        
        print("\n✅ Preprocessing complete!")
    else:
        print("\n⚠️ Run features.py first to generate features")


if __name__ == "__main__":
    main()