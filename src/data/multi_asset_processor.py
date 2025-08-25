#!/usr/bin/env python3
"""
Process multiple assets through feature engineering pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from features import FeatureEngineering
from preprocessor import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class MultiAssetProcessor:
    """Process multiple cryptocurrency assets for DAPO training"""
    
    def __init__(self):
        self.fe = FeatureEngineering()
        self.preprocessor = DataPreprocessor()
        self.assets = ["BTC", "ETH", "SOL", "HYPE"]
        self.processed_data = {}
        
    def process_asset(self, symbol: str, verbose: bool = True):
        """Process a single asset through the pipeline"""
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {symbol}")
            print('-'*60)
        
        # Load data
        df_trades, df_ob = self.preprocessor.load_data(
            symbol=symbol,
            start_date="2025-08-17",
            end_date="2025-08-19"
        )
        
        if len(df_trades) == 0 or len(df_ob) == 0:
            print(f"⚠️ No data found for {symbol}")
            return None
        
        # Sample data for processing (to keep memory manageable)
        sample_size = min(50000, len(df_trades))
        df_trades_sample = df_trades.sample(n=sample_size).sort_values('timestamp')
        
        # Get corresponding order book data
        ob_sample_size = min(50000, len(df_ob))
        df_ob_sample = df_ob.sample(n=ob_sample_size).sort_values('timestamp')
        
        # Remove any rows with null timestamps
        df_trades_sample = df_trades_sample.dropna(subset=['timestamp'])
        df_ob_sample = df_ob_sample.dropna(subset=['timestamp'])
        
        # Compute features
        if verbose:
            print(f"⚙️ Computing features...")
        
        try:
            df_features = self.fe.compute_combined_features(
                df_trades_sample,
                df_ob_sample
            )
            
            # Add asset identifier
            df_features['asset'] = symbol
            
            # Create training data
            X, y = self.fe.create_training_features(df_features)
            
            if verbose:
                print(f"✅ Created {len(X)} feature vectors")
                print(f"   Feature shape: {X.shape}")
                print(f"   Target mean: {y.mean():.4f}")
                print(f"   Target std: {y.std():.4f}")
            
            # Store results
            self.processed_data[symbol] = {
                'features': X,
                'targets': y,
                'raw_features': df_features,
                'stats': {
                    'num_samples': len(X),
                    'feature_dim': X.shape[1],
                    'target_mean': y.mean(),
                    'target_std': y.std(),
                    'num_trades': len(df_trades),
                    'num_orderbook': len(df_ob)
                }
            }
            
            return self.processed_data[symbol]
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            return None
    
    def process_all_assets(self):
        """Process all assets"""
        
        print("🔧 Multi-Asset Feature Engineering")
        print("="*60)
        
        for symbol in self.assets:
            self.process_asset(symbol)
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 PROCESSING SUMMARY")
        print('-'*60)
        
        for symbol, data in self.processed_data.items():
            if data:
                stats = data['stats']
                print(f"\n{symbol}:")
                print(f"  Samples: {stats['num_samples']}")
                print(f"  Features: {stats['feature_dim']}")
                print(f"  Target μ: {stats['target_mean']:.4f}")
                print(f"  Target σ: {stats['target_std']:.4f}")
    
    def compare_feature_distributions(self):
        """Compare feature distributions across assets"""
        
        print(f"\n{'='*60}")
        print("📈 FEATURE COMPARISON ACROSS ASSETS")
        print('-'*60)
        
        # Key features to compare
        features_to_compare = [
            'spread_bps', 'imbalance_l1', 'book_pressure',
            'volume_60s', 'trade_flow_ratio_60s'
        ]
        
        comparison = {}
        
        for symbol, data in self.processed_data.items():
            if data and 'raw_features' in data:
                df = data['raw_features']
                comparison[symbol] = {}
                
                for feat in features_to_compare:
                    if feat in df.columns:
                        comparison[symbol][f"{feat}_mean"] = df[feat].mean()
                        comparison[symbol][f"{feat}_std"] = df[feat].std()
        
        # Create comparison dataframe
        df_comp = pd.DataFrame(comparison).T
        
        if not df_comp.empty:
            print("\nMean values by asset:")
            print(df_comp[[col for col in df_comp.columns if '_mean' in col]].round(4))
            
            print("\nStandard deviations by asset:")
            print(df_comp[[col for col in df_comp.columns if '_std' in col]].round(4))
        
        return df_comp
    
    def save_processed_data(self, output_dir: str = "data/processed/multi_asset"):
        """Save processed data for each asset"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving processed data to {output_path}")
        
        for symbol, data in self.processed_data.items():
            if data:
                # Save features and targets
                np.save(output_path / f"{symbol}_features.npy", data['features'])
                np.save(output_path / f"{symbol}_targets.npy", data['targets'])
                
                # Save statistics
                pd.Series(data['stats']).to_csv(
                    output_path / f"{symbol}_stats.csv"
                )
                
                print(f"  ✓ Saved {symbol} data")
        
        print("✅ All data saved!")
    
    def generate_thesis_insights(self):
        """Generate insights for thesis"""
        
        print(f"\n{'='*60}")
        print("💡 THESIS INSIGHTS")
        print('-'*60)
        
        if not self.processed_data:
            print("No data processed yet")
            return
        
        print("\n1. DATA CHARACTERISTICS:")
        print("  Asset | Samples | Target Mean | Target Std")
        print("  ------|---------|-------------|------------")
        for symbol, data in self.processed_data.items():
            if data:
                stats = data['stats']
                print(f"  {symbol:5} | {stats['num_samples']:7} | {stats['target_mean']:11.4f} | {stats['target_std']:10.4f}")
        
        print("\n2. KEY FINDINGS:")
        
        # Find asset with highest volatility in targets
        max_vol_asset = max(
            self.processed_data.items(),
            key=lambda x: x[1]['stats']['target_std'] if x[1] else 0
        )[0]
        
        min_vol_asset = min(
            self.processed_data.items(),
            key=lambda x: x[1]['stats']['target_std'] if x[1] else float('inf')
        )[0]
        
        print(f"  • Highest target volatility: {max_vol_asset}")
        print(f"  • Lowest target volatility: {min_vol_asset}")
        
        print("\n3. IMPLICATIONS FOR DAPO:")
        print("  • Different assets require different reward scaling")
        print("  • Order book dynamics vary significantly")
        print("  • Model should adapt to asset-specific characteristics")
        
        print("\n4. SUGGESTED EXPERIMENTS:")
        print("  • Train separate models per asset")
        print("  • Train unified model with asset embedding")
        print("  • Transfer learning from BTC to other assets")
        print("  • Compare performance metrics across assets")


def main():
    """Process all assets"""
    
    processor = MultiAssetProcessor()
    
    # Process all assets
    processor.process_all_assets()
    
    # Compare features
    processor.compare_feature_distributions()
    
    # Save processed data
    processor.save_processed_data()
    
    # Generate insights
    processor.generate_thesis_insights()
    
    print("\n✅ Multi-asset processing complete!")


if __name__ == "__main__":
    main()