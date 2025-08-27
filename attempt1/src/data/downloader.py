#!/usr/bin/env python3
"""
Tardis.dev data downloader for Hyperliquid (Solo subscription - CSV files)
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from dotenv import load_dotenv
import gzip
import time

load_dotenv()

class HyperliquidDataDownloader:
    """Download Hyperliquid historical data from Tardis.dev"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize downloader with API key"""
        self.api_key = api_key or os.getenv('TARDIS_API_KEY')
        if not self.api_key:
            raise ValueError("TARDIS_API_KEY not found in environment")
        
        self.base_url = "https://datasets.tardis.dev/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Create data directories
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_day(self, 
                     symbol: str, 
                     data_type: str, 
                     date: datetime) -> Optional[Path]:
        """Download data for a specific symbol and date"""
        
        # Format URL components
        year = date.year
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"
        
        # Construct URL
        url = f"{self.base_url}/hyperliquid/{data_type}/{year}/{month}/{day}/{symbol}.csv.gz"
        
        # Output filename
        filename = f"hyperliquid_{symbol}_{data_type}_{year}{month}{day}.csv.gz"
        output_path = self.raw_data_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"✓ Already exists: {filename}")
            return output_path
        
        print(f"⏬ Downloading: {filename}")
        
        try:
            response = requests.get(url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ Downloaded: {filename} ({file_size:.2f} MB)")
            return output_path
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"⚠️ No data available: {filename}")
            else:
                print(f"❌ Error downloading {filename}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def download_range(self, 
                      symbols: List[str], 
                      data_types: List[str],
                      start_date: datetime,
                      end_date: datetime) -> List[Path]:
        """Download data for multiple symbols over a date range"""
        
        downloaded_files = []
        
        print(f"\n📊 Downloading Hyperliquid data")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Data types: {', '.join(data_types)}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print("-" * 50)
        
        # Iterate through dates
        current_date = start_date
        while current_date <= end_date:
            for symbol in symbols:
                for data_type in data_types:
                    file_path = self.download_day(symbol, data_type, current_date)
                    if file_path:
                        downloaded_files.append(file_path)
                    
                    # Small delay to be respectful to the API
                    time.sleep(0.1)
            
            current_date += timedelta(days=1)
        
        print(f"\n✅ Downloaded {len(downloaded_files)} files")
        return downloaded_files
    
    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load a downloaded CSV file"""
        try:
            df = pd.read_csv(file_path, compression='gzip')
            
            # Convert timestamp from microseconds to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            if 'local_timestamp' in df.columns:
                df['local_timestamp'] = pd.to_datetime(df['local_timestamp'], unit='us')
            
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def get_data_info(self):
        """Get information about available data"""
        
        url = "https://api.tardis.dev/v1/api-key-info"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            info = response.json()
            print("\n📋 API Key Access Information:")
            for access in info:
                print(f"  Exchange: {access.get('exchange')}")
                print(f"  Access Type: {access.get('accessType')}")
                print(f"  From: {access.get('from')}")
                print(f"  To: {access.get('to')}")
                print(f"  Data Plan: {access.get('dataPlan')}")
                print()
            
            return info
            
        except Exception as e:
            print(f"Error getting API info: {e}")
            return None


def main():
    """Example usage"""
    
    # Initialize downloader
    downloader = HyperliquidDataDownloader()
    
    # Check API access
    downloader.get_data_info()
    
    # Download sample data
    symbols = ["BTC", "ETH"]
    data_types = ["trades"]
    
    # Download last 3 days of data
    end_date = datetime(2025, 8, 19)
    start_date = end_date - timedelta(days=2)
    
    files = downloader.download_range(
        symbols=symbols,
        data_types=data_types,
        start_date=start_date,
        end_date=end_date
    )
    
    # Load and preview first file
    if files:
        print("\n📈 Sample data from first file:")
        df = downloader.load_csv(files[0])
        if df is not None:
            print(df.head())
            print(f"\nShape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()