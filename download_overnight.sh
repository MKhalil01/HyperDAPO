#!/bin/bash
# Overnight download script for full 4-month dataset

echo "🌙 Starting overnight download - $(date)"
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

# Create log directory
mkdir -p logs

# Download data for each asset with logging
echo "📊 Downloading full dataset for all assets..."
echo "This will take several hours. You can close this terminal and check logs/download.log tomorrow."

# Run the Python download script with full output logged
python3 << 'EOF' 2>&1 | tee logs/download_$(date +%Y%m%d_%H%M%S).log
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from src.data.downloader import HyperliquidDataDownloader
from datetime import datetime, timedelta
import time

print("🚀 Full Dataset Download Starting")
print("=" * 60)

downloader = HyperliquidDataDownloader()

# Configuration
assets = ["BTC", "ETH", "SOL", "HYPE"]
data_types = ["trades", "book_snapshot_5"]

# Full date range
start_date = datetime(2025, 4, 20)
end_date = datetime(2025, 8, 20)

print(f"📅 Date Range: {start_date.date()} to {end_date.date()}")
print(f"🪙 Assets: {', '.join(assets)}")
print(f"📁 Data Types: {', '.join(data_types)}")
print("-" * 60)

# Track overall progress
total_days = (end_date - start_date).days + 1
total_expected = total_days * len(assets) * len(data_types) * 2  # rough estimate

downloaded_count = 0
failed_count = 0
start_time = time.time()

# Download each asset
for asset in assets:
    print(f"\n{'='*60}")
    print(f"📥 Starting {asset} download at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    asset_start = time.time()
    
    # Download in weekly chunks to avoid timeouts
    current = start_date
    while current <= end_date:
        # Define chunk end (1 week or remaining days)
        chunk_end = min(current + timedelta(days=6), end_date)
        
        print(f"\n  📅 Downloading {current.date()} to {chunk_end.date()}...")
        
        try:
            files = downloader.download_range(
                symbols=[asset],
                data_types=data_types,
                start_date=current,
                end_date=chunk_end
            )
            downloaded_count += len(files)
            
            # Progress update
            elapsed = time.time() - start_time
            rate = downloaded_count / (elapsed / 60) if elapsed > 0 else 0
            print(f"    ✓ Downloaded {len(files)} files ({rate:.1f} files/min)")
            
        except Exception as e:
            print(f"    ❌ Error in chunk: {e}")
            failed_count += 1
        
        current = chunk_end + timedelta(days=1)
        
        # Small delay between chunks
        time.sleep(2)
    
    asset_elapsed = time.time() - asset_start
    print(f"\n✅ {asset} complete in {asset_elapsed/60:.1f} minutes")

# Final summary
total_elapsed = time.time() - start_time

print("\n" + "="*60)
print("🎉 DOWNLOAD COMPLETE!")
print("="*60)
print(f"\nTotal files downloaded: {downloaded_count}")
print(f"Failed chunks: {failed_count}")
print(f"Time taken: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
print(f"Average speed: {downloaded_count/(total_elapsed/60):.1f} files/minute")

# Check total size
import os
from pathlib import Path

data_dir = Path("data/raw")
total_size = sum(f.stat().st_size for f in data_dir.glob("*.csv.gz")) / (1024**3)
print(f"\n💾 Total data size: {total_size:.2f} GB")

print("\n✅ All data ready for Day 3!")
print("\nNext steps for tomorrow:")
print("1. Build order book simulator")
print("2. Create fill probability model")
print("3. Implement Gym trading environment")
print("4. Start DAPO model development")

# Save completion flag
with open("data/raw/download_complete.txt", "w") as f:
    f.write(f"Download completed at {datetime.now()}\n")
    f.write(f"Files: {downloaded_count}\n")
    f.write(f"Total size: {total_size:.2f} GB\n")
EOF

echo ""
echo "=================================================="
echo "✅ Download process finished - $(date)"
echo "Check logs/download_*.log for details"
echo "Data ready in data/raw/"