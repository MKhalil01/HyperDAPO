#!/usr/bin/env python3
"""
Download complete 4-month dataset for all assets
Designed to run overnight
"""

from downloader import HyperliquidDataDownloader
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download_full.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def download_full_dataset():
    """Download complete dataset for thesis"""
    
    logger.info("="*60)
    logger.info("🌙 OVERNIGHT DOWNLOAD STARTING")
    logger.info("="*60)
    
    downloader = HyperliquidDataDownloader()
    
    # Configuration
    assets = ["BTC", "ETH", "SOL", "HYPE"]
    data_types = ["trades", "book_snapshot_5"]
    
    # Full date range (4 months)
    start_date = datetime(2025, 4, 20)
    end_date = datetime(2025, 8, 20)
    
    total_days = (end_date - start_date).days + 1
    
    logger.info(f"📅 Date Range: {start_date.date()} to {end_date.date()} ({total_days} days)")
    logger.info(f"🪙 Assets: {', '.join(assets)}")
    logger.info(f"📁 Data Types: {', '.join(data_types)}")
    
    # Statistics
    stats = {
        'downloaded': 0,
        'skipped': 0,
        'failed': 0,
        'total_size_mb': 0,
        'start_time': time.time()
    }
    
    # Download each asset
    for asset in assets:
        logger.info(f"\n{'='*60}")
        logger.info(f"📥 Downloading {asset}")
        logger.info(f"{'='*60}")
        
        asset_start = time.time()
        asset_files = 0
        
        # Download in weekly batches
        current = start_date
        week_num = 1
        
        while current <= end_date:
            # Define week end
            week_end = min(current + timedelta(days=6), end_date)
            week_days = (week_end - current).days + 1
            
            logger.info(f"\n📅 Week {week_num}: {current.date()} to {week_end.date()} ({week_days} days)")
            
            try:
                # Download the week
                files = downloader.download_range(
                    symbols=[asset],
                    data_types=data_types,
                    start_date=current,
                    end_date=week_end
                )
                
                # Update statistics
                week_size = 0
                for f in files:
                    if f and f.exists():
                        size_mb = f.stat().st_size / (1024 * 1024)
                        week_size += size_mb
                        stats['total_size_mb'] += size_mb
                        stats['downloaded'] += 1
                        asset_files += 1
                
                logger.info(f"  ✅ Week {week_num}: {len(files)} files, {week_size:.1f} MB")
                
            except Exception as e:
                logger.error(f"  ❌ Week {week_num} failed: {e}")
                stats['failed'] += 1
            
            # Move to next week
            current = week_end + timedelta(days=1)
            week_num += 1
            
            # Brief pause between weeks
            time.sleep(1)
        
        # Asset summary
        asset_time = time.time() - asset_start
        logger.info(f"\n✅ {asset} Complete:")
        logger.info(f"  Files: {asset_files}")
        logger.info(f"  Time: {asset_time/60:.1f} minutes")
        logger.info(f"  Speed: {asset_files/(asset_time/60):.1f} files/minute")
    
    # Final summary
    total_time = time.time() - stats['start_time']
    
    logger.info("\n" + "="*60)
    logger.info("🎉 DOWNLOAD COMPLETE!")
    logger.info("="*60)
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"  Total files: {stats['downloaded']}")
    logger.info(f"  Total size: {stats['total_size_mb']/1000:.2f} GB")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Time taken: {total_time/3600:.1f} hours")
    logger.info(f"  Average speed: {stats['downloaded']/(total_time/60):.1f} files/minute")
    
    # Check what we have
    data_dir = Path("data/raw")
    for asset in assets:
        asset_files = list(data_dir.glob(f"hyperliquid_{asset}_*.csv.gz"))
        logger.info(f"\n{asset}: {len(asset_files)} files")
        
        if asset_files:
            # Get date range
            dates = []
            for f in asset_files:
                parts = f.stem.split('_')
                if len(parts) >= 4 and len(parts[-1]) == 8:
                    dates.append(parts[-1])
            
            if dates:
                dates.sort()
                logger.info(f"  Date range: {dates[0]} to {dates[-1]}")
    
    # Save completion marker
    with open("data/raw/download_complete.txt", "w") as f:
        f.write(f"Download completed at {datetime.now()}\n")
        f.write(f"Total files: {stats['downloaded']}\n")
        f.write(f"Total size: {stats['total_size_mb']/1000:.2f} GB\n")
        f.write(f"Time taken: {total_time/3600:.1f} hours\n")
    
    logger.info("\n✅ All data ready for Day 3 work!")
    logger.info("\nNext steps:")
    logger.info("1. Build order book simulator")
    logger.info("2. Create fill probability model")
    logger.info("3. Implement Gym trading environment")
    logger.info("4. Start DAPO model development")

if __name__ == "__main__":
    download_full_dataset()