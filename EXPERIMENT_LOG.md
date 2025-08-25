# HyperDAPO Experiment Log
## Complete Record of All Training Runs and Results

**Project Timeline:** August 23-24, 2025

---

## Phase 1: Initial 100K Model Training (Completed)

### Training Runs
| Model | Start Time | End Time | Duration | GPU | Dataset Size | Final Loss | Accuracy |
|-------|------------|----------|----------|-----|--------------|------------|----------|
| BTC   | Aug 23, 2:30 PM | 5:00 PM | 2.5 hrs | 0 | 100K | 0.2418 | 90.0% |
| ETH   | Aug 23, 3:17 PM | 6:17 PM | 3.0 hrs | 1 | 100K | 0.2605 | 88.5% |
| SOL   | Aug 23, 8:28 PM | 11:00 PM | 2.5 hrs | 0 | 100K | ~0.25 | ~89% |
| HYPE  | Aug 23, 8:29 PM | 11:00 PM | 2.5 hrs | 1 | 100K | ~0.26 | ~88% |

### Backtesting Results (100K Models)
- **BTC Model:** 72.0% fill rate (+11.3% vs traditional)
- **ETH Model:** 72.1% fill rate (+11.2% vs traditional)
- **Average Slippage:** -1.00 bps (price improvement)
- **Statistical Significance:** p < 0.05

---

## Phase 2: 1M Model Training (In Progress)

### Current Training Runs
| Model | Start Time | Expected End | GPU | Dataset Size | Status |
|-------|------------|--------------|-----|--------------|--------|
| BTC   | Aug 23, 11:49 PM | Aug 24, ~11 PM | 0 | 272,826 | Running |
| ETH   | Aug 23, 11:49 PM | Aug 24, ~11 PM | 1 | 273,000 | Running |

---

## Commands for Resuming Work

### Check 1M Training Status
ssh -p 13263 root@212.13.234.23 "nvidia-smi"
ssh -p 13263 root@212.13.234.23 "cd HyperDAPO && tail -20 btc_1m_training.log"

### Transfer Completed 1M Models (when done)
scp -r -P 13263 root@212.13.234.23:~/HyperDAPO/models/btc_1m_model models/trained_models/
scp -r -P 13263 root@212.13.234.23:~/HyperDAPO/models/eth_1m_model models/trained_models/

---

*Last Updated: August 24, 2025, 12:50 AM*
*1M Models Training - Check in ~24 hours*
