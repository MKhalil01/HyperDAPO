# Tardis.dev Data Analysis for HyperDAPO Project - Research Results

## Research Conducted
- Investigated Tardis.dev's exchange coverage and data offerings
- Researched DAPO methodology and requirements
- Analyzed limit order execution model data needs
- Evaluated data volume requirements for RL trading models
- Assessed market microstructure features importance

## Key Findings

### Tardis.dev Hyperliquid Support
**Status**: Hyperliquid is NOT currently supported by Tardis.dev
- Tardis.dev supports 20+ major exchanges but Hyperliquid is not listed
- Alternative: Hyperliquid provides native historical data access

### DAPO Methodology Research
**Status**: DAPO is primarily designed for LLM training, not financial trading
- DAPO = Decoupled Clip and Dynamic sAmpling Policy Optimization
- Developed by ByteDance Seed and Tsinghua AIR for language models
- No direct trading applications found in literature

### Data Requirements Analysis
**Status**: Comprehensive requirements identified for limit order execution
- Minimum training period: 6 months to 2 years of historical data
- Data frequency: High-frequency tick-level data preferred
- Key features: Order book L2 data, trade data, market microstructure variables

### Timeline Assessment
**Status**: Date reference needs clarification (2025 vs 2024)
- Current date context suggests 2024 data would be more appropriate
- If referring to April 2024 onwards: ~8 months of data available
- This timeframe is at the lower end but potentially sufficient

## Recommendations
1. Clarify intended project timeline and methodology
2. Consider Hyperliquid's native data API instead of Tardis.dev
3. Evaluate alternative RL approaches more suited to trading
4. Plan for minimum 1 year of historical data collection