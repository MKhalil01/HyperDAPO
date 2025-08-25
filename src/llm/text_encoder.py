"""
Text encoder for converting numerical market features to natural language
for LLM-based trading decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class MarketStateEncoder:
    """Converts numerical market features to natural language descriptions"""
    
    def __init__(self):
        self.feature_templates = {
            'spread': self._encode_spread,
            'imbalance': self._encode_imbalance,
            'volume': self._encode_volume,
            'volatility': self._encode_volatility,
            'momentum': self._encode_momentum,
            'liquidity': self._encode_liquidity,
            'trade_flow': self._encode_trade_flow,
            'price_level': self._encode_price_level
        }
    
    def encode_market_state(self, features: Dict[str, float], 
                           symbol: str = "BTC",
                           timestamp: Optional[datetime] = None) -> str:
        """
        Convert numerical features to natural language market description
        
        Args:
            features: Dictionary of feature names to values
            symbol: Trading symbol
            timestamp: Optional timestamp for context
            
        Returns:
            Natural language description of market state
        """
        descriptions = []
        
        # Add header with symbol and time if available
        if timestamp:
            descriptions.append(f"Market state for {symbol} at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}:")
        else:
            descriptions.append(f"Current market state for {symbol}:")
        
        # Core market microstructure
        if 'spread_bps' in features:
            descriptions.append(self._encode_spread(features['spread_bps']))
        
        if 'book_imbalance' in features:
            descriptions.append(self._encode_imbalance(features['book_imbalance']))
        
        if 'bid_depth_1pct' in features and 'ask_depth_1pct' in features:
            descriptions.append(self._encode_liquidity(
                features['bid_depth_1pct'], 
                features['ask_depth_1pct']
            ))
        
        # Volume and activity
        if 'volume_1m' in features:
            descriptions.append(self._encode_volume(features['volume_1m']))
        
        if 'trade_count_1m' in features:
            descriptions.append(f"Recent activity shows {int(features['trade_count_1m'])} trades in the last minute.")
        
        # Price dynamics
        if 'volatility_1h' in features:
            descriptions.append(self._encode_volatility(features['volatility_1h']))
        
        if 'price_momentum_5m' in features:
            descriptions.append(self._encode_momentum(features['price_momentum_5m']))
        
        if 'vwap_deviation' in features:
            descriptions.append(self._encode_price_level(features['vwap_deviation']))
        
        # Trade flow
        if 'trade_flow_imbalance' in features:
            descriptions.append(self._encode_trade_flow(features['trade_flow_imbalance']))
        
        # Market regime
        regime = self._classify_market_regime(features)
        descriptions.append(f"Market regime: {regime}.")
        
        return " ".join(descriptions)
    
    def _encode_spread(self, spread_bps: float) -> str:
        """Encode bid-ask spread in natural language"""
        if spread_bps < 1:
            return f"The spread is very tight at {spread_bps:.2f} basis points."
        elif spread_bps < 5:
            return f"The spread is narrow at {spread_bps:.2f} basis points."
        elif spread_bps < 10:
            return f"The spread is moderate at {spread_bps:.2f} basis points."
        else:
            return f"The spread is wide at {spread_bps:.2f} basis points, indicating lower liquidity."
    
    def _encode_imbalance(self, imbalance: float) -> str:
        """Encode order book imbalance"""
        abs_imb = abs(imbalance)
        if abs_imb < 0.1:
            return "The order book is balanced."
        elif imbalance > 0.3:
            return f"Strong buying pressure with {imbalance:.1%} bid-side imbalance."
        elif imbalance > 0.1:
            return f"Moderate buying pressure with {imbalance:.1%} bid-side imbalance."
        elif imbalance < -0.3:
            return f"Strong selling pressure with {abs(imbalance):.1%} ask-side imbalance."
        else:
            return f"Moderate selling pressure with {abs(imbalance):.1%} ask-side imbalance."
    
    def _encode_volume(self, volume: float) -> str:
        """Encode trading volume"""
        if volume < 100_000:
            return f"Low trading volume of ${volume:,.0f} in the last minute."
        elif volume < 1_000_000:
            return f"Moderate trading volume of ${volume:,.0f} in the last minute."
        elif volume < 10_000_000:
            return f"High trading volume of ${volume/1e6:.1f}M in the last minute."
        else:
            return f"Very high trading volume of ${volume/1e6:.1f}M in the last minute."
    
    def _encode_volatility(self, volatility: float) -> str:
        """Encode volatility level"""
        if volatility < 0.5:
            return f"Low volatility at {volatility:.1%} per hour."
        elif volatility < 1.0:
            return f"Moderate volatility at {volatility:.1%} per hour."
        elif volatility < 2.0:
            return f"High volatility at {volatility:.1%} per hour."
        else:
            return f"Very high volatility at {volatility:.1%} per hour, indicating significant price swings."
    
    def _encode_momentum(self, momentum: float) -> str:
        """Encode price momentum"""
        if abs(momentum) < 0.1:
            return "Price is moving sideways with no clear momentum."
        elif momentum > 0.5:
            return f"Strong upward momentum of {momentum:.1%} over 5 minutes."
        elif momentum > 0.1:
            return f"Moderate upward momentum of {momentum:.1%} over 5 minutes."
        elif momentum < -0.5:
            return f"Strong downward momentum of {momentum:.1%} over 5 minutes."
        else:
            return f"Moderate downward momentum of {momentum:.1%} over 5 minutes."
    
    def _encode_liquidity(self, bid_depth: float, ask_depth: float) -> str:
        """Encode market depth/liquidity"""
        total_depth = bid_depth + ask_depth
        if total_depth < 100_000:
            return f"Thin liquidity with ${total_depth:,.0f} within 1% of mid price."
        elif total_depth < 1_000_000:
            return f"Moderate liquidity with ${total_depth:,.0f} within 1% of mid price."
        else:
            return f"Deep liquidity with ${total_depth/1e6:.1f}M within 1% of mid price."
    
    def _encode_trade_flow(self, flow_imbalance: float) -> str:
        """Encode trade flow imbalance"""
        if abs(flow_imbalance) < 0.1:
            return "Balanced trade flow between buyers and sellers."
        elif flow_imbalance > 0.3:
            return f"Heavy buying with {flow_imbalance:.1%} net buy flow."
        elif flow_imbalance > 0:
            return f"Net buying with {flow_imbalance:.1%} buy flow imbalance."
        elif flow_imbalance < -0.3:
            return f"Heavy selling with {abs(flow_imbalance):.1%} net sell flow."
        else:
            return f"Net selling with {abs(flow_imbalance):.1%} sell flow imbalance."
    
    def _encode_price_level(self, vwap_dev: float) -> str:
        """Encode price relative to VWAP"""
        if abs(vwap_dev) < 0.1:
            return "Price is near VWAP, indicating fair value."
        elif vwap_dev > 0.5:
            return f"Price is {vwap_dev:.1%} above VWAP, potentially overextended."
        elif vwap_dev > 0:
            return f"Price is {vwap_dev:.1%} above VWAP."
        elif vwap_dev < -0.5:
            return f"Price is {abs(vwap_dev):.1%} below VWAP, potentially oversold."
        else:
            return f"Price is {abs(vwap_dev):.1%} below VWAP."
    
    def _classify_market_regime(self, features: Dict[str, float]) -> str:
        """Classify overall market regime based on features"""
        volatility = features.get('volatility_1h', 1.0)
        volume = features.get('volume_1m', 500_000)
        spread = features.get('spread_bps', 5.0)
        
        if volatility > 2.0 and volume > 5_000_000:
            return "High volatility with heavy trading"
        elif volatility < 0.5 and spread < 2:
            return "Calm with tight spreads"
        elif volume < 100_000:
            return "Low activity"
        elif spread > 10:
            return "Wide spreads indicating uncertainty"
        else:
            return "Normal trading conditions"
    
    def encode_batch(self, df: pd.DataFrame, symbol: str = "BTC") -> List[str]:
        """
        Encode a batch of market states from DataFrame
        
        Args:
            df: DataFrame with feature columns
            symbol: Trading symbol
            
        Returns:
            List of encoded market descriptions
        """
        descriptions = []
        
        for idx, row in df.iterrows():
            features = row.to_dict()
            
            # Get timestamp if available
            timestamp = None
            if isinstance(idx, pd.Timestamp):
                timestamp = idx.to_pydatetime()
            elif 'timestamp' in features:
                timestamp = pd.to_datetime(features['timestamp'])
            
            description = self.encode_market_state(features, symbol, timestamp)
            descriptions.append(description)
        
        return descriptions


class TradingContextEncoder:
    """Adds trading context and strategy information to market descriptions"""
    
    def __init__(self):
        self.position_states = {
            'flat': "No position",
            'long': "Long position",
            'short': "Short position"
        }
    
    def add_trading_context(self, 
                           market_description: str,
                           position: str = 'flat',
                           pnl: float = 0,
                           recent_fills: List[Dict] = None) -> str:
        """
        Add trading context to market description
        
        Args:
            market_description: Base market state description
            position: Current position ('flat', 'long', 'short')
            pnl: Current PnL
            recent_fills: List of recent order fills
            
        Returns:
            Enhanced description with trading context
        """
        context_parts = [market_description]
        
        # Add position information
        context_parts.append(f"Current position: {self.position_states.get(position, position)}.")
        
        # Add PnL
        if pnl != 0:
            pnl_str = f"profit of ${pnl:,.2f}" if pnl > 0 else f"loss of ${abs(pnl):,.2f}"
            context_parts.append(f"Current {pnl_str}.")
        
        # Add recent fills
        if recent_fills:
            fill_summary = self._summarize_fills(recent_fills)
            context_parts.append(fill_summary)
        
        # Add trading objective
        context_parts.append("Objective: Place optimal limit orders to maximize execution quality and minimize market impact.")
        
        return " ".join(context_parts)
    
    def _summarize_fills(self, fills: List[Dict]) -> str:
        """Summarize recent order fills"""
        if not fills:
            return "No recent fills."
        
        total_volume = sum(f.get('size', 0) for f in fills)
        avg_price = sum(f.get('price', 0) * f.get('size', 0) for f in fills) / total_volume if total_volume > 0 else 0
        
        return f"Recent execution: {len(fills)} fills totaling {total_volume:.4f} units at average price ${avg_price:,.2f}."


if __name__ == "__main__":
    # Test the encoder
    encoder = MarketStateEncoder()
    context_encoder = TradingContextEncoder()
    
    # Sample features
    sample_features = {
        'spread_bps': 3.5,
        'book_imbalance': 0.15,
        'bid_depth_1pct': 500_000,
        'ask_depth_1pct': 450_000,
        'volume_1m': 2_500_000,
        'trade_count_1m': 156,
        'volatility_1h': 1.2,
        'price_momentum_5m': 0.25,
        'vwap_deviation': -0.15,
        'trade_flow_imbalance': 0.08
    }
    
    # Encode market state
    market_desc = encoder.encode_market_state(sample_features, "BTC", datetime.now())
    print("Market Description:")
    print(market_desc)
    print()
    
    # Add trading context
    full_context = context_encoder.add_trading_context(
        market_desc,
        position='long',
        pnl=1250.50,
        recent_fills=[
            {'price': 95000, 'size': 0.1},
            {'price': 95050, 'size': 0.05}
        ]
    )
    print("\nWith Trading Context:")
    print(full_context)