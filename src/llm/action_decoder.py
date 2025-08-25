"""
Action decoder for parsing LLM outputs into executable trading actions
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    CANCEL = "cancel"
    HOLD = "hold"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


@dataclass
class TradingAction:
    """Represents a parsed trading action"""
    side: OrderSide
    order_type: OrderType
    price: Optional[float] = None
    size: Optional[float] = None
    price_offset_bps: Optional[float] = None  # Offset from mid price in basis points
    confidence: float = 1.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'side': self.side.value,
            'order_type': self.order_type.value,
            'price': self.price,
            'size': self.size,
            'price_offset_bps': self.price_offset_bps,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }


class ActionDecoder:
    """Decodes LLM outputs into structured trading actions"""
    
    def __init__(self, default_size: float = 0.01):
        self.default_size = default_size
        
        # Regex patterns for parsing
        self.action_patterns = {
            'json': re.compile(r'\{[^}]+\}'),
            'buy_limit': re.compile(r'(?:place|submit)?\s*buy\s*limit\s*(?:order\s*)?(?:at\s*)?[\$]?([\d,]+\.?\d*)', re.IGNORECASE),
            'sell_limit': re.compile(r'(?:place|submit)?\s*sell\s*limit\s*(?:order\s*)?(?:at\s*)?[\$]?([\d,]+\.?\d*)', re.IGNORECASE),
            'price_offset': re.compile(r'([\d\.]+)\s*(?:bps|basis\s*points?)\s*(?:below|above)\s*(?:mid|market)', re.IGNORECASE),
            'size': re.compile(r'(?:size|amount|quantity)[:=\s]*([\d\.]+)', re.IGNORECASE),
            'hold': re.compile(r'(?:hold|wait|no\s*action|stay\s*flat)', re.IGNORECASE),
            'cancel': re.compile(r'cancel\s*(?:all\s*)?(?:orders?)?', re.IGNORECASE)
        }
    
    def decode(self, llm_output: str, current_mid_price: float = None) -> TradingAction:
        """
        Parse LLM output into a trading action
        
        Args:
            llm_output: Raw text output from LLM
            current_mid_price: Current mid price for relative orders
            
        Returns:
            Parsed TradingAction
        """
        # Try structured JSON parsing first
        action = self._try_json_parse(llm_output)
        if action:
            return action
        
        # Fall back to pattern matching
        action = self._parse_natural_language(llm_output, current_mid_price)
        
        # Extract reasoning if present
        action.reasoning = self._extract_reasoning(llm_output)
        
        return action
    
    def _try_json_parse(self, output: str) -> Optional[TradingAction]:
        """Try to parse JSON-formatted action"""
        json_match = self.action_patterns['json'].search(output)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return TradingAction(
                    side=OrderSide(data.get('side', 'hold')),
                    order_type=OrderType(data.get('type', 'limit')),
                    price=data.get('price'),
                    size=data.get('size', self.default_size),
                    price_offset_bps=data.get('offset_bps'),
                    confidence=data.get('confidence', 1.0)
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        return None
    
    def _parse_natural_language(self, output: str, mid_price: float = None) -> TradingAction:
        """Parse natural language trading instructions"""
        output_lower = output.lower()
        
        # Check for hold/wait
        if self.action_patterns['hold'].search(output):
            return TradingAction(
                side=OrderSide.HOLD,
                order_type=OrderType.LIMIT,
                confidence=0.9
            )
        
        # Check for cancel
        if self.action_patterns['cancel'].search(output):
            return TradingAction(
                side=OrderSide.CANCEL,
                order_type=OrderType.CANCEL,
                confidence=0.95
            )
        
        # Parse buy limit orders
        buy_match = self.action_patterns['buy_limit'].search(output)
        if buy_match:
            price = self._parse_price(buy_match.group(1))
            size = self._extract_size(output)
            offset = self._extract_price_offset(output, 'buy')
            
            return TradingAction(
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=price,
                size=size,
                price_offset_bps=offset,
                confidence=0.85
            )
        
        # Parse sell limit orders
        sell_match = self.action_patterns['sell_limit'].search(output)
        if sell_match:
            price = self._parse_price(sell_match.group(1))
            size = self._extract_size(output)
            offset = self._extract_price_offset(output, 'sell')
            
            return TradingAction(
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=price,
                size=size,
                price_offset_bps=offset,
                confidence=0.85
            )
        
        # Check for price offset instructions
        offset_match = self.action_patterns['price_offset'].search(output)
        if offset_match and mid_price:
            offset_bps = float(offset_match.group(1))
            is_below = 'below' in offset_match.group().lower()
            
            # Determine side based on offset direction
            if 'buy' in output_lower or (is_below and 'sell' not in output_lower):
                side = OrderSide.BUY
                price = mid_price * (1 - offset_bps / 10000) if is_below else mid_price * (1 + offset_bps / 10000)
            else:
                side = OrderSide.SELL
                price = mid_price * (1 + offset_bps / 10000) if not is_below else mid_price * (1 - offset_bps / 10000)
            
            return TradingAction(
                side=side,
                order_type=OrderType.LIMIT,
                price=price,
                size=self._extract_size(output),
                price_offset_bps=offset_bps if is_below else -offset_bps,
                confidence=0.8
            )
        
        # Default to hold if no clear action
        return TradingAction(
            side=OrderSide.HOLD,
            order_type=OrderType.LIMIT,
            confidence=0.5
        )
    
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float"""
        price_str = price_str.replace(',', '').replace('$', '')
        try:
            return float(price_str)
        except ValueError:
            return 0.0
    
    def _extract_size(self, output: str) -> float:
        """Extract order size from output"""
        size_match = self.action_patterns['size'].search(output)
        if size_match:
            try:
                return float(size_match.group(1))
            except ValueError:
                pass
        return self.default_size
    
    def _extract_price_offset(self, output: str, side: str) -> Optional[float]:
        """Extract price offset in basis points"""
        offset_match = self.action_patterns['price_offset'].search(output)
        if offset_match:
            offset = float(offset_match.group(1))
            is_below = 'below' in offset_match.group().lower()
            
            # Buy orders below mid are negative offset, above are positive
            # Sell orders above mid are positive offset, below are negative
            if side == 'buy':
                return -offset if is_below else offset
            else:
                return offset if not is_below else -offset
        return None
    
    def _extract_reasoning(self, output: str) -> str:
        """Extract reasoning from LLM output"""
        # Look for reasoning indicators
        reasoning_patterns = [
            r'because\s+(.+?)(?:\.|$)',
            r'reasoning:\s*(.+?)(?:\.|$)',
            r'rationale:\s*(.+?)(?:\.|$)',
            r'due to\s+(.+?)(?:\.|$)'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no explicit reasoning, use first sentence after action
        sentences = output.split('.')
        if len(sentences) > 1:
            return sentences[1].strip()
        
        return ""
    
    def validate_action(self, action: TradingAction, 
                       mid_price: float,
                       min_size: float = 0.001,
                       max_size: float = 1.0) -> Tuple[bool, str]:
        """
        Validate trading action for safety and sanity
        
        Args:
            action: Trading action to validate
            mid_price: Current mid price
            min_size: Minimum order size
            max_size: Maximum order size
            
        Returns:
            (is_valid, error_message)
        """
        # Skip validation for hold/cancel
        if action.side in [OrderSide.HOLD, OrderSide.CANCEL]:
            return True, ""
        
        # Validate price
        if action.price is None and action.price_offset_bps is None:
            return False, "No price or price offset specified"
        
        if action.price:
            # Check price is reasonable (within 10% of mid)
            price_deviation = abs(action.price - mid_price) / mid_price
            if price_deviation > 0.1:
                return False, f"Price {action.price} is more than 10% from mid price {mid_price}"
            
            # Check buy/sell price logic
            if action.side == OrderSide.BUY and action.price > mid_price * 1.01:
                return False, "Buy price is above mid price - would execute immediately"
            if action.side == OrderSide.SELL and action.price < mid_price * 0.99:
                return False, "Sell price is below mid price - would execute immediately"
        
        # Validate size
        if action.size:
            if action.size < min_size:
                return False, f"Size {action.size} is below minimum {min_size}"
            if action.size > max_size:
                return False, f"Size {action.size} is above maximum {max_size}"
        
        return True, ""


class ActionFormatter:
    """Formats actions for different output formats"""
    
    @staticmethod
    def to_exchange_order(action: TradingAction, symbol: str = "BTC-USD") -> Dict:
        """Convert action to exchange order format"""
        if action.side == OrderSide.HOLD:
            return None
        
        if action.side == OrderSide.CANCEL:
            return {
                'action': 'cancel_all',
                'symbol': symbol
            }
        
        return {
            'symbol': symbol,
            'side': action.side.value,
            'type': action.order_type.value,
            'price': action.price,
            'size': action.size,
            'time_in_force': 'GTC'
        }
    
    @staticmethod
    def to_human_readable(action: TradingAction) -> str:
        """Convert action to human-readable format"""
        if action.side == OrderSide.HOLD:
            return "Hold position - no action needed"
        
        if action.side == OrderSide.CANCEL:
            return "Cancel all open orders"
        
        desc = f"{action.side.value.capitalize()} {action.order_type.value} order"
        
        if action.price:
            desc += f" at ${action.price:,.2f}"
        elif action.price_offset_bps:
            direction = "below" if action.price_offset_bps < 0 else "above"
            desc += f" {abs(action.price_offset_bps):.1f} bps {direction} mid"
        
        if action.size:
            desc += f" for {action.size:.4f} units"
        
        if action.confidence < 1.0:
            desc += f" (confidence: {action.confidence:.0%})"
        
        if action.reasoning:
            desc += f". Reason: {action.reasoning}"
        
        return desc


if __name__ == "__main__":
    # Test the decoder
    decoder = ActionDecoder(default_size=0.01)
    formatter = ActionFormatter()
    
    # Test cases
    test_outputs = [
        "Place a buy limit order at $95,000 with size 0.05.",
        "Sell limit at 96000. Size: 0.02 BTC. The market is showing weakness.",
        '{"side": "buy", "type": "limit", "price": 94500, "size": 0.03, "confidence": 0.9}',
        "Place buy order 5 basis points below mid price due to strong support level.",
        "Hold position and wait for better opportunity.",
        "Cancel all orders - volatility too high.",
        "Submit a sell limit order 10 bps above market because of resistance."
    ]
    
    current_mid = 95000
    
    print("Testing Action Decoder:\n")
    for i, output in enumerate(test_outputs, 1):
        print(f"Test {i}: {output}")
        action = decoder.decode(output, current_mid)
        print(f"Parsed: {formatter.to_human_readable(action)}")
        
        is_valid, error = decoder.validate_action(action, current_mid)
        if not is_valid:
            print(f"Validation error: {error}")
        
        exchange_order = formatter.to_exchange_order(action)
        if exchange_order:
            print(f"Exchange format: {exchange_order}")
        print()