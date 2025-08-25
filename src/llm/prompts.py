"""
Prompt templates for LLM-based cryptocurrency trading
"""

from typing import Dict, List, Optional
from datetime import datetime


class TradingPrompts:
    """Collection of prompt templates for trading decisions"""
    
    # System prompt defining the trading agent's role and capabilities
    SYSTEM_PROMPT = """You are an expert cryptocurrency trading AI specializing in optimal limit order placement. Your role is to analyze market microstructure and place limit orders that maximize execution quality while minimizing market impact.

Key principles:
1. Place limit orders at strategic price levels based on order book dynamics
2. Consider spread, liquidity, momentum, and volatility in your decisions
3. Avoid chasing the market - be patient for good entry points
4. Manage risk by using appropriate position sizing
5. Provide clear reasoning for each trading decision

Output format:
- State your trading decision clearly (buy/sell limit at price, or hold)
- Include order size if different from default (0.01 units)
- Explain your reasoning in 1-2 sentences
- Express confidence level if uncertain

Example responses:
- "Place buy limit order at $94,850. The 15 bps bid imbalance and support at VWAP suggest a good entry."
- "Sell limit at $96,200 with size 0.02. Resistance forming with ask-side pressure building."
- "Hold position. Wide spreads and high volatility make entry risky currently."
"""

    # Template for market state + decision request
    TRADING_DECISION_TEMPLATE = """
{market_state}

Based on this market state, what limit order should be placed to achieve optimal execution?

Consider:
- Current spread and liquidity conditions
- Order book imbalance and pressure
- Recent price momentum and volatility
- Risk/reward of entering at different price levels

Decision:"""

    # Few-shot examples for better performance
    FEW_SHOT_EXAMPLES = [
        {
            "market_state": "Market state for BTC: The spread is tight at 2.50 basis points. Strong buying pressure with 35.0% bid-side imbalance. Deep liquidity with $5.2M within 1% of mid price. High trading volume of $8.5M in the last minute. Low volatility at 0.4% per hour. Strong upward momentum of 0.8% over 5 minutes.",
            "decision": "Place buy limit order at 3 basis points below mid price. Strong buying momentum with tight spreads indicates continuation likely, but placing slightly below mid ensures better fill probability without chasing.",
            "outcome": "Filled within 30 seconds, price continued up 0.5%"
        },
        {
            "market_state": "Market state for BTC: The spread is wide at 12.00 basis points. Strong selling pressure with 42.0% ask-side imbalance. Thin liquidity with $450,000 within 1% of mid price. Low trading volume of $250,000 in the last minute. Very high volatility at 3.2% per hour.",
            "decision": "Hold position. Wide spreads and thin liquidity during high volatility create poor risk/reward. Wait for market to stabilize before entering.",
            "outcome": "Market dropped 2% in next 5 minutes, then recovered"
        },
        {
            "market_state": "Market state for BTC: The spread is narrow at 4.00 basis points. The order book is balanced. Moderate liquidity with $2.1M within 1% of mid price. Moderate trading volume of $1.8M in the last minute. Moderate volatility at 0.9% per hour. Price is 0.3% above VWAP.",
            "decision": "Place sell limit order at 5 basis points above mid price with size 0.015. Balanced market with price above VWAP suggests mean reversion opportunity. Slightly larger size due to good liquidity.",
            "outcome": "Filled after 2 minutes, price reverted to VWAP"
        }
    ]

    # Template for including position and PnL context
    POSITION_CONTEXT_TEMPLATE = """
{market_state}

Current Trading Context:
- Position: {position}
- PnL: {pnl_display}
- Recent fills: {recent_fills}

Objective: {objective}

Based on this information, what is the optimal limit order to place?

Decision:"""

    # Templates for different market regimes
    REGIME_SPECIFIC_PROMPTS = {
        "high_volatility": "Market is experiencing high volatility. Focus on wider spreads and smaller position sizes to manage risk.",
        "low_liquidity": "Liquidity is thin. Consider using smaller order sizes and being more patient with price levels.",
        "trending": "Clear {direction} trend detected. Consider following momentum but avoid chasing extreme moves.",
        "ranging": "Market is range-bound. Look for mean reversion opportunities at range boundaries.",
        "news_driven": "Recent news event detected. Expect continued volatility and potential false signals."
    }

    @classmethod
    def create_trading_prompt(cls, 
                            market_state: str,
                            include_examples: bool = False,
                            position: Optional[str] = None,
                            pnl: Optional[float] = None) -> str:
        """
        Create a complete trading prompt
        
        Args:
            market_state: Encoded market state description
            include_examples: Whether to include few-shot examples
            position: Current position if any
            pnl: Current PnL if any
            
        Returns:
            Complete prompt for LLM
        """
        prompt_parts = []
        
        # Add few-shot examples if requested
        if include_examples:
            prompt_parts.append("Here are some example trading decisions:\n")
            for example in cls.FEW_SHOT_EXAMPLES[:2]:  # Use 2 examples
                prompt_parts.append(f"Market: {example['market_state']}")
                prompt_parts.append(f"Decision: {example['decision']}")
                prompt_parts.append(f"Outcome: {example['outcome']}\n")
        
        # Add current market state and decision request
        if position or pnl is not None:
            pnl_display = f"${pnl:,.2f}" if pnl else "$0.00"
            prompt = cls.POSITION_CONTEXT_TEMPLATE.format(
                market_state=market_state,
                position=position or "Flat",
                pnl_display=pnl_display,
                recent_fills="None",
                objective="Maximize execution quality and minimize market impact"
            )
        else:
            prompt = cls.TRADING_DECISION_TEMPLATE.format(market_state=market_state)
        
        prompt_parts.append(prompt)
        
        return "\n".join(prompt_parts)

    @classmethod
    def create_reward_prompt(cls, 
                            market_state: str,
                            action_taken: str,
                            outcome: str,
                            reward: float) -> str:
        """
        Create a prompt for DAPO training with reward signal
        
        Args:
            market_state: Market conditions when decision was made
            action_taken: The action that was taken
            outcome: What happened after the action
            reward: Numerical reward signal
            
        Returns:
            Training prompt with reward
        """
        return f"""Market State: {market_state}

Action Taken: {action_taken}

Outcome: {outcome}

Reward: {reward:.3f}

What would be the optimal action in this situation?"""

    @classmethod
    def create_preference_pair(cls,
                              market_state: str,
                              good_action: str,
                              bad_action: str,
                              good_outcome: str,
                              bad_outcome: str) -> Dict[str, str]:
        """
        Create preference pairs for DAPO training
        
        Args:
            market_state: Market conditions
            good_action: Preferred action
            bad_action: Non-preferred action
            good_outcome: Result of good action
            bad_outcome: Result of bad action
            
        Returns:
            Dictionary with chosen and rejected responses
        """
        return {
            "prompt": f"Market State: {market_state}\n\nWhat is the optimal trading action?",
            "chosen": f"{good_action}\nReasoning: This action led to {good_outcome}",
            "rejected": f"{bad_action}\nReasoning: This action led to {bad_outcome}"
        }


class PromptOptimizer:
    """Utilities for optimizing prompts based on performance"""
    
    def __init__(self):
        self.performance_history = []
    
    def add_result(self, prompt: str, response: str, reward: float):
        """Track prompt performance for optimization"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'prompt': prompt,
            'response': response,
            'reward': reward
        })
    
    def get_best_examples(self, n: int = 3) -> List[Dict]:
        """Get the best performing examples for few-shot learning"""
        sorted_history = sorted(self.performance_history, 
                              key=lambda x: x['reward'], 
                              reverse=True)
        return sorted_history[:n]
    
    def suggest_prompt_improvements(self) -> List[str]:
        """Suggest improvements based on performance patterns"""
        suggestions = []
        
        if not self.performance_history:
            return ["No performance data yet"]
        
        # Analyze average rewards
        avg_reward = sum(h['reward'] for h in self.performance_history) / len(self.performance_history)
        
        if avg_reward < 0:
            suggestions.append("Consider more conservative prompt instructions")
        elif avg_reward < 0.5:
            suggestions.append("Add more specific market regime guidance")
        
        # Check for specific failure patterns
        recent_failures = [h for h in self.performance_history[-10:] if h['reward'] < 0]
        if len(recent_failures) > 5:
            suggestions.append("Recent poor performance - review market conditions")
        
        return suggestions


if __name__ == "__main__":
    # Test prompt generation
    prompts = TradingPrompts()
    
    # Test basic trading prompt
    market_state = "Market state for BTC: The spread is narrow at 3.50 basis points. Moderate buying pressure with 15.0% bid-side imbalance."
    
    print("Basic Trading Prompt:")
    print("-" * 50)
    basic_prompt = prompts.create_trading_prompt(market_state)
    print(basic_prompt)
    
    print("\n\nWith Few-Shot Examples:")
    print("-" * 50)
    few_shot_prompt = prompts.create_trading_prompt(market_state, include_examples=True)
    print(few_shot_prompt)
    
    print("\n\nWith Position Context:")
    print("-" * 50)
    position_prompt = prompts.create_trading_prompt(
        market_state, 
        position="Long 0.05 BTC",
        pnl=1250.50
    )
    print(position_prompt)
    
    print("\n\nPreference Pair for DAPO:")
    print("-" * 50)
    preference = prompts.create_preference_pair(
        market_state=market_state,
        good_action="Place buy limit at 5 bps below mid",
        bad_action="Market buy immediately",
        good_outcome="Filled at better price with $50 savings",
        bad_outcome="Paid the spread and suffered $25 slippage"
    )
    print(f"Prompt: {preference['prompt']}")
    print(f"Chosen: {preference['chosen']}")
    print(f"Rejected: {preference['rejected']}")