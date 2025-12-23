"""
Reward Functions for Chargax Environment

简化设计：只关注利润，归一化处理。
"""

import jax.numpy as jnp
from typing import Dict
import chex


def profit(old_state, new_state, **kwargs) -> chex.Array:
    """
    利润奖励（归一化）
    
    单步利润通常在 -1€ ~ +2€ 范围，归一化后约在 [-1, 2]
    """
    profit_delta = new_state.profit - old_state.profit
    return profit_delta

