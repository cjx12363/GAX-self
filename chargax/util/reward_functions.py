"""
Reward Functions for Chargax Environment

定义不同的reward函数，可以在训练时灵活切换。
"""

import jax.numpy as jnp
from typing import Dict, Callable
import chex


def profit(old_state, new_state, **kwargs) -> chex.Array:
    """仅利润"""
    return new_state.profit - old_state.profit


def profit_safety(old_state, new_state, alpha: float = 0.5, **kwargs) -> chex.Array:
    """利润 + 安全"""
    profit_delta = new_state.profit - old_state.profit
    exceeded_delta = new_state.exceeded_capacity - old_state.exceeded_capacity
    return profit_delta - alpha * exceeded_delta


def profit_satisfaction(old_state, new_state, alpha: float = 0.1, **kwargs) -> chex.Array:
    """利润 + 满意度"""
    profit_delta = new_state.profit - old_state.profit
    uncharged_delta = new_state.uncharged_kw - old_state.uncharged_kw
    return profit_delta - alpha * uncharged_delta


def balanced(old_state, new_state, weights: Dict[str, float] = None, **kwargs) -> chex.Array:
    """平衡多目标"""
    if weights is None:
        weights = {
            "exceeded_capacity": 0.5,
            "uncharged_kw": 0.1,
            "rejected_customers": 0.2,
            "charged_overtime": 0.05,
            "battery_degradation": 0.01
        }
    
    profit_delta = new_state.profit - old_state.profit
    
    penalties = 0.0
    if "exceeded_capacity" in weights:
        penalties += weights["exceeded_capacity"] * (new_state.exceeded_capacity - old_state.exceeded_capacity)
    if "uncharged_kw" in weights:
        penalties += weights["uncharged_kw"] * (new_state.uncharged_kw - old_state.uncharged_kw)
    if "rejected_customers" in weights:
        penalties += weights["rejected_customers"] * (new_state.rejected_customers - old_state.rejected_customers)
    if "charged_overtime" in weights:
        penalties += weights["charged_overtime"] * (new_state.charged_overtime - old_state.charged_overtime)
    if "battery_degradation" in weights:
        penalties += weights["battery_degradation"] * (new_state.total_discharged_kw - old_state.total_discharged_kw)
    
    return profit_delta - penalties


def time_satisfaction(old_state, new_state, alpha: float = 0.1, beta: float = 0.5, **kwargs) -> chex.Array:
    """利润 + 时间满意度"""
    profit_delta = new_state.profit - old_state.profit
    overtime_delta = new_state.charged_overtime - old_state.charged_overtime
    undertime_delta = new_state.charged_undertime - old_state.charged_undertime
    return profit_delta - alpha * (overtime_delta - beta * undertime_delta)


def comprehensive(old_state, new_state, 
                  capacity_alpha: float = 0.5,
                  satisfaction_alpha: float = 0.1,
                  time_alpha: float = 0.05,
                  reject_alpha: float = 0.2,
                  battery_alpha: float = 0.01,
                  beta: float = 0.5,
                  **kwargs) -> chex.Array:
    """全面reward"""
    profit_delta = new_state.profit - old_state.profit
    
    uncharged_delta = new_state.uncharged_kw - old_state.uncharged_kw
    overtime_delta = new_state.charged_overtime - old_state.charged_overtime
    undertime_delta = new_state.charged_undertime - old_state.charged_undertime
    rejected_delta = new_state.rejected_customers - old_state.rejected_customers
    exceeded_delta = new_state.exceeded_capacity - old_state.exceeded_capacity
    battery_delta = new_state.total_discharged_kw - old_state.total_discharged_kw
    
    return profit_delta - (
        satisfaction_alpha * uncharged_delta
        + time_alpha * (overtime_delta - beta * undertime_delta)
        + reject_alpha * rejected_delta
        + capacity_alpha * exceeded_delta
        + battery_alpha * battery_delta
    )



