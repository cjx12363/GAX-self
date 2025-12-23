"""
Reward Functions for Chargax Environment

定义不同的reward函数，可以在训练时灵活切换。
所有 reward 都进行归一化处理，缩放到合适比例。
"""

import jax.numpy as jnp
from typing import Dict, Callable
import chex


# 归一化参考值（基于典型场景估算）
# 单步利润通常在 0~2€ 范围
PROFIT_NORMALIZER = 1.0


def profit(old_state, new_state, **kwargs) -> chex.Array:
    """仅利润（归一化）"""
    profit_delta = new_state.profit - old_state.profit
    return profit_delta / PROFIT_NORMALIZER


def profit_safety(old_state, new_state, alpha: float = 0.5, **kwargs) -> chex.Array:
    """利润 + 安全（归一化）"""
    profit_delta = (new_state.profit - old_state.profit) / PROFIT_NORMALIZER
    # exceeded_capacity 已在 cost function 中归一化处理，这里保持原始值用于 reward shaping
    exceeded_delta = new_state.exceeded_capacity - old_state.exceeded_capacity
    capacity = kwargs.get("transformer_capacity", 250.0)
    exceeded_ratio = exceeded_delta / capacity
    return profit_delta - alpha * exceeded_ratio


def profit_satisfaction(old_state, new_state, alpha: float = 0.1, **kwargs) -> chex.Array:
    """利润 + 满意度（归一化）"""
    profit_delta = (new_state.profit - old_state.profit) / PROFIT_NORMALIZER
    # uncharged_kw 归一化：相对于变压器容量
    capacity = kwargs.get("transformer_capacity", 250.0)
    uncharged_ratio = (new_state.uncharged_kw - old_state.uncharged_kw) / capacity
    return profit_delta - alpha * uncharged_ratio


def balanced(old_state, new_state, weights: Dict[str, float] = None, **kwargs) -> chex.Array:
    """平衡多目标（归一化）"""
    if weights is None:
        weights = {
            "exceeded_capacity": 0.5,
            "uncharged_kw": 0.1,
            "rejected_customers": 0.2,
            "charged_overtime": 0.05,
            "battery_degradation": 0.01
        }
    
    profit_delta = (new_state.profit - old_state.profit) / PROFIT_NORMALIZER
    capacity = kwargs.get("transformer_capacity", 250.0)
    
    penalties = 0.0
    if "exceeded_capacity" in weights:
        exceeded_ratio = (new_state.exceeded_capacity - old_state.exceeded_capacity) / capacity
        penalties += weights["exceeded_capacity"] * exceeded_ratio
    if "uncharged_kw" in weights:
        uncharged_ratio = (new_state.uncharged_kw - old_state.uncharged_kw) / capacity
        penalties += weights["uncharged_kw"] * uncharged_ratio
    if "rejected_customers" in weights:
        penalties += weights["rejected_customers"] * (new_state.rejected_customers - old_state.rejected_customers)
    if "charged_overtime" in weights:
        penalties += weights["charged_overtime"] * (new_state.charged_overtime - old_state.charged_overtime)
    if "battery_degradation" in weights:
        discharged_ratio = (new_state.total_discharged_kw - old_state.total_discharged_kw) / capacity
        penalties += weights["battery_degradation"] * discharged_ratio
    
    return profit_delta - penalties


def time_satisfaction(old_state, new_state, alpha: float = 0.1, beta: float = 0.5, **kwargs) -> chex.Array:
    """利润 + 时间满意度（归一化）"""
    profit_delta = (new_state.profit - old_state.profit) / PROFIT_NORMALIZER
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
    """全面reward（归一化）"""
    profit_delta = (new_state.profit - old_state.profit) / PROFIT_NORMALIZER
    capacity = kwargs.get("transformer_capacity", 250.0)
    
    # 归一化各项指标
    uncharged_ratio = (new_state.uncharged_kw - old_state.uncharged_kw) / capacity
    overtime_delta = new_state.charged_overtime - old_state.charged_overtime
    undertime_delta = new_state.charged_undertime - old_state.charged_undertime
    rejected_delta = new_state.rejected_customers - old_state.rejected_customers
    exceeded_ratio = (new_state.exceeded_capacity - old_state.exceeded_capacity) / capacity
    battery_ratio = (new_state.total_discharged_kw - old_state.total_discharged_kw) / capacity
    
    return profit_delta - (
        satisfaction_alpha * uncharged_ratio
        + time_alpha * (overtime_delta - beta * undertime_delta)
        + reject_alpha * rejected_delta
        + capacity_alpha * exceeded_ratio
        + battery_alpha * battery_ratio
    )



