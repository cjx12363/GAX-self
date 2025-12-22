"""
Cost Functions for Constrained RL (SAC-PID, PPO-PID)

定义不同的cost函数，用于约束优化。
cost > 0 表示违反约束，PID控制器会增加拉格朗日乘子来惩罚这种行为。
"""

import jax.numpy as jnp
from typing import Dict, Callable
import chex


def safety(info: Dict, scale: float = 0.3, **kwargs) -> chex.Array:
    """变压器过载 (默认缩放 0.1)"""
    logging_data = info.get("logging_data", {})
    return logging_data.get("exceeded_capacity", 0.0) * scale


def satisfaction(info: Dict, scale: float = 0.1, **kwargs) -> chex.Array:
    """未充满电量 (默认缩放 0.1)"""
    logging_data = info.get("logging_data", {})
    return logging_data.get("uncharged_kw", 0.0) * scale


def satisfaction_pct(info: Dict, **kwargs) -> chex.Array:
    """未充满百分比"""
    logging_data = info.get("logging_data", {})
    return logging_data.get("uncharged_percentages", 0.0)


def rejected(info: Dict, **kwargs) -> chex.Array:
    """拒绝客户数"""
    logging_data = info.get("logging_data", {})
    return logging_data.get("rejected_customers", 0.0)


def overtime(info: Dict, **kwargs) -> chex.Array:
    """超时充电"""
    logging_data = info.get("logging_data", {})
    return logging_data.get("charged_overtime", 0.0)


def battery_degradation(info: Dict, **kwargs) -> chex.Array:
    """电池损耗"""
    logging_data = info.get("logging_data", {})
    return logging_data.get("total_discharged_kw", 0.0)


def safety_satisfaction(info: Dict, **kwargs) -> chex.Array:
    """安全+满意度"""
    logging_data = info.get("logging_data", {})
    return (
        logging_data.get("exceeded_capacity", 0.0) 
        + 0.1 * logging_data.get("uncharged_kw", 0.0)
    )


def comprehensive(info: Dict, scale: float = 0.1, **kwargs) -> chex.Array:
    """全面约束 (带整体缩放)"""
    logging_data = info.get("logging_data", {})
    raw_cost = (
        logging_data.get("exceeded_capacity", 0.0)
        + 0.1 * logging_data.get("uncharged_kw", 0.0)
        + 0.5 * logging_data.get("rejected_customers", 0.0)
    )
    return raw_cost * scale
