"""
Reward Functions for Chargax Environment

定义不同的reward函数，可以在训练时灵活切换。
"""

import jax.numpy as jnp
from typing import Dict, Callable
import chex


def profit_only_reward(old_state, new_state, **kwargs) -> chex.Array:
    """
    仅利润奖励（默认）
    最大化充电站利润
    """
    return new_state.profit - old_state.profit


def profit_with_safety_reward(old_state, new_state, alpha: float = 0.5, **kwargs) -> chex.Array:
    """
    利润 + 安全约束
    惩罚变压器过载
    """
    profit_delta = new_state.profit - old_state.profit
    exceeded_delta = new_state.exceeded_capacity - old_state.exceeded_capacity
    return profit_delta - alpha * exceeded_delta


def profit_with_satisfaction_reward(old_state, new_state, alpha: float = 0.1, **kwargs) -> chex.Array:
    """
    利润 + 用户满意度
    惩罚未充满电量
    """
    profit_delta = new_state.profit - old_state.profit
    uncharged_delta = new_state.uncharged_kw - old_state.uncharged_kw
    return profit_delta - alpha * uncharged_delta


def balanced_reward(old_state, new_state, weights: Dict[str, float] = None, **kwargs) -> chex.Array:
    """
    平衡多目标的reward函数
    
    Args:
        weights: 各项权重，默认值:
            {
                "exceeded_capacity": 0.5,
                "uncharged_kw": 0.1,
                "rejected_customers": 0.2,
                "charged_overtime": 0.05,
                "battery_degradation": 0.01
            }
    """
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


def time_satisfaction_reward(old_state, new_state, alpha: float = 0.1, beta: float = 0.5, **kwargs) -> chex.Array:
    """
    利润 + 时间满意度
    惩罚超时充电，奖励提前完成
    
    Args:
        alpha: 时间满意度权重
        beta: 提前完成的奖励系数（相对于超时惩罚）
    """
    profit_delta = new_state.profit - old_state.profit
    overtime_delta = new_state.charged_overtime - old_state.charged_overtime
    undertime_delta = new_state.charged_undertime - old_state.charged_undertime
    return profit_delta - alpha * (overtime_delta - beta * undertime_delta)


def comprehensive_reward(old_state, new_state, 
                         capacity_alpha: float = 0.5,
                         satisfaction_alpha: float = 0.1,
                         time_alpha: float = 0.05,
                         reject_alpha: float = 0.2,
                         battery_alpha: float = 0.01,
                         beta: float = 0.5,
                         **kwargs) -> chex.Array:
    """
    全面的reward函数，与环境原始get_reward一致
    
    Args:
        capacity_alpha: 变压器过载惩罚权重
        satisfaction_alpha: 充电满意度权重
        time_alpha: 时间满意度权重
        reject_alpha: 拒绝客户惩罚权重
        battery_alpha: 电池损耗惩罚权重
        beta: 提前完成奖励系数
    """
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


# ============ 预定义的reward函数配置 ============

REWARD_PROFIT_ONLY = {
    "func": profit_only_reward,
    "description": "仅利润优化"
}

REWARD_PROFIT_SAFETY = {
    "func": profit_with_safety_reward,
    "params": {"alpha": 0.5},
    "description": "利润 + 安全约束"
}

REWARD_PROFIT_SATISFACTION = {
    "func": profit_with_satisfaction_reward,
    "params": {"alpha": 0.1},
    "description": "利润 + 用户满意度"
}

REWARD_BALANCED = {
    "func": balanced_reward,
    "params": {"weights": {
        "exceeded_capacity": 0.5,
        "uncharged_kw": 0.1,
        "rejected_customers": 0.2,
    }},
    "description": "平衡多目标"
}

REWARD_COMPREHENSIVE = {
    "func": comprehensive_reward,
    "params": {
        "capacity_alpha": 0.5,
        "satisfaction_alpha": 0.1,
        "time_alpha": 0.05,
        "reject_alpha": 0.2,
        "battery_alpha": 0.01,
        "beta": 0.5
    },
    "description": "全面reward（与环境原始一致）"
}


REWARD_PRESETS = {
    "profit": REWARD_PROFIT_ONLY,
    "safety": REWARD_PROFIT_SAFETY,
    "satisfaction": REWARD_PROFIT_SATISFACTION,
    "balanced": REWARD_BALANCED,
    "comprehensive": REWARD_COMPREHENSIVE,
}


def get_reward_function(name: str = "profit") -> tuple:
    """
    获取预定义的reward函数
    
    Args:
        name: 预设名称 ("profit", "safety", "satisfaction", "balanced", "comprehensive")
    
    Returns:
        (reward_func, params, description)
    """
    preset = REWARD_PRESETS.get(name, REWARD_PROFIT_ONLY)
    return preset["func"], preset.get("params", {}), preset["description"]


def create_custom_reward(weights: Dict[str, float]) -> Callable:
    """
    创建自定义的reward函数
    
    Args:
        weights: 各惩罚项的权重
    
    Returns:
        reward函数
    
    Example:
        reward_fn = create_custom_reward({
            "exceeded_capacity": 1.0,
            "uncharged_kw": 0.2,
            "rejected_customers": 0.5
        })
    """
    def custom_reward(old_state, new_state, **kwargs) -> chex.Array:
        return balanced_reward(old_state, new_state, weights=weights)
    return custom_reward
