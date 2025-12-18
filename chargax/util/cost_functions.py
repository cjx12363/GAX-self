"""
Cost Functions for Constrained RL (SAC-PID, PPO-PID)

定义不同的cost函数，用于约束优化。
cost > 0 表示违反约束，PID控制器会增加拉格朗日乘子来惩罚这种行为。
"""

import jax.numpy as jnp
from typing import Dict, Callable
import chex


def exceeded_capacity_cost(info: Dict, **kwargs) -> chex.Array:
    """
    变压器过载cost
    当充电功率超过变压器容量时产生cost
    """
    logging_data = info.get("logging_data", {})
    return logging_data.get("exceeded_capacity", 0.0)


def uncharged_kw_cost(info: Dict, **kwargs) -> chex.Array:
    """
    未充满电量cost
    用户离开时未充满的电量（kWh）
    """
    logging_data = info.get("logging_data", {})
    return logging_data.get("uncharged_kw", 0.0)


def uncharged_percentage_cost(info: Dict, **kwargs) -> chex.Array:
    """
    未充满百分比cost
    用户离开时未充满的百分比
    """
    logging_data = info.get("logging_data", {})
    return logging_data.get("uncharged_percentages", 0.0)


def rejected_customers_cost(info: Dict, **kwargs) -> chex.Array:
    """
    拒绝客户cost
    因充电桩满而被拒绝的客户数
    """
    logging_data = info.get("logging_data", {})
    return logging_data.get("rejected_customers", 0.0)


def charged_overtime_cost(info: Dict, **kwargs) -> chex.Array:
    """
    超时充电cost
    用户等待超过预期时间
    """
    logging_data = info.get("logging_data", {})
    return logging_data.get("charged_overtime", 0.0)


def battery_degradation_cost(info: Dict, **kwargs) -> chex.Array:
    """
    电池损耗cost
    使用放电量作为电池损耗的代理指标
    """
    logging_data = info.get("logging_data", {})
    return logging_data.get("total_discharged_kw", 0.0)


def combined_cost(info: Dict, weights: Dict[str, float] = None, **kwargs) -> chex.Array:
    """
    组合多个cost的加权和
    
    Args:
        info: 环境返回的info字典
        weights: 各cost的权重，例如:
            {
                "exceeded_capacity": 1.0,
                "uncharged_kw": 0.1,
                "rejected_customers": 0.5
            }
    """
    if weights is None:
        weights = {"exceeded_capacity": 1.0}
    
    total_cost = 0.0
    logging_data = info.get("logging_data", {})
    
    for key, weight in weights.items():
        value = logging_data.get(key, 0.0)
        total_cost = total_cost + weight * value
    
    return total_cost


# ============ 预定义的cost函数配置 ============

# 仅考虑安全约束（变压器过载）
COST_SAFETY_ONLY = {
    "func": exceeded_capacity_cost,
    "limit": 25.0,  # cost_limit阈值
    "description": "仅变压器过载约束"
}

# 仅考虑用户满意度
COST_SATISFACTION_ONLY = {
    "func": uncharged_kw_cost,
    "limit": 50.0,
    "description": "仅用户满意度约束（未充满电量）"
}

# 安全 + 满意度组合
COST_SAFETY_AND_SATISFACTION = {
    "func": lambda info, **kw: combined_cost(info, weights={
        "exceeded_capacity": 1.0,
        "uncharged_kw": 0.1
    }),
    "limit": 30.0,
    "description": "安全+满意度组合约束"
}

# 全面约束
COST_COMPREHENSIVE = {
    "func": lambda info, **kw: combined_cost(info, weights={
        "exceeded_capacity": 1.0,
        "uncharged_kw": 0.1,
        "rejected_customers": 0.5,
        "charged_overtime": 0.05
    }),
    "limit": 50.0,
    "description": "全面约束（安全+满意度+拒绝+超时）"
}


# ============ 获取cost函数的工具 ============

COST_PRESETS = {
    "safety": COST_SAFETY_ONLY,
    "satisfaction": COST_SATISFACTION_ONLY,
    "safety_satisfaction": COST_SAFETY_AND_SATISFACTION,
    "comprehensive": COST_COMPREHENSIVE,
}


def get_cost_function(name: str = "safety") -> tuple:
    """
    获取预定义的cost函数
    
    Args:
        name: 预设名称 ("safety", "satisfaction", "safety_satisfaction", "comprehensive")
    
    Returns:
        (cost_func, cost_limit, description)
    """
    preset = COST_PRESETS.get(name, COST_SAFETY_ONLY)
    return preset["func"], preset["limit"], preset["description"]


def create_custom_cost(weights: Dict[str, float], limit: float = 25.0) -> Callable:
    """
    创建自定义的组合cost函数
    
    Args:
        weights: 各指标的权重
        limit: cost_limit阈值
    
    Returns:
        cost函数
    
    Example:
        cost_fn = create_custom_cost({
            "exceeded_capacity": 1.0,
            "uncharged_kw": 0.2
        }, limit=30.0)
    """
    def custom_cost(info: Dict, **kwargs) -> chex.Array:
        return combined_cost(info, weights=weights)
    return custom_cost
