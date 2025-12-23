"""
Cost Functions for Constrained RL (PID-based algorithms)

设计原则：
1. Cost 归一化到 [0, 1] 范围：表示"当前步的约束违反程度"
2. Cost = 0：完全满足约束
3. Cost = 1：严重违反约束（如变压器 100% 过载）

PID 控制器使用方式：
- 输入：episode 内各步 cost 的**平均值**
- cost_limit：期望的平均违反程度上限（如 0.05 表示平均 5% 过载）
- error = mean_cost - cost_limit
- 当 error > 0 时，PID 增加惩罚，引导策略减少违规
"""

import jax.numpy as jnp
from typing import Dict
import chex


def safety(info: Dict, **kwargs) -> chex.Array:
    """
    变压器过载约束（归一化到 [0, 1]）
    
    公式：cost = min(1, exceeded_capacity / transformer_capacity)
    
    含义：
    - 0.0: 未过载
    - 0.1: 过载了容量的 10%
    - 0.5: 过载了容量的 50%
    - 1.0: 过载了 100% 或更多（封顶）
    
    PID 用法示例：
    - cost_limit = 0.05 表示"允许平均 5% 过载"
    - 如果 episode 平均 cost = 0.1，则 error = 0.05，PID 会增加惩罚
    """
    logging_data = info.get("logging_data", {})
    exceeded = logging_data.get("exceeded_capacity", 0.0)
    capacity = logging_data.get("transformer_capacity", 1.0)
    
    # 归一化：过载量 / 容量，封顶到 1.0
    normalized_cost = exceeded / jnp.maximum(capacity, 1.0)
    
    return jnp.clip(normalized_cost, 0.0, 1.0)
