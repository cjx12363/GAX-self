"""
Cost Functions for Constrained RL (PID-based algorithms)

设计原则（基于 Stooke et al. 2020 论文）：
============================================

1. Cost 定义：
   - Cost 是单步的约束违反程度
   - 类似于 reward，会被累积成 episode return
   - cost_limit 是对 **episode 累积 cost** 的上限约束

2. 数量级设计：
   - 为了让 PID 调参更容易，我们让单步 cost 在 [0, 1] 范围
   - 这样 episode 累积 cost（约 288 步）大约在 0~288 范围
   - cost_limit 通常设为 10~50（允许一定程度的违规）

3. 归一化方式：
   - cost = exceeded_capacity / transformer_capacity
   - 这样 100% 过载 = cost = 1.0
   - 无过载 = cost = 0

4. 与 Reward 的关系：
   - Reward: 单步利润 ~1€，episode 累积 ~300€
   - Cost: 单步过载率 ~0.1，episode 累积 ~30
   - 两者数量级相当，λ ≈ 10 时影响相当
"""

import jax.numpy as jnp
from typing import Dict
import chex


def safety(info: Dict, **kwargs) -> chex.Array:
    """
    变压器过载约束（归一化到 [0, 1]）
    
    公式：cost = exceeded_capacity / transformer_capacity
    
    含义：
    - 0.0: 无过载
    - 0.1: 过载了容量的 10%
    - 0.5: 过载了容量的 50%  
    - 1.0: 过载了容量的 100%
    
    Episode 累积示例：
    - 如果平均每步 cost = 0.1，288 步累积 = 28.8
    - cost_limit = 25 表示允许 episode 累积 cost 不超过 25
    """
    logging_data = info.get("logging_data", {})
    exceeded = logging_data.get("exceeded_capacity", 0.0)
    capacity = logging_data.get("transformer_capacity", 1.0)
    
    # 归一化到 [0, 1]
    normalized_cost = exceeded / jnp.maximum(capacity, 1.0)
    
    return jnp.clip(normalized_cost, 0.0, 1.0)
