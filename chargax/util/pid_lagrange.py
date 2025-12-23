"""
PID Lagrangian Controller for Constrained RL

基于论文：Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods" (ICML 2020)

核心设计：
=========

1. 约束形式：
   - 目标：E[Σc(s,a)] ≤ d（期望 episode 累积 cost 不超过阈值 d）
   - cost_limit (d) 是对 **episode 累积 cost** 的上限

2. PID 更新公式：
   - 传统 Lagrangian：λ ← λ + lr * (J_C - d)  （只有 I 项）
   - PID Lagrangian：λ = max(0, I + Kp*e + Kd*ė)
   
   其中：
   - e = J_C - d（累积 cost 与阈值的差）
   - I ← max(0, I + Ki*e)（积分项，累积误差）
   - Kp*e（比例项，瞬时响应）
   - Kd*ė（微分项，预测响应）

3. 论文关键洞察：
   - 传统方法只有 I 项，导致振荡和超调
   - 添加 P 项提供即时响应（阻尼）
   - 添加 D 项预测趋势（预测控制）
"""

import jax
import jax.numpy as jnp
import chex
from typing import NamedTuple


@chex.dataclass(frozen=True)
class PIDLagrangeConfig:
    """
    PID 控制器配置
    
    关键参数：
    - cost_limit: episode 累积 cost 的上限（如 25.0）
    - pid_kp: 比例系数（0.1 ~ 1.0）
    - pid_ki: 积分系数（0.001 ~ 0.01）
    - pid_kd: 微分系数（默认 0，通常不需要）
    """
    cost_limit: jnp.ndarray               # Episode 累积 cost 上限
    pid_kp: jnp.ndarray = 0.1             # 比例系数
    pid_ki: jnp.ndarray = 0.001           # 积分系数
    pid_kd: jnp.ndarray = 0.0             # 微分系数
    pid_d_delay: int = 10                 # D 项延迟步数
    pid_delta_d_ema_alpha: jnp.ndarray = 0.95
    lagrangian_multiplier_init: jnp.ndarray = 0.0  # 初始乘子（论文建议从 0 开始）


@chex.dataclass
class PIDLagrangeState:
    """PID 控制器状态"""
    pid_i: jnp.ndarray            # 积分项（累积的 Ki*error）
    prev_cost: jnp.ndarray        # 上一次的累积 cost（用于计算 D 项）
    delta_d_ema: jnp.ndarray      # D 项的 EMA
    multipliers: jnp.ndarray      # 当前拉格朗日乘子


def init_pid_lagrange(config: PIDLagrangeConfig, num_channels: int) -> PIDLagrangeState:
    """初始化 PID 状态"""
    def to_array(x):
        x = jnp.atleast_1d(x)
        if x.shape[0] == 1 and num_channels > 1:
            return jnp.full((num_channels,), x[0])
        return x

    return PIDLagrangeState(
        pid_i=jnp.zeros(num_channels),
        prev_cost=jnp.zeros(num_channels),
        delta_d_ema=jnp.zeros(num_channels),
        multipliers=to_array(config.lagrangian_multiplier_init)
    )


def update_pid_lagrange(
    state: PIDLagrangeState,
    config: PIDLagrangeConfig,
    episode_cost: jnp.ndarray
) -> PIDLagrangeState:
    """
    更新 PID 乘子
    
    参数：
    - episode_cost: 当前 episode 的 **累积** cost [N]
      （如 288 步的累积值，范围约 0~288 如果单步 cost 在 [0,1]）
    
    论文公式：
    - error = episode_cost - cost_limit
    - I_new = max(0, I_old + Ki * error)
    - P = Kp * error
    - D = Kd * EMA(error - prev_error)
    - λ = max(0, I + P + D)
    """
    def to_vec(x):
        x = jnp.atleast_1d(x)
        return x if x.shape == episode_cost.shape else jnp.full_like(episode_cost, x[0])

    cost_limits = to_vec(config.cost_limit)
    kps = to_vec(config.pid_kp)
    kis = to_vec(config.pid_ki)
    kds = to_vec(config.pid_kd)
    d_emas = to_vec(config.pid_delta_d_ema_alpha)

    # 1. 误差：累积 cost 与阈值的差
    errors = episode_cost - cost_limits

    # 2. I 项：累积误差（带非负约束，防止过度补偿）
    new_pid_i = jnp.maximum(0.0, state.pid_i + kis * errors)
    
    # 3. P 项：当前误差的即时响应
    p_term = kps * errors
    
    # 4. D 项：误差变化的预测（EMA 平滑）
    error_change = episode_cost - state.prev_cost
    new_delta_d = d_emas * state.delta_d_ema + (1.0 - d_emas) * error_change
    d_term = kds * new_delta_d
    
    # 5. 最终乘子：λ = max(0, I + P + D)
    new_multipliers = jnp.maximum(0.0, new_pid_i + p_term + d_term)
    
    return PIDLagrangeState(
        pid_i=new_pid_i,
        prev_cost=episode_cost,
        delta_d_ema=new_delta_d,
        multipliers=new_multipliers
    )
