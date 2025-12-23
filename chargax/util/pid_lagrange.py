"""
PID Lagrangian Controller for Constrained RL

基于论文：Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"

设计说明：
==========

1. 输入信号 (current_costs)：
   - 应该是 episode 内各步 cost 的**平均值**，而非累加值
   - 范围通常在 [0, 1]（如果 cost 函数设计正确）

2. 约束阈值 (cost_limit)：
   - 表示"可接受的平均违反程度"
   - 例如：cost_limit = 0.05 意味着"允许平均 5% 的过载"

3. 误差定义：
   - error = current_costs - cost_limit
   - error > 0：超出约束，需要增加惩罚
   - error < 0：满足约束，可以减少惩罚

4. PID 公式：
   - P 项：Kp × error（瞬时响应）
   - I 项：累积的 Ki × error（消除稳态误差）
   - D 项：Kd × d(error)/dt（抑制振荡，默认关闭）
   - λ = max(0, I + P + D)

5. 乘子使用：
   - 在 Actor Loss 中：loss = -reward_advantage + λ × cost_advantage
   - λ 越大，策略越倾向于避免 cost
"""

import jax
import jax.numpy as jnp
import chex
from typing import NamedTuple


@chex.dataclass(frozen=True)
class PIDLagrangeConfig:
    """
    PID 控制器配置
    
    参数说明：
    - cost_limit: 约束阈值，表示可接受的平均 cost 上限 [0, 1]
    - pid_kp: 比例系数，控制对当前误差的瞬时响应强度
    - pid_ki: 积分系数，控制误差累积速度（较小值避免爆炸）
    - pid_kd: 微分系数，控制振荡抑制（默认 0 关闭）
    
    推荐调参：
    - cost 归一化到 [0, 1] 时：kp=0.5, ki=0.01, kd=0
    - cost_limit 根据任务设置（如 0.05 表示允许 5% 违规）
    """
    cost_limit: jnp.ndarray       # 约束阈值 [N]
    pid_kp: jnp.ndarray           # 比例系数 [N]
    pid_ki: jnp.ndarray           # 积分系数 [N]
    pid_kd: jnp.ndarray = 0.0     # 微分系数 [N]，默认关闭
    pid_d_delay: int = 10         # 微分项延迟步数
    pid_delta_d_ema_alpha: jnp.ndarray = 0.95  # D项EMA平滑系数
    lagrangian_multiplier_init: jnp.ndarray = 0.001  # 初始乘子值


@chex.dataclass
class PIDLagrangeState:
    """PID 控制器状态"""
    pid_i: jnp.ndarray            # 积分项 [N]
    error_history: jnp.ndarray    # 误差历史 [N, delay]
    history_index: int            # 循环队列索引
    delta_d_ema: jnp.ndarray      # D项 EMA [N]
    multipliers: jnp.ndarray      # 当前拉格朗日乘子 [N]


def init_pid_lagrange(config: PIDLagrangeConfig, num_channels: int) -> PIDLagrangeState:
    """初始化 PID 状态"""
    def to_array(x):
        x = jnp.atleast_1d(x)
        if x.shape[0] == 1 and num_channels > 1:
            return jnp.full((num_channels,), x[0])
        return x

    return PIDLagrangeState(
        pid_i=jnp.zeros(num_channels),
        error_history=jnp.zeros((num_channels, config.pid_d_delay)),
        history_index=0,
        delta_d_ema=jnp.zeros(num_channels),
        multipliers=to_array(config.lagrangian_multiplier_init)
    )


def update_pid_lagrange(
    state: PIDLagrangeState,
    config: PIDLagrangeConfig,
    current_costs: jnp.ndarray
) -> PIDLagrangeState:
    """
    更新 PID 乘子
    
    参数：
    - current_costs: 当前 episode 的**平均** cost [N]（不是累加！）
    
    公式：
    - error = current_costs - cost_limit
    - I_new = max(0, I_old + Ki × error)  # 积分项，带非负约束
    - P = Kp × error                       # 比例项
    - D = Kd × EMA(d_error/dt)            # 微分项（默认关闭）
    - λ = max(0, I + P + D)               # 最终乘子
    """
    def to_vec(x):
        x = jnp.atleast_1d(x)
        return x if x.shape == current_costs.shape else jnp.full_like(current_costs, x[0])

    cost_limits = to_vec(config.cost_limit)
    kps = to_vec(config.pid_kp)
    kis = to_vec(config.pid_ki)
    kds = to_vec(config.pid_kd)
    d_emas = to_vec(config.pid_delta_d_ema_alpha)

    # 1. 计算误差：正值表示超出约束
    errors = current_costs - cost_limits

    # 2. I 项：积分累积，带非负约束防止负积分
    new_pid_i = jnp.maximum(0.0, state.pid_i + kis * errors)
    
    # 3. P 项：当前误差的瞬时响应
    p_term = kps * errors
    
    # 4. D 项：误差变化率的 EMA（默认关闭，kd=0）
    next_index = (state.history_index + 1) % config.pid_d_delay
    prev_errors = state.error_history[:, state.history_index]
    derivatives = (errors - prev_errors) / config.pid_d_delay
    new_delta_d = d_emas * state.delta_d_ema + (1.0 - d_emas) * derivatives
    d_term = kds * new_delta_d
    
    # 5. 更新误差历史
    new_error_history = state.error_history.at[:, state.history_index].set(errors)
    
    # 6. 计算最终乘子：λ = max(0, I + P + D)
    new_multipliers = jnp.maximum(0.0, new_pid_i + p_term + d_term)
    
    return PIDLagrangeState(
        pid_i=new_pid_i,
        error_history=new_error_history,
        history_index=next_index,
        delta_d_ema=new_delta_d,
        multipliers=new_multipliers
    )
