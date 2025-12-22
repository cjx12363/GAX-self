import jax
import jax.numpy as jnp
import chex
from typing import NamedTuple

@chex.dataclass(frozen=True)
class PIDLagrangeConfig:
    """
    PID 控制器多通道配置。
    每个参数可以是标量（所有通道相同）或数组（各通道独立）。
    
    公式: lambda = max(0, I_term + Kp * current_error + Kd * deriv_error)
    """
    cost_limit: jnp.ndarray       # 对应各个通道的成本阈值 [N]
    pid_kp: jnp.ndarray           # 比例系数 [N]
    pid_ki: jnp.ndarray           # 积分系数 [N]
    pid_kd: jnp.ndarray = 0.0     # 微分系数 [N]，默认关闭
    pid_d_delay: int = 10         # 微分项延迟步数
    pid_delta_d_ema_alpha: jnp.ndarray = 0.95  # D项EMA平滑系数 [N]
    lagrangian_multiplier_init: jnp.ndarray = 0.001  # 初始乘子值 [N]

@chex.dataclass
class PIDLagrangeState:
    """PID 控制器内部状态 (支持多通道)"""
    pid_i: jnp.ndarray            # 积分项 [N]
    error_history: jnp.ndarray    # 误差历史 [N, delay]
    history_index: int            # 当前循环队列索引
    delta_d_ema: jnp.ndarray      # D项 EMA [N]
    multipliers: jnp.ndarray      # 当前拉格朗日乘子向量 [N]

def init_pid_lagrange(config: PIDLagrangeConfig, num_channels: int) -> PIDLagrangeState:
    """
    初始化多通道 PID 状态。
    """
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
    多通道并行更新 PID 乘子。
    
    公式: lambda = max(0, I_term + Kp * current_error + Kd * deriv_error)
    
    修复逻辑：
    - P项：直接使用当前误差的瞬时响应，无EMA平滑
    - I项：new_i = max(0, old_i + ki * error)，解耦积分更新
    - D项：保留EMA平滑，默认关闭(kd=0)
    """
    def to_vec(x):
        x = jnp.atleast_1d(x)
        return x if x.shape == current_costs.shape else jnp.full_like(current_costs, x[0])

    cost_limits = to_vec(config.cost_limit)
    kps = to_vec(config.pid_kp)
    kis = to_vec(config.pid_ki)
    kds = to_vec(config.pid_kd)
    d_emas = to_vec(config.pid_delta_d_ema_alpha)

    # 1. 计算当前误差
    errors = current_costs - cost_limits

    # 2. I项：解耦积分更新 new_i = max(0, old_i + ki * error)
    new_pid_i = jnp.maximum(0.0, state.pid_i + kis * errors)
    
    # 3. P项：直接使用当前误差，无EMA
    p_term = kps * errors
    
    # 4. D项：EMA平滑的误差变化率
    next_index = (state.history_index + 1) % config.pid_d_delay
    prev_errors = state.error_history[:, state.history_index]
    derivatives = (errors - prev_errors) / config.pid_d_delay
    new_delta_d = d_emas * state.delta_d_ema + (1.0 - d_emas) * derivatives
    d_term = kds * new_delta_d
    
    # 5. 更新误差历史
    new_error_history = state.error_history.at[:, state.history_index].set(errors)
    
    # 6. 计算最终乘子: lambda = max(0, I_term + Kp * error + Kd * deriv)
    new_multipliers = jnp.maximum(0.0, new_pid_i + p_term + d_term)
    
    return PIDLagrangeState(
        pid_i=new_pid_i,
        error_history=new_error_history,
        history_index=next_index,
        delta_d_ema=new_delta_d,
        multipliers=new_multipliers
    )
