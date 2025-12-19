import jax
import jax.numpy as jnp
import chex
from typing import NamedTuple

@chex.dataclass(frozen=True)
class PIDLagrangeConfig:
    """
    PID 控制器多通道配置。
    每个参数可以是标量（所有通道相同）或数组（各通道独立）。
    """
    cost_limit: jnp.ndarray       # 对应各个通道的成本阈值 [N]
    pid_kp: jnp.ndarray           # 比例系数 [N]
    pid_ki: jnp.ndarray           # 积分系数 [N]
    pid_kd: jnp.ndarray           # 微分系数 [N]
    pid_d_delay: int = 10         # 微分项延迟步数
    pid_delta_p_ema_alpha: jnp.ndarray = 0.95  # P项EMA平滑系数 [N]
    pid_delta_d_ema_alpha: jnp.ndarray = 0.95  # D项EMA平滑系数 [N]
    lambda_lr: jnp.ndarray = 0.035             # 乘子更新步长 [N]
    lagrangian_multiplier_init: jnp.ndarray = 0.001  # 初始乘子值 [N]

@chex.dataclass
class PIDLagrangeState:
    """PID 控制器内部状态 (支持多通道)"""
    pid_i: jnp.ndarray            # 积分项 [N]
    error_history: jnp.ndarray     # 误差历史 [N, delay]
    history_index: int            # 当前循环队列索引
    delta_p_ema: jnp.ndarray      # P项 EMA [N]
    delta_d_ema: jnp.ndarray      # D项 EMA [N]
    multipliers: jnp.ndarray       # 当前拉格朗日乘子向量 [N]

def init_pid_lagrange(config: PIDLagrangeConfig, num_channels: int) -> PIDLagrangeState:
    """
    初始化多通道 PID 状态。
    """
    # 辅助函数：确保参数是 (num_channels,) 维度的数组
    def to_array(x):
        x = jnp.atleast_1d(x)
        if x.shape[0] == 1 and num_channels > 1:
            return jnp.full((num_channels,), x[0])
        return x

    return PIDLagrangeState(
        pid_i=jnp.zeros(num_channels),
        error_history=jnp.zeros((num_channels, config.pid_d_delay)),
        history_index=0,
        delta_p_ema=jnp.zeros(num_channels),
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
    
    Args:
        state: 当前 PID 状态
        config: PID 配置参数
        current_costs: 当前周期的平均成本向量 [N]
        
    Returns:
        new_state: 更新后的 PID 状态
    """
    # 1. 计算当前误差
    def to_vec(x):
        x = jnp.atleast_1d(x)
        return x if x.shape == current_costs.shape else jnp.full_like(current_costs, x[0])

    cost_limits = to_vec(config.cost_limit)
    kps = to_vec(config.pid_kp)
    kis = to_vec(config.pid_ki)
    kds = to_vec(config.pid_kd)
    p_emas = to_vec(config.pid_delta_p_ema_alpha)
    d_emas = to_vec(config.pid_delta_d_ema_alpha)
    lrs = to_vec(config.lambda_lr)

    errors = current_costs - cost_limits

    # 2. 更新积分项 pid_i (位置式控制的核心累积量)
    # I_t+1 = max(0, I_t + lr * Ki * error)
    new_pid_i = jnp.maximum(0.0, state.pid_i + lrs * kis * errors)
    
    # 3. P项更新 (EMA 平滑误差)
    delta_p = p_emas * state.delta_p_ema + (1.0 - p_emas) * errors
    
    # 4. D项更新 (利用循环队列计算延迟差值)
    next_index = (state.history_index + 1) % config.pid_d_delay
    prev_errors = state.error_history[:, state.history_index]
    
    derivatives = (errors - prev_errors) / config.pid_d_delay
    delta_d = d_emas * state.delta_d_ema + (1.0 - d_emas) * derivatives
    
    # 5. 更新误差历史
    new_error_history = state.error_history.at[:, state.history_index].set(errors)
    
    # 6. 计算最终响应式乘子 lambda
    # lambda = max(0, I + Kp * P + Kd * D)
    # 注意：这里的 I 已经包含了步长 lr，而 P 和 D 通常直接作用于实时误差响应
    # 我们保持 Kp, Kd 的物理意义为瞬时增益
    new_multipliers = jnp.maximum(0.0, new_pid_i + kps * delta_p + kds * delta_d)
    
    return PIDLagrangeState(
        pid_i=new_pid_i,
        error_history=new_error_history,
        history_index=next_index,
        delta_p_ema=delta_p,
        delta_d_ema=delta_d,
        multipliers=new_multipliers
    )
    
    return PIDLagrangeState(
        pid_i=new_pid_i,
        error_history=new_error_history,
        history_index=next_index,
        delta_p_ema=delta_p,
        delta_d_ema=delta_d,
        multipliers=new_multipliers
    )
