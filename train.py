"""统一训练脚本：支持 PPO, PPO-PID, SAC-PID"""
import jax
import jax.numpy as jnp
import time
import wandb
import chex
import numpy as np
from tqdm import tqdm

from chargax import (
    Chargax, 
    build_ppo_trainer, 
    build_ppo_pid_trainer, 
    build_sac_pid_trainer
)
from chargax.algorithms import ppo, ppo_pid, sac_pid
from env_config import create_env, get_env_info, get_groupname

# ==================== 1. 训练实验设置 ====================
SEED = 42
ALGORITHM = "ppo_pid"  # 可选: "ppo", "ppo_pid", "sac_pid"
TOTAL_TIMESTEPS = 1_000_000
RUNTAG = None

# ==================== 2. PID 约束参数 (适用于 ppo_pid, sac_pid) ====================
# A. 物理限制与缩放对齐 (解决量级不匹配问题)
PHYSICAL_LIMIT_KW = 25.0       # 物理意义：允许变压器过载 25.0 kW
COST_SCALE = 0.3               # 必须与 cost_functions.py 中的 scale 保持一致！
COST_LIMIT = PHYSICAL_LIMIT_KW * COST_SCALE  # 自动计算算法使用的 Limit

# B. PID 控制参数
PID_KP = 0.01                  # 比例系数 (P): 降低以避免初期过冲 (0.1 -> 0.01)
PID_KI = 1                     # 积分系数 (I): 主要靠积分消除稳态误差
PID_KD = 0.05                  # 微分系数 (D): 预测趋势，抑制震荡
LAMBDA_LR = 0.0005             # 乘子学习率
# ======================================================

if __name__ == "__main__":
    env = create_env()
    
    # 根据算法选择不同的构建器
    # 神经网络/强化学习相关的超参数已在各自算法文件的 Config 类中定义
    if ALGORITHM == "ppo":
        trainer_fn, config = build_ppo_trainer(
            env,
            config_params={
                "total_timesteps": TOTAL_TIMESTEPS,
                "seed": SEED
            }
        )
        pbar_module = ppo
        pbar_unit = "iter"
        
    elif ALGORITHM == "ppo_pid":
        trainer_fn, config = build_ppo_pid_trainer(
            env,
            config_params={
                "total_timesteps": TOTAL_TIMESTEPS,
                "seed": SEED,
                "cost_limit": COST_LIMIT,
                "pid_kp": PID_KP,
                "pid_ki": PID_KI,
                "pid_kd": PID_KD,
                "lambda_lr": LAMBDA_LR,
            }
        )
        pbar_module = ppo_pid
        pbar_unit = "iter"
        
    elif ALGORITHM == "sac_pid":
        trainer_fn, config, eval_func = build_sac_pid_trainer(
            env,
            config_params={
                "total_timesteps": TOTAL_TIMESTEPS,
                "seed": SEED,
                "cost_limit": COST_LIMIT,
                "pid_kp": PID_KP,
                "pid_ki": PID_KI,
                "pid_kd": PID_KD,
                "lambda_lr": LAMBDA_LR,
            }
        )
        pbar_module = sac_pid
        pbar_unit = "step"
    else:
        raise ValueError(f"不支持的算法: {ALGORITHM}")

    # 配置合并与 WandB 初始化
    filtered_env_dict = {
        k: v for k, v in env.__dict__.items() if not isinstance(v, chex.Array)
    }
    merged_config = {**filtered_env_dict, **config.__dict__, **get_env_info()}

    start_time = time.time()
    print(f"正在启动 {ALGORITHM} 训练...")
    print("开始 JAX 编译...")
    trainer_fn = jax.jit(trainer_fn).lower().compile()
    print(f"JAX 编译完成，耗时 {(time.time() - start_time):.2f} 秒，开始训练...")

    groupname = get_groupname(ALGORITHM)
    tags = [RUNTAG, ALGORITHM] if RUNTAG else [ALGORITHM]
    wandb.init(project="chargax", config=merged_config, group=groupname, tags=tags, dir="./wandb")

    # 进度条
    pbar_module._pbar = tqdm(total=config.num_iterations, desc=f"{ALGORITHM.upper()} Training", unit=pbar_unit)
    
    c_time = time.time()
    trained_runner_state, train_metrics = trainer_fn()
    pbar_module._pbar.close()

    print(f"\n训练完成，耗时 {time.time() - c_time:.2f} 秒")

    # 打印算法特定的最终状态
    if ALGORITHM in ["ppo_pid", "sac_pid"]:
        trained_state = trained_runner_state[0]
        print(f"最终拉格朗日乘子: {trained_state.pid_state.multipliers}")
        if ALGORITHM == "sac_pid":
            print(f"最终熵系数 alpha: {jnp.exp(trained_state.log_alpha):.4f}")

    wandb.finish()
