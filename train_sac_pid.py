"""SAC-PID训练脚本"""

from chargax import (
    Chargax,
    get_electricity_prices,
)
from chargax.algorithms import build_sac_pid_trainer
from chargax.algorithms import sac_pid

import jax 
import jax.numpy as jnp
import time
import wandb
import chex
import numpy as np
from tqdm import tqdm

# ==================== 训练参数 ====================
SEED = 42
USER_PROFILES = "residential"
ARRIVAL_FREQUENCY = "medium"
CAR_PROFILES = "eu"
NUM_DC_GROUPS = 5
GROUPNAME = None
RUNTAG = None

# 训练配置
TOTAL_TIMESTEPS = 1000000

# Reward函数配置 (在环境中设置)
# 可选: "profit", "safety", "satisfaction", "balanced", "comprehensive"
REWARD_TYPE = "profit"

# Cost函数配置 (在环境中设置)
# 可选: "safety", "satisfaction", "safety_satisfaction", "comprehensive"
COST_TYPE = "safety"

# SAC-PID特有参数
COST_LIMIT = 25.0
PID_KP = 0.1
PID_KI = 0.01
PID_KD = 0.01
LAMBDA_LR = 0.035

# SAC参数
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 100000
BATCH_SIZE = 256
LEARNING_STARTS = 1000
GRADIENT_STEPS = 64

# 额外环境参数
ENV_PARAMETERS = {}
# ================================================


def create_baseline_rewards(env: Chargax, num_iterations=100):
    """创建随机和最大动作的baseline"""
    
    def step_env_random(carry, _):
        rng, obs, env_state, done, episode_reward = carry
        rng, action_key, step_key = jax.random.split(rng, 3)
        action = env.action_space.sample(action_key)
        (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
        episode_reward += reward
        done = jnp.logical_or(terminated, truncated)
        return (rng, obs, env_state, done, episode_reward), info

    def step_env_max(carry, _):
        rng, obs, env_state, done, episode_reward = carry
        rng, action_key, step_key = jax.random.split(rng, 3)
        max_action = env.action_space.nvec.max()
        action = np.ones_like(env.action_space.nvec) * max_action
        if env.include_battery:
            action[-1] = 0.0
        (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
        episode_reward += reward
        done = jnp.logical_or(terminated, truncated)
        return (rng, obs, env_state, done, episode_reward), info

    baseline_rewards = {}
    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    obs, env_state = env.reset(reset_key)
    
    for method in ["random_actions", "maximum_actions"]:
        baseline_rewards[method] = {
            "episode_rewards": np.zeros(num_iterations),
            "profit": np.zeros(num_iterations),
            "exceeded_capacity": np.zeros(num_iterations),
            "total_charged_kw": np.zeros(num_iterations),
            "total_discharged_kw": np.zeros(num_iterations),
            "rejected_customers": np.zeros(num_iterations),
            "uncharged_percentages": np.zeros(num_iterations),
            "uncharged_kw": np.zeros(num_iterations),
            "charged_overtime": np.zeros(num_iterations),
            "charged_undertime": np.zeros(num_iterations),
        }
        step_env = step_env_random if method == "random_actions" else step_env_max
        
        for i in range(num_iterations):
            (rng, obs, env_state, done, episode_reward), info = jax.lax.scan(
                step_env, 
                (rng, obs, env_state, False, 0.0), 
                length=env.episode_length
            )
            baseline_rewards[method]["episode_rewards"][i] = episode_reward
            baseline_rewards[method]["profit"][i] = info["logging_data"]["profit"][-1]
            baseline_rewards[method]["exceeded_capacity"][i] = info["logging_data"]["exceeded_capacity"][-1]
            baseline_rewards[method]["total_charged_kw"][i] = info["logging_data"]["total_charged_kw"][-1]
            baseline_rewards[method]["total_discharged_kw"][i] = info["logging_data"]["total_discharged_kw"][-1]
            baseline_rewards[method]["rejected_customers"][i] = info["logging_data"]["rejected_customers"][-1]
            baseline_rewards[method]["uncharged_percentages"][i] = info["logging_data"]["uncharged_percentages"][-1]
            baseline_rewards[method]["uncharged_kw"][i] = info["logging_data"]["uncharged_kw"][-1]
            baseline_rewards[method]["charged_overtime"][i] = info["logging_data"]["charged_overtime"][-1]
            baseline_rewards[method]["charged_undertime"][i] = info["logging_data"]["charged_undertime"][-1]
            
    baseline_rewards = jax.tree.map(lambda x: np.mean(x), baseline_rewards)
    return baseline_rewards


if __name__ == "__main__":
    env = Chargax(
        elec_grid_buy_price=get_electricity_prices("2023_NL"),
        elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,
        user_profiles=USER_PROFILES,
        arrival_frequency=ARRIVAL_FREQUENCY,
        car_profiles=CAR_PROFILES,
        num_dc_groups=NUM_DC_GROUPS,
        reward_type=REWARD_TYPE,  # reward函数类型
        cost_type=COST_TYPE,      # cost函数类型
        **ENV_PARAMETERS
    )

    baselines = create_baseline_rewards(env)
    
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
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "tau": TAU,
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_starts": LEARNING_STARTS,
            "gradient_steps": GRADIENT_STEPS,
        },
        baselines=baselines,
    )

    filtered_env_dict = {
        k: v for k, v in env.__dict__.items() if not isinstance(v, chex.Array)
    }
    merged_config = {**filtered_env_dict, **config.__dict__}

    start_time = time.time()
    print("开始JAX编译...")
    trainer_fn = jax.jit(trainer_fn).lower().compile()
    print(f"JAX编译完成，耗时 {(time.time() - start_time):.2f} 秒，开始训练...")

    groupname = GROUPNAME if GROUPNAME else f"{USER_PROFILES}_{ARRIVAL_FREQUENCY}{CAR_PROFILES}"
    if NUM_DC_GROUPS is not None:
        groupname += f"_{NUM_DC_GROUPS}"
    env_parameters_str = "_".join([f"{k}_{v}" for k, v in ENV_PARAMETERS.items()])
    groupname = f"{groupname}_{env_parameters_str}_sac_pid"
    
    tags = [RUNTAG, "sac-pid"] if RUNTAG else ["sac-pid"]
    wandb.init(project="chargax", config=merged_config, group=groupname, tags=tags, dir="./wandb")
    
    sac_pid._pbar = tqdm(total=config.num_iterations, desc="SAC-PID Training", unit="step")
    c_time = time.time()
    trained_runner_state, train_metrics = trainer_fn()
    sac_pid._pbar.close()
    
    print("训练完成")
    print(f"训练耗时 {time.time() - c_time:.2f} 秒")

    trained_state = trained_runner_state[0]
    print(f"最终拉格朗日乘子: {trained_state.lagrangian_multiplier:.4f}")
    print(f"最终熵系数alpha: {jnp.exp(trained_state.log_alpha):.4f}")

    wandb.finish()
