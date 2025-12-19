"""PPO训练脚本"""

import jax
import jax.numpy as jnp
import time
import wandb
import chex
import numpy as np
from tqdm import tqdm

from chargax import Chargax, get_electricity_prices, build_random_trainer, build_ppo_trainer
from chargax.algorithms import ppo
from env_config import create_env, get_env_info, get_groupname

# ==================== 训练参数 ====================
SEED = 42
ALGORITHM = "ppo"              # 可选: "ppo", "random"
TOTAL_TIMESTEPS = 1_000_000
RUNTAG = None
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
    env = create_env()
    baselines = create_baseline_rewards(env)

    if ALGORITHM == "ppo":
        trainer_fn, config = build_ppo_trainer(
            env,
            config_params={
                "total_timesteps": TOTAL_TIMESTEPS,
                "seed": SEED
            },
            baselines=baselines
        )
    elif ALGORITHM == "random":
        trainer_fn, config = build_random_trainer(
            env,
            config_params={
                "total_timesteps": TOTAL_TIMESTEPS,
                "seed": SEED
            },
            baselines=baselines
        )
    else:
        raise ValueError(f"未知算法: {ALGORITHM}")

    filtered_env_dict = {
        k: v for k, v in env.__dict__.items() if not isinstance(v, chex.Array)
    }
    merged_config = {**filtered_env_dict, **config.__dict__, **get_env_info()}

    start_time = time.time()
    print("开始JAX编译...")
    trainer_fn = jax.jit(trainer_fn).lower().compile()
    print(f"JAX编译完成，耗时 {(time.time() - start_time):.2f} 秒，开始训练...")

    groupname = get_groupname(ALGORITHM)
    tags = [RUNTAG, ALGORITHM] if RUNTAG else [ALGORITHM]
    wandb.init(project="chargax", config=merged_config, group=groupname, tags=tags, dir="./wandb")

    ppo._pbar = tqdm(total=config.num_iterations, desc="Training", unit="iter")
    c_time = time.time()
    trained_runner_state, train_rewards = trainer_fn()
    ppo._pbar.close()

    print("训练完成")
    print(f"训练耗时 {time.time() - c_time:.2f} 秒")

    wandb.finish()
