"""
PPO-PID: PPO with PID Lagrangian for Constrained Optimization

该算法结合了PPO和PID控制的拉格朗日方法，用于处理带约束的强化学习问题。
PID控制器动态调整拉格朗日乘子，以满足成本约束。

参考文献:
- Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import chex
import optax
from typing import NamedTuple, List
from tqdm import tqdm

from chargax import Chargax, LogWrapper, NormalizeVecObservation
from chargax.algorithms.networks import ActorNetworkMultiDiscrete, CriticNetwork
import wandb

_pbar = None


def create_ppo_pid_networks(
    key,
    in_shape: int,
    actor_features: List[int],
    critic_features: List[int],
    actions_nvec: int,
):
    """创建PPO-PID网络 (actor + value critic + cost critic)"""
    actor_key, critic_key, cost_critic_key = jax.random.split(key, 3)
    actor = ActorNetworkMultiDiscrete(actor_key, in_shape, actor_features, actions_nvec)
    critic = CriticNetwork(critic_key, in_shape, critic_features)
    cost_critic = CriticNetwork(cost_critic_key, in_shape, critic_features)
    return actor, critic, cost_critic


@chex.dataclass(frozen=True)
class PPOPIDConfig:
    """PPO-PID配置"""
    # PPO基础参数
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 100.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0
    ent_coef: float = 0.01
    vf_coef: float = 0.25
    
    # PID拉格朗日参数
    cost_limit: float = 25.0  # 成本约束阈值
    lagrangian_multiplier_init: float = 0.001  # 拉格朗日乘子初始值
    lambda_lr: float = 0.035  # 拉格朗日乘子学习率
    lambda_optimizer: str = "adam"  # 拉格朗日乘子优化器
    
    # PID控制器参数
    pid_kp: float = 0.1  # 比例系数
    pid_ki: float = 0.01  # 积分系数
    pid_kd: float = 0.01  # 微分系数
    pid_d_delay: int = 10  # 微分项延迟步数
    pid_delta_p_ema_alpha: float = 0.95  # P项EMA平滑系数
    pid_delta_d_ema_alpha: float = 0.95  # D项EMA平滑系数
    
    # 训练参数
    total_timesteps: int = 5e6
    num_envs: int = 12
    num_steps: int = 300
    num_minibatches: int = 4
    update_epochs: int = 4
    
    seed: int = 42
    debug: bool = False
    evaluate_deterministically: bool = False

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.num_steps // self.num_envs)
    
    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def batch_size(self):
        return self.minibatch_size * self.num_minibatches


@chex.dataclass(frozen=True)
class Transition:
    """存储转换数据"""
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    cost: chex.Array  # 成本信号
    done: chex.Array
    value: chex.Array
    cost_value: chex.Array  # 成本价值估计
    log_prob: chex.Array
    info: chex.Array


@chex.dataclass
class PIDState:
    """PID控制器状态"""
    pid_i: float  # 积分项
    cost_ds: chex.Array  # 用于计算微分项的历史成本
    cost_d_index: int  # 当前索引
    delta_p_ema: float  # P项EMA
    delta_d_ema: float  # D项EMA


class TrainState(NamedTuple):
    """训练状态"""
    actor: eqx.Module
    critic: eqx.Module
    cost_critic: eqx.Module
    optimizer_state: optax.OptState
    lagrangian_multiplier: float
    pid_state: PIDState


def build_ppo_pid_trainer(
    env: Chargax,
    config_params: dict = {},
    baselines: dict = {},
):
    """构建PPO-PID训练器"""
    
    env = LogWrapper(env)
    env = NormalizeVecObservation(env)
    observation_space = env.observation_space
    action_space = env.action_space
    logging_baselines = baselines

    config = PPOPIDConfig(**config_params)

    rng = jax.random.PRNGKey(config.seed)
    rng, network_key, reset_key = jax.random.split(rng, 3)

    # 创建网络
    actor, critic, cost_critic = create_ppo_pid_networks(
        key=network_key,
        in_shape=observation_space.shape[0],
        actor_features=[256, 256],
        critic_features=[256, 256],
        actions_nvec=action_space.nvec
    )

    # 优化器
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_iterations
        )
        return config.learning_rate * frac
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_lr else config.learning_rate,
            eps=1e-5
        ),
    )
    optimizer_state = optimizer.init({
        "actor": actor,
        "critic": critic,
        "cost_critic": cost_critic
    })

    # 初始化PID状态
    pid_state = PIDState(
        pid_i=0.0,
        cost_ds=jnp.zeros(config.pid_d_delay),
        cost_d_index=0,
        delta_p_ema=0.0,
        delta_d_ema=0.0
    )

    train_state = TrainState(
        actor=actor,
        critic=critic,
        cost_critic=cost_critic,
        optimizer_state=optimizer_state,
        lagrangian_multiplier=config.lagrangian_multiplier_init,
        pid_state=pid_state
    )

    rng, key = jax.random.split(rng)
    reset_key = jax.random.split(key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_key)

    def pid_update(pid_state: PIDState, ep_cost_avg: float) -> tuple:
        """PID控制器更新拉格朗日乘子
        
        error = cost - cost_limit
        P项: Kp * error
        I项: Ki * ∫error dt
        D项: Kd * d(error)/dt = Kd * (error - prev_error)
        """
        error = ep_cost_avg - config.cost_limit
        
        # P项 (带EMA平滑)
        delta_p = config.pid_delta_p_ema_alpha * pid_state.delta_p_ema + \
                  (1 - config.pid_delta_p_ema_alpha) * error
        
        # I项: 累积误差
        pid_i = pid_state.pid_i + error * config.pid_ki
        
        # D项: 误差变化率 (带延迟和EMA平滑)
        # 存储历史error用于计算微分
        error_history = pid_state.cost_ds.at[pid_state.cost_d_index].set(error)
        next_index = (pid_state.cost_d_index + 1) % config.pid_d_delay
        
        # derivative = (error - prev_error) / delay
        prev_error = pid_state.cost_ds[next_index]
        derivative = (error - prev_error) / config.pid_d_delay
        delta_d = config.pid_delta_d_ema_alpha * pid_state.delta_d_ema + \
                  (1 - config.pid_delta_d_ema_alpha) * derivative
        
        # PID输出: Kp * P + Ki * I + Kd * D
        pid_delta = config.pid_kp * delta_p + pid_i + config.pid_kd * delta_d
        
        # 更新PID状态
        new_pid_state = PIDState(
            pid_i=pid_i,
            cost_ds=error_history,  # 存储error而非cost
            cost_d_index=next_index,
            delta_p_ema=delta_p,
            delta_d_ema=delta_d
        )
        
        return new_pid_state, pid_delta

    def eval_func(train_state, rng):
        """评估函数"""
        def step_env(carry):
            rng, obs, env_state, done, episode_reward, episode_cost = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            action_dist = train_state.actor(obs)
            if config.evaluate_deterministically:
                action = jnp.argmax(action_dist.logits, axis=-1)
            else:
                action = action_dist.sample(seed=action_key)
            (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
            done = jnp.logical_or(terminated, truncated)
            episode_reward += reward
            # 从info中获取cost，如果没有则默认为0
            cost = info.get("cost", 0.0)
            episode_cost += cost
            return (rng, obs, env_state, done, episode_reward, episode_cost)
        
        def cond_func(carry):
            _, _, _, done, _, _ = carry
            return jnp.logical_not(done)
        
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key)
        done = False
        episode_reward = 0.0
        episode_cost = 0.0

        rng, obs, env_state, done, episode_reward, episode_cost = jax.lax.while_loop(
            cond_func, step_env, (rng, obs, env_state, done, episode_reward, episode_cost)
        )

        return episode_reward, episode_cost

    def train_func(rng=rng):
        """训练函数"""
        
        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            action_dist = jax.vmap(train_state.actor)(last_obs)
            value = jax.vmap(train_state.critic)(last_obs)
            cost_value = jax.vmap(train_state.cost_critic)(last_obs)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)

            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, config.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_key, env_state, action)
            done = jnp.logical_or(terminated, truncated)
            
            # 获取cost信号，如果环境没有提供则默认为0
            cost = info.get("cost", jnp.zeros_like(reward))

            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                cost=cost,
                done=done,
                value=value,
                cost_value=cost_value,
                log_prob=log_prob,
                info=info
            )

            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition
        
        def _calculate_gae(gae_and_next_value, transition):
            """计算reward的GAE"""
            gae, next_value = gae_and_next_value
            value, reward, done = (
                transition.value,
                transition.reward,
                transition.done,
            )
            delta = reward + config.gamma * next_value * (1 - done) - value
            gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
            return (gae, value), (gae, gae + value)
        
        def _calculate_cost_gae(gae_and_next_value, transition):
            """计算cost的GAE"""
            gae, next_value = gae_and_next_value
            cost_value, cost, done = (
                transition.cost_value,
                transition.cost,
                transition.done,
            )
            delta = cost + config.gamma * next_value * (1 - done) - cost_value
            gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
            return (gae, cost_value), (gae, gae + cost_value)
        
        def _update_epoch(update_state, _):
            """执行一个epoch的更新"""

            @eqx.filter_value_and_grad(has_aux=True)
            def __ppo_pid_loss_fn(params, trajectory_minibatch, advantages, returns, 
                                   cost_advantages, cost_returns, lagrangian_multiplier):
                action_dist = jax.vmap(params["actor"])(trajectory_minibatch.observation)
                log_prob = action_dist.log_prob(trajectory_minibatch.action).sum(axis=-1)
                entropy = action_dist.entropy().mean()
                value = jax.vmap(params["critic"])(trajectory_minibatch.observation)
                cost_value = jax.vmap(params["cost_critic"])(trajectory_minibatch.observation)

                # Actor loss (带拉格朗日约束)
                ratio = jnp.exp(log_prob - trajectory_minibatch.log_prob.sum(axis=-1))
                
                # 标准化advantages
                _advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                _cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
                
                # 组合reward和cost的advantages
                combined_advantages = _advantages - lagrangian_multiplier * _cost_advantages
                
                actor_loss1 = combined_advantages * ratio
                actor_loss2 = (
                    jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) 
                    * combined_advantages
                )
                actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

                # Value loss
                value_pred_clipped = trajectory_minibatch.value + jnp.clip(
                    value - trajectory_minibatch.value, -config.clip_coef_vf, config.clip_coef_vf
                )
                value_losses = jnp.square(value - returns)
                value_losses_clipped = jnp.square(value_pred_clipped - returns)
                value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                # Cost value loss
                cost_value_pred_clipped = trajectory_minibatch.cost_value + jnp.clip(
                    cost_value - trajectory_minibatch.cost_value, 
                    -config.clip_coef_vf, config.clip_coef_vf
                )
                cost_value_losses = jnp.square(cost_value - cost_returns)
                cost_value_losses_clipped = jnp.square(cost_value_pred_clipped - cost_returns)
                cost_value_loss = jnp.maximum(cost_value_losses, cost_value_losses_clipped).mean()

                # Total loss
                total_loss = (
                    actor_loss 
                    + config.vf_coef * value_loss
                    + config.vf_coef * cost_value_loss
                    - config.ent_coef * entropy
                )
                return total_loss, (actor_loss, value_loss, cost_value_loss, entropy)
            
            def __update_over_minibatch(carry, minibatch):
                train_state, lagrangian_multiplier = carry
                trajectory_mb, advantages_mb, returns_mb, cost_advantages_mb, cost_returns_mb = minibatch
                
                (total_loss, aux), grads = __ppo_pid_loss_fn(
                    {
                        "actor": train_state.actor,
                        "critic": train_state.critic,
                        "cost_critic": train_state.cost_critic
                    }, 
                    trajectory_mb, advantages_mb, returns_mb,
                    cost_advantages_mb, cost_returns_mb, lagrangian_multiplier
                )
                
                updates, optimizer_state = optimizer.update(grads, train_state.optimizer_state)
                new_networks = optax.apply_updates({
                    "actor": train_state.actor,
                    "critic": train_state.critic,
                    "cost_critic": train_state.cost_critic
                }, updates)
                
                train_state = TrainState(
                    actor=new_networks["actor"],
                    critic=new_networks["critic"],
                    cost_critic=new_networks["cost_critic"],
                    optimizer_state=optimizer_state,
                    lagrangian_multiplier=train_state.lagrangian_multiplier,
                    pid_state=train_state.pid_state
                )
                return (train_state, lagrangian_multiplier), (total_loss, aux)
            
            train_state, trajectory_batch, advantages, returns, cost_advantages, cost_returns, rng = update_state
            rng, key = jax.random.split(rng)

            batch_idx = jax.random.permutation(key, config.batch_size)
            batch = (trajectory_batch, advantages, returns, cost_advantages, cost_returns)
            
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
            )
            
            (train_state, _), (total_loss, aux) = jax.lax.scan(
                __update_over_minibatch, 
                (train_state, train_state.lagrangian_multiplier), 
                minibatches
            )
            
            update_state = (train_state, trajectory_batch, advantages, returns, 
                           cost_advantages, cost_returns, rng)
            return update_state, (total_loss, aux)

        def train_step(runner_state, _):
            # Rollout
            runner_state, trajectory_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            train_state, env_state, last_obs, rng = runner_state
            
            # 计算reward GAE
            last_value = jax.vmap(train_state.critic)(last_obs)
            _, (advantages, returns) = jax.lax.scan(
                _calculate_gae,
                (jnp.zeros_like(last_value), last_value),
                trajectory_batch,
                reverse=True,
                unroll=16
            )
            
            # 计算cost GAE
            last_cost_value = jax.vmap(train_state.cost_critic)(last_obs)
            _, (cost_advantages, cost_returns) = jax.lax.scan(
                _calculate_cost_gae,
                (jnp.zeros_like(last_cost_value), last_cost_value),
                trajectory_batch,
                reverse=True,
                unroll=16
            )
    
            # 更新epochs
            update_state = (train_state, trajectory_batch, advantages, returns, 
                           cost_advantages, cost_returns, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            train_state = update_state[0]
            
            # 计算episode平均cost并更新PID
            ep_cost_avg = trajectory_batch.cost.sum() / config.num_envs
            new_pid_state, pid_delta = pid_update(train_state.pid_state, ep_cost_avg)
            
            # 更新拉格朗日乘子 (确保非负)
            new_lagrangian = jnp.maximum(
                0.0, 
                train_state.lagrangian_multiplier + config.lambda_lr * pid_delta
            )
            
            train_state = TrainState(
                actor=train_state.actor,
                critic=train_state.critic,
                cost_critic=train_state.cost_critic,
                optimizer_state=train_state.optimizer_state,
                lagrangian_multiplier=new_lagrangian,
                pid_state=new_pid_state
            )
            
            metric = trajectory_batch.info
            metric["loss_info"] = loss_info
            metric["lagrangian_multiplier"] = new_lagrangian
            metric["ep_cost_avg"] = ep_cost_avg
            rng = update_state[-1]

            rng, eval_key = jax.random.split(rng)
            eval_rewards, eval_cost = eval_func(train_state, eval_key)
            metric["eval_rewards"] = eval_rewards
            metric["eval_cost"] = eval_cost

            def callback(info):
                global _pbar
                if _pbar is not None:
                    timestep = int(info["train_timestep"][-1][0] * config.num_envs)
                    _pbar.update(1)
                    _pbar.set_postfix({
                        "timestep": timestep, 
                        "eval_reward": f"{info['eval_rewards']:.2f}",
                        "eval_cost": f"{info['eval_cost']:.2f}",
                        "lambda": f"{info['lagrangian_multiplier']:.4f}"
                    })
                if config.debug:
                    print(f'timestep={(info["train_timestep"][-1][0] * config.num_envs)}, '
                          f'eval_rewards={info["eval_rewards"]}, '
                          f'eval_cost={info["eval_cost"]}, '
                          f'lambda={info["lagrangian_multiplier"]}')
                if wandb.run:
                    if "logging_data" not in info:
                        info["logging_data"] = {}
                    finished_episodes = info["returned_episode"]
                    if finished_episodes.any():
                        info["logging_data"] = jax.tree.map(
                            lambda x: x[finished_episodes].mean(), info["logging_data"]
                        )
                        wandb.log({
                            "timestep": info["train_timestep"][-1][0] * config.num_envs,
                            "eval_rewards": info["eval_rewards"],
                            "eval_cost": info["eval_cost"],
                            "lagrangian_multiplier": info["lagrangian_multiplier"],
                            "ep_cost_avg": info["ep_cost_avg"],
                            **info["logging_data"],
                            **logging_baselines
                        })

            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, _

        rng, key = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, key)
        trained_runner_state, train_metrics = jax.lax.scan(
            train_step, runner_state, None, config.num_iterations
        )

        return trained_runner_state, train_metrics

    return train_func, config
