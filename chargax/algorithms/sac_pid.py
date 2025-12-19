"""
SAC-PID: Soft Actor-Critic with PID Lagrangian for Constrained Optimization

该算法结合了SAC和PID控制的拉格朗日方法，用于处理带约束的强化学习问题。
PID控制器动态调整拉格朗日乘子，以满足成本约束。

参考文献:
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"
- Stooke et al., "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import chex
import optax
from typing import NamedTuple, List, Tuple
from functools import partial

from chargax import Chargax, LogWrapper, NormalizeVecObservation
from chargax.util.pid_lagrange import (
    PIDLagrangeConfig, 
    PIDLagrangeState, 
    init_pid_lagrange, 
    update_pid_lagrange
)
import wandb

_pbar = None


class SACActorMultiDiscrete(eqx.Module):
    """SAC多离散动作空间的Actor网络"""
    layers: list
    output_heads: list

    def __init__(self, key, in_shape, hidden_layers: List[int], actions_nvec):
        import distrax
        if isinstance(actions_nvec, chex.Array):
            actions_nvec = actions_nvec.tolist()

        keys = jax.random.split(key, len(hidden_layers) + 1)
        self.layers = [eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])]
        for i in range(len(hidden_layers) - 1):
            self.layers.append(
                eqx.nn.Linear(hidden_layers[i], hidden_layers[i + 1], key=keys[i + 1])
            )

        head_keys = jax.random.split(keys[-1], len(actions_nvec))
        self.output_heads = [
            eqx.nn.Linear(hidden_layers[-1], action, key=head_keys[i])
            for i, action in enumerate(actions_nvec)
        ]
        if len(set(actions_nvec)) == 1:
            self.output_heads = jax.tree_util.tree_map(
                lambda *v: jnp.stack(v), *self.output_heads
            )

    def __call__(self, x):
        import distrax
        def forward(head, x):
            return head(x)
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        logits = jax.vmap(forward, in_axes=(0, None))(self.output_heads, x)
        return distrax.Categorical(logits=logits)


class QNetworkMultiDiscrete(eqx.Module):
    """SAC的Q网络，支持多离散动作空间"""
    layers: list
    output_heads: list

    out_size: int = 1

    def __init__(self, key, in_shape, hidden_layers: List[int], actions_nvec, out_size: int = 1):
        if isinstance(actions_nvec, chex.Array):
            actions_nvec = actions_nvec.tolist()
        keys = jax.random.split(key, len(hidden_layers) + 1)
        self.layers = [eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])]
        for i in range(len(hidden_layers) - 1):
            self.layers.append(
                eqx.nn.Linear(hidden_layers[i], hidden_layers[i + 1], key=keys[i + 1])
            )
        head_keys = jax.random.split(keys[-1], len(actions_nvec))
        self.output_heads = [
            eqx.nn.Linear(hidden_layers[-1], action * out_size, key=head_keys[i])
            for i, action in enumerate(actions_nvec)
        ]
        self.out_size = out_size
        if len(set(actions_nvec)) == 1:
            self.output_heads = jax.tree_util.tree_map(
                lambda *v: jnp.stack(v), *self.output_heads
            )

    def __call__(self, x):
        def forward(head, x):
            return head(x)
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        out = jax.vmap(forward, in_axes=(0, None))(self.output_heads, x)
        if self.out_size > 1:
            # reshape [num_heads, action * out_size] -> [num_heads, action, out_size]
            out = out.reshape(out.shape[0], -1, self.out_size)
        return out


def create_sac_pid_networks(key, in_shape, actor_features, critic_features, actions_nvec, num_costs: int = 1):
    """创建SAC-PID网络 (支持多通道 cost critic)"""
    keys = jax.random.split(key, 5)
    actor = SACActorMultiDiscrete(keys[0], in_shape, actor_features, actions_nvec)
    q1 = QNetworkMultiDiscrete(keys[1], in_shape, critic_features, actions_nvec)
    q2 = QNetworkMultiDiscrete(keys[2], in_shape, critic_features, actions_nvec)
    q1_cost = QNetworkMultiDiscrete(keys[3], in_shape, critic_features, actions_nvec, out_size=num_costs)
    q2_cost = QNetworkMultiDiscrete(keys[4], in_shape, critic_features, actions_nvec, out_size=num_costs)
    q1_target = jax.tree.map(lambda x: x, q1)
    q2_target = jax.tree.map(lambda x: x, q2)
    q1_cost_target = jax.tree.map(lambda x: x, q1_cost)
    q2_cost_target = jax.tree.map(lambda x: x, q2_cost)
    return actor, q1, q2, q1_target, q2_target, q1_cost, q2_cost, q1_cost_target, q2_cost_target


@chex.dataclass(frozen=True)
class SACPIDConfig:
    """SAC-PID配置"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    autotune_alpha: bool = True
    target_entropy_scale: float = 0.89
    
    # PID拉格朗日参数
    num_costs: int = 1
    cost_limit: Union[float, jnp.ndarray] = 25.0
    lagrangian_multiplier_init: Union[float, jnp.ndarray] = 0.001
    lambda_lr: Union[float, jnp.ndarray] = 0.035
    pid_kp: Union[float, jnp.ndarray] = 0.1
    pid_ki: Union[float, jnp.ndarray] = 0.01
    pid_kd: Union[float, jnp.ndarray] = 0.01
    pid_d_delay: int = 10
    pid_delta_p_ema_alpha: Union[float, jnp.ndarray] = 0.95
    pid_delta_d_ema_alpha: Union[float, jnp.ndarray] = 0.95
    
    # 训练参数 (类似PPO批量模式)
    total_timesteps: int = 1000000
    num_envs: int = 12
    num_steps: int = 300
    buffer_size: int = 100000
    batch_size: int = 256
    learning_starts: int = 1000
    gradient_steps: int = 64
    
    seed: int = 42
    debug: bool = False
    evaluate_deterministically: bool = False

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.num_steps // self.num_envs)


@chex.dataclass
class ReplayBuffer:
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    costs: chex.Array
    next_obs: chex.Array
    dones: chex.Array
    pos: int
    size: int


@chex.dataclass
class PIDState:
    """已弃用"""
    pass


class TrainState(NamedTuple):
    actor: eqx.Module
    q1: eqx.Module
    q2: eqx.Module
    q1_target: eqx.Module
    q2_target: eqx.Module
    q1_cost: eqx.Module
    q2_cost: eqx.Module
    q1_cost_target: eqx.Module
    q2_cost_target: eqx.Module
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState
    q_cost_opt_state: optax.OptState
    log_alpha: chex.Array
    alpha_opt_state: optax.OptState
    pid_state: PIDLagrangeState


@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    cost: chex.Array
    done: chex.Array
    info: chex.Array


def build_sac_pid_trainer(env: Chargax, config_params: dict = {}):
    """
    构建SAC-PID训练器 (批量模式，类似PPO)
    
    Args:
        env: Chargax环境 (cost通过env.cost_type配置)
        config_params: 配置参数
        baselines: 基线数据用于wandb日志
    
    Note:
        reward和cost函数通过环境配置:
        - env.reward_type: "profit", "safety", "satisfaction", "balanced", "comprehensive"
        - env.cost_type: "safety", "satisfaction", "safety_satisfaction", "comprehensive"
    """
    env = LogWrapper(env)
    env = NormalizeVecObservation(env)
    obs_space = env.observation_space
    act_space = env.action_space

    config = SACPIDConfig(**config_params)

    rng = jax.random.PRNGKey(config.seed)
    rng, net_key = jax.random.split(rng)

    # 创建网络
    (actor, q1, q2, q1_target, q2_target, 
     q1_cost, q2_cost, q1_cost_target, q2_cost_target) = create_sac_pid_networks(
        net_key, obs_space.shape[0], [256, 256], [256, 256], act_space.nvec, num_costs=config.num_costs)

    # 优化器
    actor_opt = optax.adam(config.learning_rate)
    q_opt = optax.adam(config.learning_rate)
    q_cost_opt = optax.adam(config.learning_rate)
    alpha_opt = optax.adam(config.learning_rate)

    num_actions = act_space.nvec[0]
    target_entropy = -config.target_entropy_scale * jnp.log(1.0 / num_actions) * len(act_space.nvec)
    log_alpha = jnp.array(jnp.log(config.alpha), dtype=jnp.float32)

    # 初始化通用 PID
    pid_util_config = PIDLagrangeConfig(
        cost_limit=jnp.atleast_1d(config.cost_limit),
        pid_kp=jnp.atleast_1d(config.pid_kp),
        pid_ki=jnp.atleast_1d(config.pid_ki),
        pid_kd=jnp.atleast_1d(config.pid_kd),
        pid_d_delay=config.pid_d_delay,
        pid_delta_p_ema_alpha=jnp.atleast_1d(config.pid_delta_p_ema_alpha),
        pid_delta_d_ema_alpha=jnp.atleast_1d(config.pid_delta_d_ema_alpha),
        lambda_lr=jnp.atleast_1d(config.lambda_lr),
        lagrangian_multiplier_init=jnp.atleast_1d(config.lagrangian_multiplier_init)
    )

    train_state = TrainState(
        actor=actor, q1=q1, q2=q2, q1_target=q1_target, q2_target=q2_target,
        q1_cost=q1_cost, q2_cost=q2_cost, q1_cost_target=q1_cost_target, q2_cost_target=q2_cost_target,
        actor_opt_state=actor_opt.init(actor),
        q_opt_state=q_opt.init({"q1": q1, "q2": q2}),
        q_cost_opt_state=q_cost_opt.init({"q1_cost": q1_cost, "q2_cost": q2_cost}),
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt.init(log_alpha),
        pid_state=init_pid_lagrange(pid_util_config, config.num_costs)
    )

    # Replay Buffer
    buffer = ReplayBuffer(
        obs=jnp.zeros((config.buffer_size,) + obs_space.shape),
        actions=jnp.zeros((config.buffer_size, len(act_space.nvec)), dtype=jnp.int32),
        rewards=jnp.zeros(config.buffer_size),
        costs=jnp.zeros((config.buffer_size, config.num_costs)),
        next_obs=jnp.zeros((config.buffer_size,) + obs_space.shape),
        dones=jnp.zeros(config.buffer_size, dtype=jnp.bool_),
        pos=0, size=0
    )

    rng, key = jax.random.split(rng)
    reset_keys = jax.random.split(key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset)(reset_keys)


    def add_batch_to_buffer(buffer, obs, actions, rewards, costs, next_obs, dones):
        """批量添加到buffer"""
        batch_size = obs.shape[0] * obs.shape[1]  # num_steps * num_envs
        obs_flat = obs.reshape((batch_size,) + obs.shape[2:])
        actions_flat = actions.reshape((batch_size,) + actions.shape[2:])
        rewards_flat = rewards.reshape(batch_size)
        costs_flat = costs.reshape(batch_size, -1)
        next_obs_flat = next_obs.reshape((batch_size,) + next_obs.shape[2:])
        dones_flat = dones.reshape(batch_size)
        
        def add_one(carry, data):
            buf, pos = carry
            o, a, r, c, no, d = data
            buf = ReplayBuffer(
                obs=buf.obs.at[pos].set(o),
                actions=buf.actions.at[pos].set(a),
                rewards=buf.rewards.at[pos].set(r),
                costs=buf.costs.at[pos].set(c),
                next_obs=buf.next_obs.at[pos].set(no),
                dones=buf.dones.at[pos].set(d),
                pos=(pos + 1) % config.buffer_size,
                size=jnp.minimum(buf.size + 1, config.buffer_size)
            )
            return (buf, buf.pos), None
        
        (new_buffer, _), _ = jax.lax.scan(
            add_one, (buffer, buffer.pos),
            (obs_flat, actions_flat, rewards_flat, costs_flat, next_obs_flat, dones_flat)
        )
        return new_buffer

    def sample_buffer(buffer, key):
        indices = jax.random.randint(key, (config.batch_size,), 0, buffer.size)
        return (buffer.obs[indices], buffer.actions[indices], buffer.rewards[indices],
                buffer.costs[indices], buffer.next_obs[indices], buffer.dones[indices])

    def soft_update(target, online, tau):
        return jax.tree.map(lambda t, o: tau * o + (1 - tau) * t, target, online)

    # pid_update 逻辑已迁至 util.pid_lagrange


    def update_critic(ts, batch, rng):
        obs, actions, rewards, costs, next_obs, dones = batch
        alpha = jnp.exp(ts.log_alpha)

        next_dist = jax.vmap(ts.actor)(next_obs)
        next_probs = jax.nn.softmax(next_dist.logits, axis=-1)
        next_q1 = jax.vmap(ts.q1_target)(next_obs)
        next_q2 = jax.vmap(ts.q2_target)(next_obs)
        next_q_min = jnp.minimum(next_q1, next_q2)
        next_q1_cost = jax.vmap(ts.q1_cost_target)(next_obs)
        next_q2_cost = jax.vmap(ts.q2_cost_target)(next_obs)
        next_q_cost_min = jnp.minimum(next_q1_cost, next_q2_cost)
        
        log_probs = jnp.log(next_probs + 1e-8)
        entropy = -jnp.sum(next_probs * log_probs, axis=-1)
        next_v = jnp.sum(next_probs * next_q_min, axis=-1).sum(axis=-1) + alpha * entropy.sum(axis=-1)
        # next_v_cost: [batch, num_costs]
        next_v_cost = jnp.sum(next_probs[..., None] * next_q_cost_min, axis=-2).sum(axis=-2)
        
        target_q = rewards + config.gamma * (1 - dones) * next_v
        # target_q_cost: [batch, num_costs]
        target_q_cost = costs + config.gamma * (1 - dones[..., None]) * next_v_cost

        @eqx.filter_value_and_grad
        def q_loss_fn(params):
            q1_pred = jax.vmap(params["q1"])(obs)
            q2_pred = jax.vmap(params["q2"])(obs)
            batch_idx = jnp.arange(obs.shape[0])
            q1_vals = jnp.array([q1_pred[batch_idx, d, actions[:, d]] for d in range(actions.shape[1])]).sum(0)
            q2_vals = jnp.array([q2_pred[batch_idx, d, actions[:, d]] for d in range(actions.shape[1])]).sum(0)
            return jnp.mean((q1_vals - target_q)**2) + jnp.mean((q2_vals - target_q)**2)

        @eqx.filter_value_and_grad
        def q_cost_loss_fn(params):
            q1c = jax.vmap(params["q1_cost"])(obs)
            q2c = jax.vmap(params["q2_cost"])(obs)
            batch_idx = jnp.arange(obs.shape[0])
            # q1c_vals: [batch, num_costs]
            q1c_vals = jnp.array([q1c[batch_idx, d, actions[:, d]] for d in range(actions.shape[1])]).sum(0)
            q2c_vals = jnp.array([q2c[batch_idx, d, actions[:, d]] for d in range(actions.shape[1])]).sum(0)
            return jnp.mean((q1c_vals - target_q_cost)**2) + jnp.mean((q2c_vals - target_q_cost)**2)

        q_loss, q_grads = q_loss_fn({"q1": ts.q1, "q2": ts.q2})
        qc_loss, qc_grads = q_cost_loss_fn({"q1_cost": ts.q1_cost, "q2_cost": ts.q2_cost})
        
        q_updates, q_opt_state = q_opt.update(q_grads, ts.q_opt_state)
        new_q = optax.apply_updates({"q1": ts.q1, "q2": ts.q2}, q_updates)
        qc_updates, qc_opt_state = q_cost_opt.update(qc_grads, ts.q_cost_opt_state)
        new_qc = optax.apply_updates({"q1_cost": ts.q1_cost, "q2_cost": ts.q2_cost}, qc_updates)

        return ts._replace(q1=new_q["q1"], q2=new_q["q2"], q1_cost=new_qc["q1_cost"], 
                          q2_cost=new_qc["q2_cost"], q_opt_state=q_opt_state, q_cost_opt_state=qc_opt_state), q_loss


    def update_actor(ts, batch):
        obs = batch[0]
        alpha = jnp.exp(ts.log_alpha)
        lam = ts.pid_state.multipliers # [num_costs]

        @eqx.filter_value_and_grad
        def actor_loss_fn(actor):
            dist = jax.vmap(actor)(obs)
            probs = jax.nn.softmax(dist.logits, axis=-1)
            log_probs = jnp.log(probs + 1e-8)
            q1 = jax.vmap(ts.q1)(obs)
            q2 = jax.vmap(ts.q2)(obs)
            q_min = jnp.minimum(q1, q2)
            q1c = jax.vmap(ts.q1_cost)(obs)
            q2c = jax.vmap(ts.q2_cost)(obs)
            qc_min = jnp.minimum(q1c, q2c)
            exp_q = jnp.sum(probs * q_min, axis=-1).sum(axis=-1)
            # exp_qc: [batch, num_costs]
            exp_qc = jnp.sum(probs[..., None] * qc_min, axis=-2).sum(axis=-2)
            entropy = -jnp.sum(probs * log_probs, axis=-1).sum(axis=-1)
            # lam @ exp_qc: 向量内积
            return jnp.mean(-exp_q + jnp.dot(exp_qc, lam) - alpha * entropy)

        loss, grads = actor_loss_fn(ts.actor)
        updates, opt_state = actor_opt.update(grads, ts.actor_opt_state)
        new_actor = optax.apply_updates(ts.actor, updates)
        return ts._replace(actor=new_actor, actor_opt_state=opt_state), loss

    def update_alpha(ts, batch):
        obs = batch[0]
        @eqx.filter_value_and_grad
        def alpha_loss_fn(log_alpha):
            dist = jax.vmap(ts.actor)(obs)
            probs = jax.nn.softmax(dist.logits, axis=-1)
            log_probs = jnp.log(probs + 1e-8)
            entropy = -jnp.sum(probs * log_probs, axis=-1).sum(axis=-1)
            return jnp.mean(jnp.exp(log_alpha) * (entropy - target_entropy))
        loss, grads = alpha_loss_fn(ts.log_alpha)
        updates, opt_state = alpha_opt.update(grads, ts.alpha_opt_state)
        new_log_alpha = optax.apply_updates(ts.log_alpha, updates)
        return ts._replace(log_alpha=new_log_alpha, alpha_opt_state=opt_state), loss

    def update_targets(ts):
        return ts._replace(
            q1_target=soft_update(ts.q1_target, ts.q1, config.tau),
            q2_target=soft_update(ts.q2_target, ts.q2, config.tau),
            q1_cost_target=soft_update(ts.q1_cost_target, ts.q1_cost, config.tau),
            q2_cost_target=soft_update(ts.q2_cost_target, ts.q2_cost, config.tau)
        )


    def eval_func(ts, rng):
        def step_env(carry):
            rng, obs, es, done, ep_r, ep_c = carry
            rng, ak, sk = jax.random.split(rng, 3)
            dist = ts.actor(obs)
            action = jnp.argmax(dist.logits, axis=-1) if config.evaluate_deterministically else dist.sample(seed=ak)
            (obs, r, term, trunc, info), es = env.step(sk, es, action)
            # cost直接从info中获取（环境已计算）
            cost = info.get("cost", 0.0)
            return (rng, obs, es, jnp.logical_or(term, trunc), ep_r + r, ep_c + cost)
        rng, rk = jax.random.split(rng)
        obs, es = env.reset(rk)
        _, _, _, _, ep_r, ep_c = jax.lax.while_loop(lambda c: ~c[3], step_env, (rng, obs, es, False, 0.0, 0.0))
        return ep_r, ep_c

    def train_func(rng=rng):
        def _env_step(runner_state, _):
            ts, es, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)
            dist = jax.vmap(ts.actor)(last_obs)
            action = dist.sample(seed=sample_key)
            rng, key = jax.random.split(rng)
            step_keys = jax.random.split(key, config.num_envs)
            (obsv, reward, term, trunc, info), es = jax.vmap(env.step)(step_keys, es, action)
            done = jnp.logical_or(term, trunc)
            # cost直接从info中获取（环境已计算），支持多通道 [num_envs, num_costs]
            cost = info.get("cost", jnp.zeros((config.num_envs, config.num_costs)))
            transition = Transition(observation=last_obs, action=action, reward=reward, cost=cost, done=done, info=info)
            return (ts, es, obsv, rng), transition

        def _gradient_step(carry, _):
            ts, buffer, rng = carry
            rng, key = jax.random.split(rng)
            batch = sample_buffer(buffer, key)
            ts, _ = update_critic(ts, batch, rng)
            ts, _ = update_actor(ts, batch)
            ts = jax.lax.cond(config.autotune_alpha, lambda t: update_alpha(t, batch)[0], lambda t: t, ts)
            ts = update_targets(ts)
            return (ts, buffer, rng), None

        def train_step(runner_state, _):
            ts, buffer, es, last_obs, rng = runner_state
            
            # 收集num_steps步数据
            (ts, es, last_obs, rng), traj = jax.lax.scan(
                _env_step, (ts, es, last_obs, rng), None, config.num_steps)
            
            # 计算next_obs
            next_obs = jnp.concatenate([traj.observation[1:], last_obs[None]], axis=0)
            
            # 添加到buffer
            buffer = add_batch_to_buffer(buffer, traj.observation, traj.action, 
                                         traj.reward, traj.cost, next_obs, traj.done)
            
            # 梯度更新
            def do_updates(args):
                ts, buf, rng = args
                (ts, buf, rng), _ = jax.lax.scan(_gradient_step, (ts, buf, rng), None, config.gradient_steps)
                return ts, buf, rng
            ts, buffer, rng = jax.lax.cond(
                buffer.size >= config.learning_starts, do_updates, lambda x: x, (ts, buffer, rng))
            
            # PID更新
            ep_cost_avg = traj.cost.mean(axis=(0, 1)) # [num_costs]
            new_pid_state = update_pid_lagrange(ts.pid_state, pid_util_config, ep_cost_avg)
            ts = ts._replace(pid_state=new_pid_state)
            
            # 评估
            rng, eval_key = jax.random.split(rng)
            eval_r, eval_c = eval_func(ts, eval_key)
            
            metric = traj.info
            metric["eval_rewards"] = eval_r
            metric["eval_cost"] = eval_c
            metric["lagrangian"] = new_pid_state.multipliers
            metric["alpha"] = jnp.exp(ts.log_alpha)
            metric["ep_cost_avg"] = ep_cost_avg

            def callback(info):
                global _pbar
                if _pbar is not None:
                    _pbar.update(1)
                    _pbar.set_postfix({
                        "eval_r": f"{info['eval_rewards']:.2f}",
                        "eval_c": f"{info['eval_cost']}",
                        "lam": f"{info['lagrangian']}"
                    })
                if config.debug:
                    print(f"eval_r={info['eval_rewards']:.2f}, eval_c={info['eval_cost']:.2f}, lam={info['lagrangian']:.4f}")
                if wandb.run:
                    finished = info.get("returned_episode", jnp.zeros(config.num_envs, dtype=bool))
                    if finished.any():
                        log_data = info.get("logging_data", {})
                        if log_data:
                            log_data = jax.tree.map(lambda x: x[finished].mean(), log_data)
                        wandb.log({
                            "eval_rewards": info["eval_rewards"],
                            "eval_cost": info["eval_cost"],
                            "lagrangian": info["lagrangian"],
                            "alpha": info["alpha"],
                            "ep_cost_avg": info["ep_cost_avg"],
                            **log_data
                        })
            jax.debug.callback(callback, metric)
            
            return (ts, buffer, es, last_obs, rng), metric

        rng, key = jax.random.split(rng)
        reset_keys = jax.random.split(key, config.num_envs)
        init_obs, init_es = jax.vmap(env.reset)(reset_keys)
        
        runner_state = (train_state, buffer, init_es, init_obs, rng)
        final_state, metrics = jax.lax.scan(train_step, runner_state, None, config.num_iterations)
        return final_state, metrics

    return train_func, config, eval_func
