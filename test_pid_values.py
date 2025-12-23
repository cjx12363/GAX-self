"""测试 PID 数值设计的合理性"""
import jax
import jax.numpy as jnp
import numpy as np

from env_config import create_env
from chargax.util.pid_lagrange import (
    PIDLagrangeConfig, 
    init_pid_lagrange, 
    update_pid_lagrange
)

def test_env_cost_distribution():
    """测试环境产生的 cost 分布"""
    env = create_env()
    
    print("=" * 60)
    print("环境和 PID 参数验证")
    print("=" * 60)
    
    # 环境信息
    print(f"\n📊 环境配置:")
    print(f"  - Episode 长度: {env.episode_length} 步")
    print(f"  - 变压器容量: {env.transformer_capacity_kw} kW")
    print(f"  - 分钟/步: {env.minutes_per_timestep}")
    
    # 运行几个 episode 收集 cost 数据
    print(f"\n🔄 运行随机 episode 收集 cost 数据...")
    
    key = jax.random.PRNGKey(42)
    num_episodes = 10
    episode_costs = []
    step_costs = []
    
    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        
        episode_cost = 0.0
        for step in range(env.episode_length):
            key, action_key, step_key = jax.random.split(key, 3)
            # 随机动作
            action = jax.random.randint(action_key, (env.action_space.nvec.shape[0],), 0, env.num_discretization_levels * 2)
            
            timestep, state = env.step(step_key, state, action)
            cost = timestep.info.get("cost", 0.0)
            episode_cost += float(cost)
            step_costs.append(float(cost))
        
        episode_costs.append(episode_cost)
        print(f"  Episode {ep+1}: 累积 cost = {episode_cost:.2f}")
    
    # 统计分析
    print(f"\n📈 Cost 统计分析:")
    print(f"  - 单步 cost 范围: [{min(step_costs):.4f}, {max(step_costs):.4f}]")
    print(f"  - 单步 cost 均值: {np.mean(step_costs):.4f}")
    print(f"  - 单步 cost 标准差: {np.std(step_costs):.4f}")
    print(f"  - Episode 累积 cost 范围: [{min(episode_costs):.2f}, {max(episode_costs):.2f}]")
    print(f"  - Episode 累积 cost 均值: {np.mean(episode_costs):.2f}")
    
    # PID 参数分析
    print(f"\n⚙️ PID 参数设计分析:")
    cost_limit = 10
    pid_kp = 0.1
    pid_ki = 0.001
    
    avg_episode_cost = np.mean(episode_costs)
    error = avg_episode_cost - cost_limit
    
    print(f"  - cost_limit: {cost_limit}")
    print(f"  - 平均 episode cost: {avg_episode_cost:.2f}")
    print(f"  - 误差 (cost - limit): {error:.2f}")
    print(f"  - P 项贡献: Kp × error = {pid_kp} × {error:.2f} = {pid_kp * error:.4f}")
    print(f"  - I 项每次增量: Ki × error = {pid_ki} × {error:.2f} = {pid_ki * error:.4f}")
    
    # 乘子增长模拟
    print(f"\n📐 乘子增长模拟 (假设误差恒定):")
    config = PIDLagrangeConfig(
        cost_limit=jnp.array([cost_limit]),
        pid_kp=jnp.array([pid_kp]),
        pid_ki=jnp.array([pid_ki]),
        pid_kd=jnp.array([0.0]),
    )
    state = init_pid_lagrange(config, 1)
    
    print(f"  初始乘子: {float(state.multipliers[0]):.4f}")
    for i in [1, 10, 50, 100, 200, 500]:
        for _ in range(i - (1 if i == 1 else [1, 10, 50, 100, 200, 500][[1, 10, 50, 100, 200, 500].index(i) - 1])):
            state = update_pid_lagrange(state, config, jnp.array([avg_episode_cost]))
        print(f"  第 {i:3d} 次迭代后乘子: {float(state.multipliers[0]):.4f}")

    # 建议
    print(f"\n💡 建议:")
    if avg_episode_cost > cost_limit:
        ratio = avg_episode_cost / cost_limit
        print(f"  ⚠️ 当前随机策略的 cost ({avg_episode_cost:.2f}) 是 limit ({cost_limit}) 的 {ratio:.1f}x")
        print(f"     这意味着算法需要显著减少过载行为")
        
    if pid_ki < 0.01:
        print(f"  ⚠️ Ki = {pid_ki} 可能过小，建议尝试 0.01 或 0.1")
        
    if error > 50:
        print(f"  ⚠️ 误差较大 ({error:.2f})，可能需要调整 cost_limit 或环境参数")

if __name__ == "__main__":
    test_env_cost_distribution()
