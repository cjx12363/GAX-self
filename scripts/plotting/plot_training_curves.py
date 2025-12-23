"""
图表一：双轴训练曲线 (Dual-Axis Training Curves)

同时展示算法的学习效率 (Reward) 和安全性 (Cost)。

功能：
- 左纵轴：累积奖励 (Reward)
- 右纵轴：累积代价 (Cost)
- 包含均值线和置信区间 (多种子运行)
- 红色虚线表示安全阈值 (cost_limit)
- EMA 平滑处理

使用方法：
    python plot_training_curves.py --data logs/ --output figures/training_curves.pdf
    python plot_training_curves.py --demo  # 使用模拟数据演示
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.plotting.style_config import (
    setup_style, COLORS, save_figure, create_figure,
    smooth_curve, plot_with_confidence, add_constraint_line,
    get_algorithm_label
)


def plot_training_curves(
    data: dict,
    cost_limit: float = 25.0,
    smooth_alpha: float = 0.9,
    output_path: str = None,
    title: str = None
):
    """
    绑制双轴训练曲线
    
    参数:
        data: 包含各算法数据的字典，格式:
            {
                "ppo_pid": {
                    "steps": np.array,
                    "reward_mean": np.array,
                    "reward_std": np.array,
                    "cost_mean": np.array,
                    "cost_std": np.array
                },
                ...
            }
        cost_limit: 安全阈值
        smooth_alpha: EMA平滑系数
        output_path: 输出路径
        title: 图表标题
    """
    setup_style()
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=150)
    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
    
    # 绑制各算法曲线
    for algo_key, algo_data in data.items():
        steps = algo_data["steps"]
        color = COLORS.get(algo_key, "#000000")
        label = get_algorithm_label(algo_key)
        
        # 左轴：Reward
        reward_mean = smooth_curve(algo_data["reward_mean"], smooth_alpha)
        reward_std = smooth_curve(algo_data["reward_std"], smooth_alpha)
        
        ax1.plot(steps, reward_mean, color=color, linestyle="-", 
                linewidth=1.5, label=f"{label} (Reward)")
        ax1.fill_between(steps, reward_mean - reward_std, reward_mean + reward_std,
                        color=color, alpha=0.2)
        
        # 右轴：Cost
        cost_mean = smooth_curve(algo_data["cost_mean"], smooth_alpha)
        cost_std = smooth_curve(algo_data["cost_std"], smooth_alpha)
        
        ax2.plot(steps, cost_mean, color=color, linestyle="--", 
                linewidth=1.5, label=f"{label} (Cost)")
        ax2.fill_between(steps, cost_mean - cost_std, cost_mean + cost_std,
                        color=color, alpha=0.1)
    
    # 添加安全阈值线
    ax2.axhline(y=cost_limit, color=COLORS["constraint"], linestyle=":", 
               linewidth=2, label=f"Cost Limit (d={cost_limit})")
    
    # 设置轴标签
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Cumulative Reward", fontsize=11, color=COLORS["reward"])
    ax2.set_ylabel("Cumulative Cost", fontsize=11, color=COLORS["cost"])
    
    # 设置轴颜色
    ax1.tick_params(axis='y', labelcolor=COLORS["reward"])
    ax2.tick_params(axis='y', labelcolor=COLORS["cost"])
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
              frameon=False, fontsize=9)
    
    # 网格
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 标题
    if title:
        ax1.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    # 保存
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def generate_demo_data(num_steps: int = 500, num_seeds: int = 5) -> dict:
    """生成演示数据"""
    steps = np.linspace(0, 1000000, num_steps)
    
    def generate_curves(algo_type: str):
        """生成不同算法的特征曲线"""
        np.random.seed(42)
        
        if algo_type == "ppo_pid":
            # PID-Lagrangian: 快速收敛，低代价
            reward = 80 + 40 * (1 - np.exp(-steps / 200000)) + np.random.randn(num_steps) * 5
            cost = 30 * np.exp(-steps / 150000) + 15 + np.random.randn(num_steps) * 3
        elif algo_type == "ppo_lag":
            # PPO-Lagrangian: 有震荡
            t = steps / 100000
            reward = 70 + 35 * (1 - np.exp(-t)) + 10 * np.sin(t * 2) * np.exp(-t/5) + np.random.randn(num_steps) * 8
            cost = 25 + 15 * np.sin(t * 3) * np.exp(-t/3) + np.random.randn(num_steps) * 5
        elif algo_type == "ppo":
            # PPO (无约束): 高奖励但高代价
            reward = 90 + 50 * (1 - np.exp(-steps / 150000)) + np.random.randn(num_steps) * 6
            cost = 60 + 20 * np.sin(steps / 100000) + np.random.randn(num_steps) * 10
        else:
            reward = 50 + np.random.randn(num_steps) * 10
            cost = 40 + np.random.randn(num_steps) * 10
        
        # 模拟多种子
        reward_all = np.array([reward + np.random.randn(num_steps) * 5 for _ in range(num_seeds)])
        cost_all = np.array([cost + np.random.randn(num_steps) * 3 for _ in range(num_seeds)])
        
        return {
            "steps": steps,
            "reward_mean": np.mean(reward_all, axis=0),
            "reward_std": np.std(reward_all, axis=0),
            "cost_mean": np.mean(cost_all, axis=0),
            "cost_std": np.std(cost_all, axis=0)
        }
    
    return {
        "ppo_pid": generate_curves("ppo_pid"),
        "ppo_lag": generate_curves("ppo_lag"),
        "ppo": generate_curves("ppo")
    }


def load_data_from_logs(log_dir: str) -> dict:
    """从日志文件加载数据"""
    from chargax.util.training_logger import TrainingLogger
    
    log_path = Path(log_dir)
    data = {}
    
    # 按算法分组日志文件
    for algo in ["ppo_pid", "ppo_lag", "ppo"]:
        algo_files = list(log_path.glob(f"{algo}_*.json"))
        if algo_files:
            agg_data = TrainingLogger.aggregate_seeds([str(f) for f in algo_files])
            data[algo] = {
                "steps": np.array(agg_data["steps"]),
                "reward_mean": np.array(agg_data["reward_mean"]),
                "reward_std": np.array(agg_data["reward_std"]),
                "cost_mean": np.array(agg_data["cost_mean"]),
                "cost_std": np.array(agg_data["cost_std"])
            }
    
    return data


def main():
    parser = argparse.ArgumentParser(description="绘制双轴训练曲线")
    parser.add_argument("--data", type=str, default=None, help="日志目录路径")
    parser.add_argument("--output", type=str, default="figures/training_curves", 
                       help="输出文件路径（不含扩展名）")
    parser.add_argument("--cost-limit", type=float, default=25.0, help="安全阈值")
    parser.add_argument("--smooth", type=float, default=0.9, help="EMA平滑系数")
    parser.add_argument("--demo", action="store_true", help="使用模拟数据演示")
    parser.add_argument("--title", type=str, default=None, help="图表标题")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.demo or args.data is None:
        print("使用模拟数据生成演示图表...")
        data = generate_demo_data()
    else:
        print(f"从 {args.data} 加载数据...")
        data = load_data_from_logs(args.data)
    
    # 绑图
    fig = plot_training_curves(
        data=data,
        cost_limit=args.cost_limit,
        smooth_alpha=args.smooth,
        output_path=args.output,
        title=args.title
    )
    
    plt.show()


if __name__ == "__main__":
    main()
