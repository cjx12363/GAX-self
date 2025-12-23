"""
图表三：帕累托前沿分析 (Pareto Frontier Analysis)

展示方法在安全性与收益之间的权衡能力。

功能：
- 横轴：平均安全代价 (Average Cost)
- 纵轴：平均收益 (Average Return)
- 不同算法用不同标记
- 理想区域在左上角（高收益、低代价）

使用方法：
    python plot_pareto_frontier.py --data results.csv --output figures/pareto.pdf
    python plot_pareto_frontier.py --demo
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.plotting.style_config import (
    setup_style, COLORS, MARKERS, save_figure, get_algorithm_label
)


def plot_pareto_frontier(
    data: dict,
    cost_limit: float = 25.0,
    output_path: str = None,
    title: str = "Pareto Frontier Analysis"
):
    """
    绑制帕累托前沿图
    
    参数:
        data: 各算法的性能数据，格式:
            {
                "ppo_pid": {
                    "costs": [c1, c2, ...],  # 不同超参数/种子的代价
                    "returns": [r1, r2, ...]  # 对应的收益
                },
                ...
            }
        cost_limit: 安全阈值（绘制垂直线）
        output_path: 输出路径
        title: 图表标题
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    for algo_key, algo_data in data.items():
        costs = np.array(algo_data["costs"])
        returns = np.array(algo_data["returns"])
        
        color = COLORS.get(algo_key, "#000000")
        marker = MARKERS.get(algo_key, "o")
        label = get_algorithm_label(algo_key)
        
        # 散点图
        ax.scatter(costs, returns, c=color, marker=marker, s=80, 
                  label=label, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # 如果有多个点，绘制均值点（更大的标记）
        if len(costs) > 1:
            ax.scatter([np.mean(costs)], [np.mean(returns)], c=color, 
                      marker=marker, s=200, alpha=1.0, edgecolors='black', 
                      linewidth=1.5)
    
    # 绘制安全阈值线
    ax.axvline(x=cost_limit, color=COLORS["constraint"], linestyle="--",
              linewidth=1.5, label=f"Cost Limit (d={cost_limit})", alpha=0.8)
    
    # 标注理想区域
    ax.annotate(
        "Ideal Region\n(High Return, Low Cost)",
        xy=(0.15, 0.85), xycoords='axes fraction',
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
    )
    
    # 添加箭头指示
    ax.annotate('', xy=(0.05, 0.95), xycoords='axes fraction',
               xytext=(0.25, 0.75), textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    
    ax.set_xlabel("Average Cost", fontsize=11)
    ax.set_ylabel("Average Return", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if title:
        ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def generate_demo_data(num_points: int = 10) -> dict:
    """生成演示数据"""
    np.random.seed(42)
    
    # PID-Lagrangian: 左上角区域（低代价、高收益）
    pid_costs = np.random.normal(20, 3, num_points)
    pid_returns = np.random.normal(120, 8, num_points)
    
    # PPO-Lagrangian: 分布更广，有些点超出阈值
    lag_costs = np.random.normal(28, 8, num_points)
    lag_returns = np.random.normal(110, 15, num_points)
    
    # PPO (无约束): 高收益但高代价
    ppo_costs = np.random.normal(50, 10, num_points)
    ppo_returns = np.random.normal(140, 10, num_points)
    
    # CPO: 保守策略，低代价但也低收益
    cpo_costs = np.random.normal(15, 3, num_points)
    cpo_returns = np.random.normal(90, 12, num_points)
    
    return {
        "ppo_pid": {"costs": pid_costs, "returns": pid_returns},
        "ppo_lag": {"costs": lag_costs, "returns": lag_returns},
        "ppo": {"costs": ppo_costs, "returns": ppo_returns},
        "cpo": {"costs": cpo_costs, "returns": cpo_returns}
    }


def load_data_from_csv(csv_path: str) -> dict:
    """从CSV文件加载数据"""
    import csv
    
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row['algorithm']
            if algo not in data:
                data[algo] = {"costs": [], "returns": []}
            data[algo]["costs"].append(float(row['cost']))
            data[algo]["returns"].append(float(row['return']))
    
    return data


def main():
    parser = argparse.ArgumentParser(description="绘制帕累托前沿图")
    parser.add_argument("--data", type=str, default=None, help="CSV数据文件")
    parser.add_argument("--output", type=str, default="figures/pareto_frontier",
                       help="输出文件路径")
    parser.add_argument("--cost-limit", type=float, default=25.0, help="安全阈值")
    parser.add_argument("--demo", action="store_true", help="使用模拟数据")
    
    args = parser.parse_args()
    
    if args.demo or args.data is None:
        print("使用模拟数据生成演示图表...")
        data = generate_demo_data()
    else:
        print(f"从 {args.data} 加载数据...")
        data = load_data_from_csv(args.data)
    
    fig = plot_pareto_frontier(
        data=data,
        cost_limit=args.cost_limit,
        output_path=args.output
    )
    
    plt.show()


if __name__ == "__main__":
    main()
