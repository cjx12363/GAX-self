"""
图表五：用户满意度小提琴图 (Violin Plots for User Satisfaction)

展示不同算法在用户SOC偏差上的分布特性，
小提琴图比柱状图更能展示数据的分布特性（长尾效应）。

功能：
- 横轴：不同算法 (PID, PPO-Lag, MPC, Random)
- 纵轴：SOC偏差 (SOC_target - SOC_final)，越接近0越好
- 小提琴宽度代表样本分布密度

使用方法：
    python plot_user_satisfaction.py --data satisfaction.csv --output figures/violin.pdf
    python plot_user_satisfaction.py --demo
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.plotting.style_config import (
    setup_style, COLORS, save_figure, get_algorithm_label
)


def plot_user_satisfaction(
    data: dict,
    output_path: str = None,
    title: str = "User Satisfaction: SOC Deviation Distribution"
):
    """
    绘制用户满意度小提琴图
    
    参数:
        data: 各算法的SOC偏差数据
            {
                "ppo_pid": [soc_dev1, soc_dev2, ...],
                "ppo_lag": [...],
                ...
            }
        output_path: 输出路径
        title: 图表标题
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    
    # 准备数据
    algorithms = list(data.keys())
    all_data = [np.array(data[algo]) for algo in algorithms]
    positions = np.arange(len(algorithms))
    
    # 获取颜色
    colors = [COLORS.get(algo, "#666666") for algo in algorithms]
    
    # 绘制小提琴图
    parts = ax.violinplot(all_data, positions=positions, showmeans=True, 
                         showmedians=True, widths=0.7)
    
    # 自定义颜色
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # 设置其他元素颜色
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1)
    
    # 添加散点显示原始数据分布
    for i, (algo, vals) in enumerate(zip(algorithms, all_data)):
        # 添加抖动
        jitter = np.random.normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, 
                  alpha=0.3, s=10, color=colors[i])
    
    # 设置x轴标签
    ax.set_xticks(positions)
    ax.set_xticklabels([get_algorithm_label(algo) for algo in algorithms], 
                       fontsize=10, rotation=15)
    
    # 添加零线（理想情况）
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Ideal (No Deviation)')
    
    ax.set_xlabel("Algorithm", fontsize=11)
    ax.set_ylabel("SOC Deviation (SOC_target - SOC_final)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    if title:
        ax.set_title(title, fontsize=12)
    
    # 添加注释说明分布特征
    # 找到有长尾效应的算法
    for i, (algo, vals) in enumerate(zip(algorithms, all_data)):
        if np.percentile(vals, 95) > np.percentile(vals, 75) * 1.5:
            ax.annotate(
                'Long tail',
                xy=(i, np.percentile(vals, 95)),
                xytext=(i + 0.3, np.percentile(vals, 95) + 0.05),
                fontsize=8, color='red', alpha=0.7,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5)
            )
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


from scripts.plotting.demo_data import generate_user_satisfaction_demo_data as generate_demo_data


def load_data_from_csv(csv_path: str) -> dict:
    """从CSV加载数据"""
    import csv
    
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row['algorithm']
            if algo not in data:
                data[algo] = []
            data[algo].append(float(row['soc_deviation']))
    
    return data


def main():
    parser = argparse.ArgumentParser(description="绘制用户满意度小提琴图")
    parser.add_argument("--data", type=str, default=None, help="CSV数据文件")
    parser.add_argument("--output", type=str, default="figures/user_satisfaction",
                       help="输出文件路径")
    parser.add_argument("--demo", action="store_true", help="使用模拟数据")
    
    args = parser.parse_args()
    
    if args.demo or args.data is None:
        print("使用模拟数据生成演示图表...")
        data = generate_demo_data()
    else:
        print(f"从 {args.data} 加载数据...")
        data = load_data_from_csv(args.data)
    
    fig = plot_user_satisfaction(
        data=data,
        output_path=args.output
    )
    
    plt.show()


if __name__ == "__main__":
    main()
