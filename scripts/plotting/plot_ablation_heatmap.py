"""
图表六：消融实验热力图 (Ablation Study Heatmap)

回答 "PID参数如何选择" 以及 "D项是否真的必要" 的问题。

功能：
- 横轴：比例系数 K_p
- 纵轴：积分系数 K_i 或微分系数 K_d
- 颜色深浅：最终性能指标（如 Reward/Cost 比率）
- 可选：用多条曲线展示不同 K_d 值下的 Cost 收敛

使用方法：
    python plot_ablation_heatmap.py --data ablation.csv --output figures/ablation.pdf
    python plot_ablation_heatmap.py --demo
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.plotting.style_config import (
    setup_style, COLORS, save_figure
)


def plot_ablation_heatmap(
    kp_values: np.ndarray,
    ki_values: np.ndarray,
    performance_matrix: np.ndarray,
    output_path: str = None,
    title: str = "PID Parameter Ablation Study",
    metric_name: str = "Reward / Cost Ratio",
    highlight_best: bool = True
):
    """
    绘制消融实验热力图
    
    参数:
        kp_values: K_p 参数值数组
        ki_values: K_i 参数值数组
        performance_matrix: 性能矩阵 [len(ki), len(kp)]
        output_path: 输出路径
        title: 图表标题
        metric_name: 指标名称
        highlight_best: 是否高亮最佳点
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    
    # 绘制热力图
    im = ax.imshow(performance_matrix, cmap='viridis', aspect='auto',
                  origin='lower', interpolation='nearest')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(kp_values)))
    ax.set_yticks(np.arange(len(ki_values)))
    ax.set_xticklabels([f"{v:.2f}" for v in kp_values])
    ax.set_yticklabels([f"{v:.3f}" for v in ki_values])
    
    ax.set_xlabel("Proportional Gain K_p", fontsize=11)
    ax.set_ylabel("Integral Gain K_i", fontsize=11)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, fontsize=10)
    
    # 在每个格子中显示数值
    for i in range(len(ki_values)):
        for j in range(len(kp_values)):
            value = performance_matrix[i, j]
            text_color = 'white' if value < (performance_matrix.max() + performance_matrix.min()) / 2 else 'black'
            ax.text(j, i, f"{value:.2f}", ha='center', va='center', 
                   color=text_color, fontsize=8)
    
    # 高亮最佳点
    if highlight_best:
        best_idx = np.unravel_index(np.argmax(performance_matrix), performance_matrix.shape)
        ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=3))
        ax.annotate('Best', 
                   xy=(best_idx[1], best_idx[0]),
                   xytext=(best_idx[1] + 1.5, best_idx[0] + 0.5),
                   fontsize=10, color='red', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    if title:
        ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_kd_ablation_curves(
    steps: np.ndarray,
    cost_curves: dict,
    kd_values: list,
    cost_limit: float = 25.0,
    output_path: str = None,
    title: str = "Effect of Derivative Gain K_d on Cost Convergence"
):
    """
    绘制不同 K_d 值下的 Cost 收敛曲线
    
    参数:
        steps: 训练步数
        cost_curves: K_d 值到 cost 曲线的映射
        kd_values: K_d 值列表
        cost_limit: 安全阈值
        output_path: 输出路径
        title: 标题
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(kd_values)))
    
    for kd, color in zip(kd_values, colors):
        cost = cost_curves[kd]
        label = f"K_d = {kd}" if kd > 0 else "K_d = 0 (PI only)"
        linestyle = "-" if kd > 0 else "--"
        ax.plot(steps, cost, color=color, linestyle=linestyle, 
               linewidth=1.5, label=label)
    
    # 安全阈值
    ax.axhline(y=cost_limit, color=COLORS["constraint"], linestyle=":",
              linewidth=2, label=f"Cost Limit (d={cost_limit})")
    
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Cumulative Cost", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if title:
        ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path + "_kd_curves")
    
    return fig


from scripts.plotting.demo_data import generate_ablation_heatmap_demo_data as generate_demo_data


def main():
    parser = argparse.ArgumentParser(description="绘制消融实验热力图")
    parser.add_argument("--data", type=str, default=None, help="CSV数据文件")
    parser.add_argument("--output", type=str, default="figures/ablation_heatmap",
                       help="输出文件路径")
    parser.add_argument("--demo", action="store_true", help="使用模拟数据")
    parser.add_argument("--plot-kd", action="store_true", 
                       help="同时绘制K_d消融曲线")
    
    args = parser.parse_args()
    
    if args.demo or args.data is None:
        print("使用模拟数据生成演示图表...")
        kp_values, ki_values, performance, steps, cost_curves, kd_values = generate_demo_data()
    else:
        # 从CSV加载
        import csv
        kp_set = set()
        ki_set = set()
        data_dict = {}
        
        with open(args.data, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kp = float(row['kp'])
                ki = float(row['ki'])
                perf = float(row['performance'])
                kp_set.add(kp)
                ki_set.add(ki)
                data_dict[(kp, ki)] = perf
        
        kp_values = np.array(sorted(kp_set))
        ki_values = np.array(sorted(ki_set))
        performance = np.zeros((len(ki_values), len(kp_values)))
        for i, ki in enumerate(ki_values):
            for j, kp in enumerate(kp_values):
                performance[i, j] = data_dict.get((kp, ki), 0)
    
    # 绘制热力图
    fig1 = plot_ablation_heatmap(
        kp_values=kp_values,
        ki_values=ki_values,
        performance_matrix=performance,
        output_path=args.output
    )
    
    # 绘制K_d消融曲线
    if args.plot_kd or args.demo:
        if args.demo:
            fig2 = plot_kd_ablation_curves(
                steps=steps,
                cost_curves=cost_curves,
                kd_values=kd_values,
                output_path=args.output
            )
    
    plt.show()


if __name__ == "__main__":
    main()
