"""
图表二：拉格朗日乘子演变图 (Multiplier Dynamics)

对比 Standard Lagrangian 和 PID-Lagrangian 的乘子 λ 变化，
展示 PID 控制器消除震荡的效果。

功能：
- 纵轴：λ 值（对数坐标或线性坐标）
- 对比展示两种方法的 λ 曲线
- 标准方法应呈现震荡，PID方法应平滑收敛

使用方法：
    python plot_multiplier_dynamics.py --data logs/ --output figures/multiplier.pdf
    python plot_multiplier_dynamics.py --demo
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.plotting.style_config import (
    setup_style, COLORS, save_figure, smooth_curve,
    get_algorithm_label
)


def plot_multiplier_dynamics(
    data: dict,
    log_scale: bool = False,
    smooth_alpha: float = 0.0,  # 原始数据更能展示震荡
    output_path: str = None,
    title: str = "Lagrangian Multiplier Dynamics"
):
    """
    绑制拉格朗日乘子演变对比图
    
    参数:
        data: 包含各算法数据的字典
        log_scale: 是否使用对数坐标
        smooth_alpha: 平滑系数（0表示不平滑）
        output_path: 输出路径
        title: 图表标题
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    
    for algo_key, algo_data in data.items():
        steps = algo_data["steps"]
        multiplier = algo_data["multiplier_mean"]
        
        if smooth_alpha > 0:
            multiplier = smooth_curve(multiplier, smooth_alpha)
        
        color = COLORS.get(algo_key, "#000000")
        label = get_algorithm_label(algo_key)
        linestyle = "--" if "lag" in algo_key and "pid" not in algo_key else "-"
        
        ax.plot(steps, multiplier, color=color, linestyle=linestyle,
               linewidth=1.5, label=label)
        
        # 添加置信区间（如果有标准差数据）
        if "multiplier_std" in algo_data:
            std = algo_data["multiplier_std"]
            if smooth_alpha > 0:
                std = smooth_curve(std, smooth_alpha)
            ax.fill_between(steps, multiplier - std, multiplier + std,
                          color=color, alpha=0.15)
    
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Lagrangian Multiplier λ", fontsize=11)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if title:
        ax.set_title(title, fontsize=12)
    
    # 添加说明文字
    ax.annotate(
        "Oscillation\n(Standard)",
        xy=(0.3, 0.7), xycoords='axes fraction',
        fontsize=9, color=COLORS.get("ppo_lag", "#DE8F05"),
        ha='center', alpha=0.7
    )
    ax.annotate(
        "Smooth convergence\n(PID)",
        xy=(0.7, 0.4), xycoords='axes fraction',
        fontsize=9, color=COLORS.get("ppo_pid", "#0173B2"),
        ha='center', alpha=0.7
    )
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


from scripts.plotting.demo_data import generate_multiplier_dynamics_demo_data as generate_demo_data


def main():
    parser = argparse.ArgumentParser(description="绘制拉格朗日乘子演变图")
    parser.add_argument("--data", type=str, default=None, help="日志目录路径")
    parser.add_argument("--output", type=str, default="figures/multiplier_dynamics",
                       help="输出文件路径")
    parser.add_argument("--log-scale", action="store_true", help="使用对数坐标")
    parser.add_argument("--smooth", type=float, default=0.0, help="平滑系数")
    parser.add_argument("--demo", action="store_true", help="使用模拟数据")
    
    args = parser.parse_args()
    
    if args.demo or args.data is None:
        print("使用模拟数据生成演示图表...")
        data = generate_demo_data()
    else:
        # 从日志加载数据
        from chargax.util.training_logger import TrainingLogger
        log_path = Path(args.data)
        data = {}
        for algo in ["ppo_pid", "ppo_lag"]:
            files = list(log_path.glob(f"{algo}_*.json"))
            if files:
                agg = TrainingLogger.aggregate_seeds([str(f) for f in files])
                data[algo] = {
                    "steps": np.array(agg["steps"]),
                    "multiplier_mean": np.array(agg["multiplier_mean"]),
                    "multiplier_std": np.array(agg["multiplier_std"])
                }
    
    fig = plot_multiplier_dynamics(
        data=data,
        log_scale=args.log_scale,
        smooth_alpha=args.smooth,
        output_path=args.output
    )
    
    plt.show()


if __name__ == "__main__":
    main()
