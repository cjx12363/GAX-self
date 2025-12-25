"""
图表四：24小时调度堆叠图 (24-Hour Dispatch Stack Plot)

直观展示智能体在一天内的调度行为，验证其是否学会了"削峰填谷"和"躲避过载"。

功能：
- 横轴：时间 (00:00 - 24:00)
- 左纵轴：功率 (kW)
- 右纵轴：电价 (Price)
- 图层堆叠：基础负荷（灰色）+ EV充电负荷（蓝色/橙色）
- 变压器容量上限红色粗虚线
- 电价曲线覆盖在图上

使用方法：
    python plot_dispatch_schedule.py --data episode.json --output figures/dispatch.pdf
    python plot_dispatch_schedule.py --demo
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


def plot_dispatch_schedule(
    time_hours: np.ndarray,
    base_load: np.ndarray,
    ev_load_pid: np.ndarray,
    ev_load_baseline: np.ndarray = None,
    electricity_price: np.ndarray = None,
    transformer_capacity: float = 200.0,
    output_path: str = None,
    title: str = "24-Hour Dispatch Schedule"
):
    """
    绑制24小时调度堆叠图
    
    参数:
        time_hours: 时间数组（小时）
        base_load: 基础负荷 (kW)
        ev_load_pid: PID方法的EV充电负荷 (kW)
        ev_load_baseline: 基线方法的EV充电负荷（可选）
        electricity_price: 电价序列（可选）
        transformer_capacity: 变压器容量上限 (kW)
        output_path: 输出路径
        title: 图表标题
    """
    setup_style()
    
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    
    # 绘制基础负荷（灰色填充区域）
    ax1.fill_between(time_hours, 0, base_load, 
                    color=COLORS["base_load"], alpha=0.6, 
                    label="Base Load", step="mid")
    
    # 绘制PID方法的EV负荷（蓝色，堆叠在基础负荷之上）
    total_load_pid = base_load + ev_load_pid
    ax1.fill_between(time_hours, base_load, total_load_pid,
                    color=COLORS["ev_load_pid"], alpha=0.7,
                    label="EV Load (PID-Lag)", step="mid")
    
    # 绘制基线方法的EV负荷（如果提供）
    if ev_load_baseline is not None:
        total_load_baseline = base_load + ev_load_baseline
        ax1.plot(time_hours, total_load_baseline, 
                color=COLORS["ev_load_baseline"], linestyle="--",
                linewidth=1.5, label="Total Load (PPO-Lag)")
    
    # 变压器容量限制
    ax1.axhline(y=transformer_capacity, color=COLORS["capacity_limit"],
               linestyle="--", linewidth=2.5, 
               label=f"Transformer Limit ({transformer_capacity} kW)")
    
    # 标注关键时段
    # 傍晚高峰 (18:00-20:00)
    ax1.axvspan(18, 20, alpha=0.1, color='red', label='Peak Hours')
    # 深夜低谷 (02:00-05:00)
    ax1.axvspan(2, 5, alpha=0.1, color='green', label='Off-Peak Hours')
    
    ax1.set_xlabel("Time of Day (Hour)", fontsize=11)
    ax1.set_ylabel("Power (kW)", fontsize=11)
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=45)
    
    # 电价曲线（右轴）
    if electricity_price is not None:
        ax2 = ax1.twinx()
        ax2.plot(time_hours, electricity_price, color=COLORS["price"],
                linewidth=2, linestyle="-", alpha=0.8, label="Electricity Price")
        ax2.set_ylabel("Price (€/kWh)", fontsize=11, color=COLORS["price"])
        ax2.tick_params(axis='y', labelcolor=COLORS["price"])
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper left', frameon=False, fontsize=8)
    else:
        ax1.legend(loc='upper left', frameon=False, fontsize=9)
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    if title:
        ax1.set_title(title, fontsize=12)
    
    # 添加注释说明"贴线飞行"现象
    # 找到接近容量限制的点
    if np.any(total_load_pid > transformer_capacity * 0.9):
        ax1.annotate(
            '"Kissing the bound"\n(PID智能体)',
            xy=(10, transformer_capacity * 0.95), 
            fontsize=8, ha='center', color=COLORS["ev_load_pid"],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


from scripts.plotting.demo_data import generate_dispatch_schedule_demo_data as generate_demo_data


def main():
    parser = argparse.ArgumentParser(description="绘制24小时调度堆叠图")
    parser.add_argument("--data", type=str, default=None, help="JSON数据文件")
    parser.add_argument("--output", type=str, default="figures/dispatch_schedule",
                       help="输出文件路径")
    parser.add_argument("--capacity", type=float, default=200.0, 
                       help="变压器容量 (kW)")
    parser.add_argument("--demo", action="store_true", help="使用模拟数据")
    
    args = parser.parse_args()
    
    if args.demo or args.data is None:
        print("使用模拟数据生成演示图表...")
        time_hours, base_load, ev_load_pid, ev_load_baseline, price = generate_demo_data()
    else:
        import json
        with open(args.data, 'r') as f:
            data = json.load(f)
        time_hours = np.linspace(0, 24, len(data['base_load']))
        base_load = np.array(data['base_load'])
        ev_load_pid = np.array(data['ev_load'])
        ev_load_baseline = np.array(data.get('ev_load_baseline', []))
        price = np.array(data.get('electricity_price', []))
        if len(ev_load_baseline) == 0:
            ev_load_baseline = None
        if len(price) == 0:
            price = None
    
    fig = plot_dispatch_schedule(
        time_hours=time_hours,
        base_load=base_load,
        ev_load_pid=ev_load_pid,
        ev_load_baseline=ev_load_baseline if ev_load_baseline is not None and len(ev_load_baseline) > 0 else None,
        electricity_price=price,
        transformer_capacity=args.capacity,
        output_path=args.output
    )
    
    plt.show()


if __name__ == "__main__":
    main()
