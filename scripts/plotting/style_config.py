"""
科研绘图样式配置

符合 IEEE/Nature 期刊论文规范的绘图样式设置。
使用 SciencePlots 库 + 自定义配色方案。

安装依赖：
    pip install SciencePlots

使用方法：
    from style_config import setup_style, COLORS, save_figure
    setup_style()
    # ... 绑图代码 ...
    save_figure(fig, "output.pdf")
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Optional, Tuple, List
import warnings

# ==================== 配色方案 ====================
# 对色盲友好的配色 (参考 Seaborn colorblind palette)
COLORS = {
    # 主要算法颜色
    "ppo_pid": "#0173B2",       # 深蓝色 - PID-Lagrangian
    "ppo_lag": "#DE8F05",       # 橙色 - PPO-Lagrangian
    "ppo": "#868686",           # 灰色 - 无约束PPO
    "cpo": "#029E73",           # 绿色 - CPO
    "mpc": "#CC78BC",           # 紫色 - MPC基线
    "random": "#A0A0A0",        # 浅灰色 - 随机策略
    
    # 辅助颜色
    "constraint": "#D55E00",    # 红橙色 - 约束阈值线
    "baseline": "#56B4E9",      # 天蓝色 - 基线
    "reward": "#0173B2",        # 蓝色 - 奖励曲线
    "cost": "#D55E00",          # 红橙色 - 代价曲线
    
    # 调度图颜色
    "base_load": "#868686",     # 灰色 - 基础负荷
    "ev_load_pid": "#0173B2",   # 蓝色 - PID方法EV负荷
    "ev_load_baseline": "#DE8F05",  # 橙色 - 基线EV负荷
    "price": "#CC78BC",         # 紫色 - 电价曲线
    "capacity_limit": "#D55E00" # 红色 - 容量限制
}

# 标记样式
MARKERS = {
    "ppo_pid": "o",
    "ppo_lag": "s",
    "ppo": "^",
    "cpo": "D",
    "mpc": "v",
    "random": "x"
}

# 线型
LINESTYLES = {
    "ppo_pid": "-",
    "ppo_lag": "--",
    "ppo": ":",
    "cpo": "-.",
    "mpc": "--"
}


# ==================== 样式设置 ====================
def setup_style(use_scienceplots: bool = True, context: str = "paper"):
    """
    设置科研绘图样式
    
    参数:
        use_scienceplots: 是否使用 SciencePlots 库
        context: 上下文 ("paper", "notebook", "talk", "poster")
    """
    # 尝试使用 SciencePlots
    if use_scienceplots:
        try:
            import scienceplots
            plt.style.use(['science', 'ieee', 'no-latex'])
            print("已启用 SciencePlots 科研样式")
        except ImportError:
            warnings.warn(
                "SciencePlots 未安装，使用默认样式。"
                "请运行: pip install SciencePlots"
            )
            _apply_default_style()
    else:
        _apply_default_style()
    
    # 覆盖一些设置以确保一致性
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        
        # 线条设置
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        
        # 网格设置
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        
        # 图例设置
        'legend.frameon': False,
        'legend.loc': 'best',
        
        # 保存设置
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # 去除顶部和右侧边框
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _apply_default_style():
    """应用默认科研样式（不依赖 SciencePlots）"""
    plt.style.use('seaborn-v0_8-whitegrid')
    

# ==================== 绘图工具函数 ====================
def save_figure(
    fig: plt.Figure, 
    path: str, 
    formats: List[str] = ["pdf", "png"],
    dpi: int = 300
):
    """
    保存图表为多种格式
    
    参数:
        fig: matplotlib Figure 对象
        path: 保存路径（不含扩展名）
        formats: 保存格式列表
        dpi: PNG 分辨率
    """
    import os
    
    # 确保目录存在
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # 去除可能的扩展名
    base_path = os.path.splitext(path)[0]
    
    for fmt in formats:
        save_path = f"{base_path}.{fmt}"
        fig.savefig(
            save_path, 
            format=fmt, 
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.05
        )
        print(f"图表已保存: {save_path}")


def create_figure(
    nrows: int = 1, 
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150
) -> Tuple[plt.Figure, np.ndarray]:
    """
    创建标准尺寸的图表
    
    参数:
        nrows: 行数
        ncols: 列数
        figsize: 图表尺寸 (宽, 高)，英寸
        dpi: 分辨率
        
    返回:
        (fig, axes) 元组
    """
    # IEEE 单栏宽度约 3.5 英寸，双栏约 7 英寸
    if figsize is None:
        width = 3.5 * ncols if ncols <= 2 else 7.0
        height = 2.5 * nrows
        figsize = (width, height)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    
    return fig, np.atleast_1d(axes)


def smooth_curve(
    data: np.ndarray, 
    alpha: float = 0.9,
    mode: str = "ema"
) -> np.ndarray:
    """
    对曲线进行平滑处理
    
    参数:
        data: 原始数据
        alpha: EMA 平滑系数 (0.85-0.95 推荐)
        mode: 平滑方式 ("ema" 或 "rolling")
        
    返回:
        平滑后的数据
    """
    if mode == "ema":
        # 指数移动平均
        smoothed = np.zeros_like(data, dtype=float)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]
        return smoothed
    elif mode == "rolling":
        # 滑动窗口平均
        window = max(1, int(len(data) * (1 - alpha) / 10))
        return np.convolve(data, np.ones(window)/window, mode='same')
    else:
        return data


def plot_with_confidence(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    color: str,
    label: str = "",
    alpha: float = 0.2,
    smooth: bool = True,
    smooth_alpha: float = 0.9
):
    """
    绑制带置信区间的曲线
    
    参数:
        ax: matplotlib Axes 对象
        x: x 轴数据
        y_mean: y 均值
        y_std: y 标准差
        color: 颜色
        label: 图例标签
        alpha: 置信区间透明度
        smooth: 是否平滑
        smooth_alpha: 平滑系数
    """
    if smooth:
        y_mean = smooth_curve(y_mean, alpha=smooth_alpha)
        y_std = smooth_curve(y_std, alpha=smooth_alpha)
    
    # 绘制均值线
    ax.plot(x, y_mean, color=color, label=label, linewidth=1.5)
    
    # 绘制置信区间
    ax.fill_between(
        x, 
        y_mean - y_std, 
        y_mean + y_std, 
        color=color, 
        alpha=alpha,
        linewidth=0
    )


def add_constraint_line(
    ax: plt.Axes,
    value: float,
    label: str = "Constraint",
    color: str = None,
    linestyle: str = "--"
):
    """
    添加约束阈值线
    
    参数:
        ax: matplotlib Axes 对象
        value: 阈值
        label: 标签
        color: 颜色
        linestyle: 线型
    """
    if color is None:
        color = COLORS["constraint"]
    
    ax.axhline(
        y=value, 
        color=color, 
        linestyle=linestyle, 
        linewidth=1.5,
        label=label,
        alpha=0.8
    )


# ==================== 算法名称映射 ====================
ALGORITHM_NAMES = {
    "ppo_pid": "PID-Lagrangian",
    "ppo_lag": "PPO-Lagrangian",
    "ppo": "PPO (Unconstrained)",
    "cpo": "CPO",
    "mpc": "MPC",
    "random": "Random"
}

def get_algorithm_label(key: str) -> str:
    """获取算法的显示名称"""
    return ALGORITHM_NAMES.get(key, key)


def get_algorithm_style(key: str) -> dict:
    """获取算法的绑图样式"""
    return {
        "color": COLORS.get(key, "#000000"),
        "marker": MARKERS.get(key, "o"),
        "linestyle": LINESTYLES.get(key, "-"),
        "label": get_algorithm_label(key)
    }


# ==================== 初始化 ====================
if __name__ == "__main__":
    # 测试样式配置
    setup_style()
    
    # 创建测试图表
    fig, axes = create_figure(1, 2)
    
    x = np.linspace(0, 10, 100)
    for i, (algo, color) in enumerate([("ppo_pid", COLORS["ppo_pid"]), 
                                        ("ppo_lag", COLORS["ppo_lag"])]):
        y = np.sin(x + i) + np.random.randn(100) * 0.1
        plot_with_confidence(axes[0], x, y, np.abs(np.random.randn(100)*0.2), 
                           color, get_algorithm_label(algo))
    
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True)
    
    add_constraint_line(axes[1], 25, "Cost Limit")
    axes[1].plot(x, 20 + 10*np.sin(x), color=COLORS["cost"], label="Cost")
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Cost")
    axes[1].legend()
    
    plt.tight_layout()
    save_figure(fig, "test_style", formats=["png"])
    plt.show()
