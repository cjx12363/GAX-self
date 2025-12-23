# 绘图脚本包初始化
"""
科研绘图脚本集合

包含以下图表类型：
- plot_training_curves: 双轴训练曲线
- plot_multiplier_dynamics: 拉格朗日乘子演变
- plot_pareto_frontier: 帕累托前沿分析
- plot_dispatch_schedule: 24小时调度堆叠图
- plot_user_satisfaction: 用户满意度小提琴图
- plot_ablation_heatmap: 消融实验热力图

使用方法：
    # 使用模拟数据测试各图表
    python -m scripts.plotting.plot_training_curves --demo
    python -m scripts.plotting.plot_multiplier_dynamics --demo
    ...

依赖：
    pip install matplotlib seaborn numpy SciencePlots
"""

from .style_config import (
    setup_style,
    COLORS,
    MARKERS,
    LINESTYLES,
    save_figure,
    create_figure,
    smooth_curve,
    plot_with_confidence,
    add_constraint_line,
    get_algorithm_label,
    get_algorithm_style
)

__all__ = [
    'setup_style',
    'COLORS',
    'MARKERS', 
    'LINESTYLES',
    'save_figure',
    'create_figure',
    'smooth_curve',
    'plot_with_confidence',
    'add_constraint_line',
    'get_algorithm_label',
    'get_algorithm_style'
]
