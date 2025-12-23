"""
Chargax Utility Functions

简化设计：
- reward: profit（利润）
- cost: safety（变压器过载，归一化到[0,1]）
"""

from chargax.util.cost_functions import safety
from chargax.util.reward_functions import profit
from chargax.util.training_logger import TrainingLogger, init_logger, get_logger

__all__ = [
    "safety",   # Cost: 变压器过载约束 [0, 1]
    "profit",   # Reward: 利润
    "TrainingLogger",  # 训练数据记录器
    "init_logger",
    "get_logger",
]

