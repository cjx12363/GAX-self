"""
训练数据记录器模块

用于记录 PID-Lagrangian 算法训练过程中的各项指标，
支持多种子运行数据聚合和科研绘图数据导出。

数据结构：
=========
{
    "metadata": {
        "algorithm": "ppo_pid",
        "seed": 42,
        "config": {...}
    },
    "iterations": [
        {
            "step": 1000,
            "reward": 123.45,
            "cost": 15.2,
            "multiplier": [0.05],
            "pid_i": [0.03],
            "eval_reward": 130.0,
            "eval_cost": 12.0
        },
        ...
    ],
    "episodes": [
        {
            "schedule": [...],  # 24小时调度数据
            "soc_final": [...],  # 用户最终SOC
            "soc_target": [...]  # 用户目标SOC
        }
    ]
}
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime


@dataclass
class IterationLog:
    """单次迭代的日志数据"""
    step: int
    reward: float
    cost: float
    multiplier: List[float]
    pid_i: List[float]
    eval_reward: float = 0.0
    eval_cost: float = 0.0
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0


@dataclass
class EpisodeLog:
    """单个episode的详细数据（用于调度图和满意度分析）"""
    # 24小时调度数据 (288步, 每步5分钟)
    base_load: List[float] = field(default_factory=list)  # 基础负荷 kW
    ev_load: List[float] = field(default_factory=list)    # EV充电负荷 kW
    electricity_price: List[float] = field(default_factory=list)  # 电价
    
    # 用户满意度数据
    soc_final: List[float] = field(default_factory=list)  # 离场时实际SOC
    soc_target: List[float] = field(default_factory=list)  # 目标SOC
    
    # 时间戳
    timestamp: str = ""


@dataclass
class TrainingLog:
    """完整的训练日志"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    iterations: List[IterationLog] = field(default_factory=list)
    episodes: List[EpisodeLog] = field(default_factory=list)


class TrainingLogger:
    """
    训练数据记录器
    
    用于记录训练过程中的各项指标，支持：
    - 迭代级别的 reward/cost/λ 等指标
    - Episode 级别的调度和满意度数据
    - JSON 格式导出
    - 多种子运行数据聚合
    
    使用示例：
    ---------
    logger = TrainingLogger(algorithm="ppo_pid", seed=42)
    logger.log_iteration(step=1000, reward=100.0, cost=20.0, multiplier=[0.1])
    logger.save("logs/run_42.json")
    
    # 聚合多个种子的数据
    logs = TrainingLogger.aggregate_seeds(["logs/run_1.json", "logs/run_2.json"])
    """
    
    def __init__(
        self, 
        algorithm: str = "ppo_pid",
        seed: int = 42,
        config: Optional[Dict] = None
    ):
        """
        初始化记录器
        
        参数:
            algorithm: 算法名称 ("ppo", "ppo_pid", "ppo_lag", etc.)
            seed: 随机种子
            config: 训练配置字典
        """
        self.log = TrainingLog(
            metadata={
                "algorithm": algorithm,
                "seed": seed,
                "config": config or {},
                "start_time": datetime.now().isoformat(),
                "version": "1.0"
            }
        )
        
    def log_iteration(
        self,
        step: int,
        reward: float,
        cost: float,
        multiplier: Union[float, List[float], np.ndarray],
        pid_i: Optional[Union[float, List[float], np.ndarray]] = None,
        eval_reward: float = 0.0,
        eval_cost: float = 0.0,
        actor_loss: float = 0.0,
        critic_loss: float = 0.0,
        entropy: float = 0.0
    ):
        """
        记录单次迭代的数据
        
        参数:
            step: 当前时间步
            reward: 累积奖励
            cost: 累积代价
            multiplier: 拉格朗日乘子 (标量或数组)
            pid_i: PID积分项 (可选)
            eval_reward: 评估奖励
            eval_cost: 评估代价
            actor_loss: Actor损失
            critic_loss: Critic损失
            entropy: 策略熵
        """
        # 转换为列表格式
        if isinstance(multiplier, (int, float)):
            multiplier = [float(multiplier)]
        elif isinstance(multiplier, np.ndarray):
            multiplier = multiplier.tolist()
        else:
            multiplier = list(multiplier)
            
        if pid_i is None:
            pid_i = [0.0] * len(multiplier)
        elif isinstance(pid_i, (int, float)):
            pid_i = [float(pid_i)]
        elif isinstance(pid_i, np.ndarray):
            pid_i = pid_i.tolist()
        else:
            pid_i = list(pid_i)
        
        iteration = IterationLog(
            step=int(step),
            reward=float(reward),
            cost=float(cost),
            multiplier=multiplier,
            pid_i=pid_i,
            eval_reward=float(eval_reward),
            eval_cost=float(eval_cost),
            actor_loss=float(actor_loss),
            critic_loss=float(critic_loss),
            entropy=float(entropy)
        )
        self.log.iterations.append(iteration)
        
    def log_episode(
        self,
        base_load: Optional[List[float]] = None,
        ev_load: Optional[List[float]] = None,
        electricity_price: Optional[List[float]] = None,
        soc_final: Optional[List[float]] = None,
        soc_target: Optional[List[float]] = None
    ):
        """
        记录单个episode的详细数据
        
        参数:
            base_load: 基础负荷序列 (kW)
            ev_load: EV充电负荷序列 (kW)
            electricity_price: 电价序列
            soc_final: 用户离场时的实际SOC列表
            soc_target: 用户目标SOC列表
        """
        def to_list(x):
            if x is None:
                return []
            if isinstance(x, np.ndarray):
                return x.tolist()
            return list(x)
        
        episode = EpisodeLog(
            base_load=to_list(base_load),
            ev_load=to_list(ev_load),
            electricity_price=to_list(electricity_price),
            soc_final=to_list(soc_final),
            soc_target=to_list(soc_target),
            timestamp=datetime.now().isoformat()
        )
        self.log.episodes.append(episode)
        
    def save(self, path: str):
        """
        保存日志到JSON文件
        
        参数:
            path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # 更新结束时间
        self.log.metadata["end_time"] = datetime.now().isoformat()
        
        # 转换为可序列化的字典
        data = {
            "metadata": self.log.metadata,
            "iterations": [asdict(it) for it in self.log.iterations],
            "episodes": [asdict(ep) for ep in self.log.episodes]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"日志已保存到: {path}")
        
    @classmethod
    def load(cls, path: str) -> 'TrainingLogger':
        """
        从JSON文件加载日志
        
        参数:
            path: 文件路径
            
        返回:
            TrainingLogger 实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger = cls(
            algorithm=data["metadata"].get("algorithm", "unknown"),
            seed=data["metadata"].get("seed", 0),
            config=data["metadata"].get("config", {})
        )
        logger.log.metadata = data["metadata"]
        logger.log.iterations = [
            IterationLog(**it) for it in data.get("iterations", [])
        ]
        logger.log.episodes = [
            EpisodeLog(**ep) for ep in data.get("episodes", [])
        ]
        
        return logger
    
    @staticmethod
    def aggregate_seeds(paths: List[str]) -> Dict[str, Any]:
        """
        聚合多个种子运行的数据
        
        参数:
            paths: 日志文件路径列表
            
        返回:
            包含均值和标准差的聚合数据
        """
        all_rewards = []
        all_costs = []
        all_multipliers = []
        all_steps = None
        
        for path in paths:
            logger = TrainingLogger.load(path)
            steps = [it.step for it in logger.log.iterations]
            rewards = [it.reward for it in logger.log.iterations]
            costs = [it.cost for it in logger.log.iterations]
            multipliers = [it.multiplier[0] if it.multiplier else 0.0 
                          for it in logger.log.iterations]
            
            if all_steps is None:
                all_steps = steps
            
            all_rewards.append(rewards)
            all_costs.append(costs)
            all_multipliers.append(multipliers)
        
        # 计算统计量
        all_rewards = np.array(all_rewards)
        all_costs = np.array(all_costs)
        all_multipliers = np.array(all_multipliers)
        
        return {
            "steps": all_steps,
            "reward_mean": np.mean(all_rewards, axis=0).tolist(),
            "reward_std": np.std(all_rewards, axis=0).tolist(),
            "cost_mean": np.mean(all_costs, axis=0).tolist(),
            "cost_std": np.std(all_costs, axis=0).tolist(),
            "multiplier_mean": np.mean(all_multipliers, axis=0).tolist(),
            "multiplier_std": np.std(all_multipliers, axis=0).tolist(),
            "num_seeds": len(paths)
        }
    
    def get_data_for_plotting(self) -> Dict[str, np.ndarray]:
        """
        获取用于绘图的数据
        
        返回:
            包含各项指标数组的字典
        """
        iterations = self.log.iterations
        
        return {
            "steps": np.array([it.step for it in iterations]),
            "rewards": np.array([it.reward for it in iterations]),
            "costs": np.array([it.cost for it in iterations]),
            "multipliers": np.array([it.multiplier[0] if it.multiplier else 0.0 
                                     for it in iterations]),
            "eval_rewards": np.array([it.eval_reward for it in iterations]),
            "eval_costs": np.array([it.eval_cost for it in iterations])
        }


# 便捷函数：创建全局logger实例
_global_logger: Optional[TrainingLogger] = None


def init_logger(algorithm: str, seed: int, config: Optional[Dict] = None) -> TrainingLogger:
    """初始化全局logger"""
    global _global_logger
    _global_logger = TrainingLogger(algorithm=algorithm, seed=seed, config=config)
    return _global_logger


def get_logger() -> Optional[TrainingLogger]:
    """获取全局logger"""
    return _global_logger
