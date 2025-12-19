"""环境配置文件 - 统一管理Chargax环境的创建和配置"""

from chargax import Chargax, get_electricity_prices
from chargax.util.reward_functions import profit
from chargax.util.cost_functions import safety


# ==================== 环境配置 ====================
# 充电站配置
NUM_CHARGERS = 16
NUM_CHARGERS_PER_GROUP = 2
NUM_DC_GROUPS = 5
TRANSFORMER_CAPACITY_KW = None  # None表示无限制

# 场景配置
CAR_PROFILES = "eu"  # 可选: "eu", "us", "world"
USER_PROFILES = "shopping"  # 可选: "highway", "residential", "workplace", "shopping"
ARRIVAL_FREQUENCY = 100  # 可选: "low", "medium", "high" 或具体数字

# 电价配置
ELECTRICITY_PRICE_DATASET = "2023_NL"

# 环境选项
NUM_DISCRETIZATION_LEVELS = 10
MINUTES_PER_TIMESTEP = 5
INCLUDE_BATTERY = True
ALLOW_DISCHARGING = True
CUSTOMER_SELL_PRICE = 0.75  # €/kWh
# ================================================


def create_env() -> Chargax:
    """创建并返回配置好的Chargax环境"""
    electricity_prices = get_electricity_prices(
        dataset=ELECTRICITY_PRICE_DATASET,
        minutes_per_timestep=MINUTES_PER_TIMESTEP
    )
    
    env = Chargax(
        # 电价
        elec_grid_buy_price=electricity_prices,
        elec_grid_sell_price=electricity_prices * 0.8,  # 卖电价格为买电价格的80%
        elec_customer_sell_price=CUSTOMER_SELL_PRICE,
        
        # 场景
        car_profiles=CAR_PROFILES,
        user_profiles=USER_PROFILES,
        arrival_frequency=ARRIVAL_FREQUENCY,
        
        # 充电站
        num_chargers=NUM_CHARGERS,
        num_chargers_per_group=NUM_CHARGERS_PER_GROUP,
        num_dc_groups=NUM_DC_GROUPS,
        transformer_capacity_kw=TRANSFORMER_CAPACITY_KW,
        
        # 环境选项
        num_discretization_levels=NUM_DISCRETIZATION_LEVELS,
        minutes_per_timestep=MINUTES_PER_TIMESTEP,
        include_battery=INCLUDE_BATTERY,
        allow_discharging=ALLOW_DISCHARGING,
        
        # 奖励和成本函数
        reward_fn=profit,
        cost_fn=safety,
    )
    
    return env


def get_env_info() -> dict:
    """返回环境配置信息，用于wandb记录"""
    return {
        "num_chargers": NUM_CHARGERS,
        "num_chargers_per_group": NUM_CHARGERS_PER_GROUP,
        "num_dc_groups": NUM_DC_GROUPS,
        "transformer_capacity_kw": TRANSFORMER_CAPACITY_KW,
        "car_profiles": CAR_PROFILES,
        "user_profiles": USER_PROFILES,
        "arrival_frequency": ARRIVAL_FREQUENCY,
        "electricity_price_dataset": ELECTRICITY_PRICE_DATASET,
        "num_discretization_levels": NUM_DISCRETIZATION_LEVELS,
        "minutes_per_timestep": MINUTES_PER_TIMESTEP,
        "include_battery": INCLUDE_BATTERY,
        "allow_discharging": ALLOW_DISCHARGING,
        "customer_sell_price": CUSTOMER_SELL_PRICE,
    }


def get_groupname(algorithm: str) -> str:
    """生成wandb的group名称"""
    return f"{algorithm}_{USER_PROFILES}_{CAR_PROFILES}_freq{ARRIVAL_FREQUENCY}"
