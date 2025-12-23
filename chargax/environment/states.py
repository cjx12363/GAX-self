import chex
from typing import Union, List, Optional, Literal
import jax.numpy as jnp
import numpy as np
import jax
import equinox as eqx
from dataclasses import replace, fields

@chex.dataclass(frozen=True)
class CarProfiles:
    frequencies: chex.Array
    threshold_tau: chex.Array
    capacity: chex.Array
    ac_max_rate: chex.Array
    dc_max_rate: chex.Array

class ChargersState(eqx.Module):
    # 车辆变量
    car_time_till_leave: chex.Array
    car_battery_now_kw: chex.Array
    car_battery_capacity_kw: chex.Array
    car_desired_battery_percentage: chex.Array
    car_arrival_battery_kw: chex.Array # 用于补偿/阻止智能体放电超过到达时的电量
    charge_sensitive: chex.Array # False = 时间敏感型

    # 我们需要跟踪每个 EV 的放电情况
    # 当我们放电并随后再次充电时，我们不能让
    # 客户支付两次电费
    car_discharged_this_session_kw: chex.Array 

    car_ac_absolute_max_charge_rate_kw: chex.Array
    car_ac_optimal_charge_threshold: chex.Array
    car_dc_absolute_max_charge_rate_kw: chex.Array
    car_dc_optimal_charge_threshold: chex.Array
    
    # 充电桩变量
    charger_current_now: chex.Array
    charger_is_car_connected: chex.Array
    charger_voltage: np.ndarray # 我们假设所有充电桩的电压都是常数
    charger_is_dc: np.ndarray 
    # 最大电流在 EVSE 中设置

    @property
    def car_battery_percentage(self) -> jnp.ndarray:
        return self.car_battery_now_kw / (self.car_battery_capacity_kw + 1e-8)
    
    @property
    def car_battery_desired_remaining(self) -> jnp.ndarray:
        return self.car_desired_battery_percentage - self.car_battery_percentage

    @property
    def car_battery_desired_remaining_kw(self) -> jnp.ndarray:
        desired_battery_kw = self.car_desired_battery_percentage * self.car_battery_capacity_kw
        return desired_battery_kw - self.car_battery_now_kw

    @property
    def charger_output_now_kw(self) -> jnp.ndarray:
        return (self.charger_voltage * self.charger_current_now) / 1000.0
    
    @property
    def charger_throughput_now_kw(self) -> jnp.ndarray:
        """ 使用电流的绝对值来计算吞吐量 """
        return (self.charger_voltage * jnp.abs(self.charger_current_now)) / 1000.0
    
    @property
    def car_max_current_intake(self) -> jnp.ndarray:
        return self._car_max_current(self.car_battery_percentage)
    
    @property
    def car_max_current_outtake(self) -> jnp.ndarray:
        return self._car_max_current(1 - self.car_battery_percentage)

    def _car_max_current(self, battery_percentage: jnp.ndarray) -> jnp.ndarray:
        tau, abs_max_rate = jax.tree.map(
            lambda x, y: jnp.where(
                self.charger_is_dc, x, y
            ), 
            (self.car_dc_optimal_charge_threshold, self.car_dc_absolute_max_charge_rate_kw),
            (self.car_ac_optimal_charge_threshold, self.car_ac_absolute_max_charge_rate_kw)
        )
        # 达到阈值后，充电速率线性衰减至 5%
        max_charge_rate_kw = jnp.where(
            battery_percentage > tau,
            abs_max_rate * (1 - (battery_percentage - tau) / (1 - tau) + 0.10),
            abs_max_rate
        ) * self.charger_is_car_connected # 如果未连接车辆，充电速率为 0
        max_charge_rate_w = max_charge_rate_kw * 1000.0
        return (max_charge_rate_w / (self.charger_voltage + 1e-8)) # 添加极小值以避免除以零

    def __init__(
            self, 
            station: 'ChargingStation' = None, 
            **kwargs
        ):
        if kwargs: # 允许使用 replace(..., **kwargs)
            self.__dict__.update(kwargs)
            return
        
        num_chargers = station.num_chargers

        # if sample_method == "empty":
        for field in fields(self):
            setattr(self, field.name, jnp.zeros(num_chargers))

        # 设置类型
        self.car_time_till_leave = self.car_time_till_leave.astype(int)
        self.charge_sensitive = self.charge_sensitive.astype(bool)
        
        # 设置充电桩电压和 dc/ac
        rated_voltages = [np.repeat(evse.voltage_rated, len(evse.connections)) for evse in station.evses]
        max_kw = [np.repeat(evse.group_capacity_max_kw, len(evse.connections)) for evse in station.evses]
        voltages_per_charger = np.concatenate(rated_voltages)
        max_kw_per_charger = np.concatenate(max_kw)
        self.charger_voltage = voltages_per_charger
        self.charger_is_dc = max_kw_per_charger > 50.0

        # 对每种采样方法初始化这些空值
        self.charger_current_now = jnp.zeros(num_chargers)
        self.charger_is_car_connected = jnp.zeros(num_chargers, dtype=bool)
        self.car_discharged_this_session_kw = jnp.zeros(num_chargers)
        

class StationSplitter(eqx.Module):
    """
    Splitter 代表配电盘、电缆、变压器或其他辅助设备的任何组合，
    这些设备可能会导致损耗并属于充电网络的一部分。
    一个 Splitter 可以包含：
    - EVSE
    - 其他 Splitter
    """
    connections: List[Union['StationEVSE', 'StationSplitter']]
    group_capacity_max_kw: float = eqx.field(static=True)
    efficiency: float = eqx.field(static=True, default=0.995)

    @property
    def charger_ids_per_children_evse(self) -> np.ndarray:
        return jax.tree.leaves(self.connections)
    
    @property
    def charger_ids_children(self) -> np.ndarray:
        return np.concatenate(self.charger_ids_per_children_evse)
    
    @property
    def number_of_chargers_children(self) -> int:
        return len(self.charger_ids_children)
    
    @property
    def evses_children(self) -> List['StationEVSE']:
        return jax.tree.leaves(self.connections, is_leaf=lambda x: isinstance(x, StationEVSE))
    
    @property
    def splitters_children(self) -> List['StationSplitter']:
        return jax.tree.leaves(self.connections, is_leaf=lambda x: isinstance(x, StationSplitter))

    @property
    def efficiency_per_charger(self) -> jnp.ndarray:
        efficiency_to_chargepoint = np.ones(self.number_of_chargers_children)
        for path, charger_ids in jax.tree.leaves_with_path(self):
            curr_node = self
            for node in path[:-1]: # 省略最后一个节点（即充电桩）
                if isinstance(node, jax.tree_util.GetAttrKey):
                    curr_node = curr_node.connections
                elif isinstance(node, jax.tree_util.SequenceKey):
                    curr_node = curr_node[node.idx]
                    efficiency_to_chargepoint[charger_ids] *= (
                        curr_node.efficiency
                    )
                    
        efficiency_to_chargepoint *= self.efficiency

        return efficiency_to_chargepoint
    
    def total_kw_throughput(self, charger_state: 'ChargersState') -> float:
        children_charger_outputs = charger_state.charger_throughput_now_kw[self.charger_ids_children]
        return jnp.sum(children_charger_outputs)

    def normalize_currents(self, charger_state: 'ChargersState') -> 'ChargersState':

        max_capacity = self.group_capacity_max_kw
        curr_load = self.total_kw_throughput(charger_state)
        normalization_factor = jax.lax.select(
            curr_load > max_capacity,
            max_capacity / curr_load,
            1.
        )
        currents = charger_state.charger_current_now
        currents = currents.at[self.charger_ids_children].set(
            currents[self.charger_ids_children] * normalization_factor
        )
        return replace(
            charger_state,
            charger_current_now=currents
        )
    
    def get_parent(self, root: 'StationSplitter'):
        """
        从根节点开始寻找当前 StationSplitter 实例的父节点。

        参数:
            root (StationSplitter): 开始搜索的树根节点。
        
        返回:
            Optional[StationSplitter]: 如果找到，则返回当前实例的父节点，否则返回 None。
        """
        for connection in root.connections:
            if isinstance(connection, StationSplitter):
                if self in connection.connections:
                    return connection
                parent = self.get_parent(connection)
                if parent:
                    return parent
        return None
    
class StationEVSE(StationSplitter):
    """
    一个 EVSE -- 等级结构中的最终 splitter -- 是连接到充电桩的 splitter。
    """
    connections: np.ndarray = eqx.field(converter=np.asarray) # 充电桩索引
    voltage_rated: float = eqx.field(static=True)
    current_max: float = eqx.field(static=True)

    def __init__(self, connections: np.ndarray, voltage_rated: float = 230.0, current_max: float = 50.0, **kwargs):
        self.__dict__.update(kwargs)
        self.connections = connections
        self.voltage_rated = voltage_rated
        self.current_max = current_max
        self.group_capacity_max_kw = (self.voltage_rated * self.current_max) / 1000.0

class StationBattery(eqx.Module):
    """
    充电中心使用的蓄电池。可用于储存多余能量或向电网供电。
    """
    capacity_kw: float = 100000.0
    battery_now: float = 0.0
    max_rate_kw: float = 1000.0
    tau: float = 1.0

    @property
    def battery_percentage(self) -> float:
        return self.battery_now / self.capacity_kw

class ChargingStation(eqx.Module):
    """
        充电站是充电桩层级结构的最高层。
        它包含充电桩的拓扑结构，作为一个 ChargerNode 树，
        其中根节点是电网连接。
        以及一个包含所有充电桩状态的单一对象。
    """
    charger_layout: StationSplitter

    def __init__(self, num_chargers: int = 16, num_chargers_per_group: int = 2, num_dc_groups: int = 5, transformer_capacity_kw: float = None):
        from .station_builder import build_default_station
        self.charger_layout = build_default_station(
            num_chargers=num_chargers,
            num_chargers_per_group=num_chargers_per_group,
            num_dc_groups=num_dc_groups,
            transformer_capacity_kw=transformer_capacity_kw
        )

    @property
    def root(self) -> StationSplitter:
        """ 用于获取充电桩布局根节点的便捷方法 """
        return self.charger_layout

    @property
    def num_chargers(self) -> int:
        return self.charger_layout.number_of_chargers_children
    
    @property
    def charger_ids(self) -> np.ndarray:
        return self.charger_layout.charger_ids_children
    
    @property
    def charger_ids_per_evse(self) -> np.ndarray:
        return self.charger_layout.charger_ids_per_children_evse
    
    @property
    def evses(self) -> List[StationEVSE]:
        return self.charger_layout.evses_children
    
    @property
    def splitters(self) -> List[StationSplitter]:
        return self.charger_layout.splitters_children
    
@chex.dataclass(frozen=True)
class EnvState:
    day_of_year: int # Sampled at reset()

    chargers_state: ChargersState
    battery_state: StationBattery = StationBattery()
    timestep: int = 0
    is_workday: bool = True

    # 奖励变量
    profit: float = 0.0
    uncharged_percentages: float = 0.0
    uncharged_kw: float = 0.0
    charged_overtime: int = 0 # 超过期望充电时间的分钟数
    charged_undertime: int = 0 # 低于期望充电时间的分钟数（正向奖励）
    rejected_customers: int = 0
    left_customers: int = 0
    exceeded_capacity: float = 0.0
    total_charged_kw: float = 0.0
    total_discharged_kw: float = 0.0
