"""
充电站拓扑构建器

定义不同的充电站物理布局配置
"""

import numpy as np
from .states import StationSplitter, StationEVSE


def build_default_station(
    num_chargers: int = 16,
    num_chargers_per_group: int = 2,
    num_dc_groups: int = 5,
    transformer_capacity_kw: float = None,
    dc_voltage: float = 500.0,
    dc_current: float = 300.0,
    ac_voltage: float = 230.0,
    ac_current: float = 50.0,
) -> StationSplitter:
    """
    构建默认充电站拓扑
    
    Args:
        num_chargers: 充电桩总数
        num_chargers_per_group: 每组充电桩数
        num_dc_groups: DC快充组数
        transformer_capacity_kw: 变压器容量限制 (kW)，None表示无限制
        dc_voltage: DC充电桩电压 (V)
        dc_current: DC充电桩电流 (A)
        ac_voltage: AC充电桩电压 (V)
        ac_current: AC充电桩电流 (A)
    
    Returns:
        StationSplitter: 充电站根节点
    
    拓扑结构:
        电网 ←→ [变压器/根节点]
                    │
        ┌───────────┴───────────┐
        │                       │
        DC快充区              AC慢充区
        (num_dc_groups组)     (剩余组)
    """
    assert num_chargers % num_chargers_per_group == 0, "充电桩数必须能被每组数量整除"
    assert num_chargers_per_group >= 1, "每组至少1个充电桩"
    assert num_chargers > num_chargers_per_group, "总数必须大于每组数量"

    charger_indices = np.arange(num_chargers)
    charger_indices = charger_indices.reshape(-1, num_chargers_per_group)

    # DC快充组
    DC_EVSEs = [
        StationEVSE(
            connections=ci,
            voltage_rated=dc_voltage,
            current_max=dc_current
        )
        for ci in charger_indices[:num_dc_groups]
    ]
    
    # AC慢充组
    AC_EVSEs = [
        StationEVSE(
            connections=ci,
            voltage_rated=ac_voltage,
            current_max=ac_current
        )
        for ci in charger_indices[num_dc_groups:]
    ]
    
    EVSEs = DC_EVSEs + AC_EVSEs

    # 计算总容量
    combined_total_capacity = sum([evse.group_capacity_max_kw for evse in EVSEs])
    actual_capacity = transformer_capacity_kw if transformer_capacity_kw is not None else combined_total_capacity

    # 构建根节点（变压器）
    grid_connection_node = StationSplitter(
        connections=EVSEs,
        group_capacity_max_kw=actual_capacity
    )

    return grid_connection_node


def build_hierarchical_station(
    num_dc_chargers: int = 10,
    num_ac_chargers: int = 6,
    dc_transformer_capacity_kw: float = 800.0,
    ac_transformer_capacity_kw: float = 100.0,
    total_transformer_capacity_kw: float = 500.0,
) -> StationSplitter:
    """
    构建分层变压器拓扑
    
    拓扑结构:
        电网 ←→ [主变压器 500kW]
                    │
        ┌───────────┴───────────┐
        │                       │
    [DC变压器 800kW]      [AC变压器 100kW]
        │                       │
    DC充电桩×10            AC充电桩×6
    """
    # DC充电桩
    dc_indices = np.arange(num_dc_chargers).reshape(-1, 2)
    DC_EVSEs = [
        StationEVSE(connections=ci, voltage_rated=500.0, current_max=300.0)
        for ci in dc_indices
    ]
    dc_splitter = StationSplitter(
        connections=DC_EVSEs,
        group_capacity_max_kw=dc_transformer_capacity_kw
    )

    # AC充电桩
    ac_indices = np.arange(num_dc_chargers, num_dc_chargers + num_ac_chargers).reshape(-1, 2)
    AC_EVSEs = [
        StationEVSE(connections=ci, voltage_rated=230.0, current_max=50.0)
        for ci in ac_indices
    ]
    ac_splitter = StationSplitter(
        connections=AC_EVSEs,
        group_capacity_max_kw=ac_transformer_capacity_kw
    )

    # 主变压器
    root = StationSplitter(
        connections=[dc_splitter, ac_splitter],
        group_capacity_max_kw=total_transformer_capacity_kw
    )

    return root
