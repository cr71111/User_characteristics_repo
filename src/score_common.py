# -*- coding: utf-8 -*-
"""
公共评分函数模块
从 fact_layer.py 和 snapshot_layer.py 提取的共享逻辑，
确保 L1 和 L2 层使用完全一致的评分算法。
"""

# ── SOC 扣分阈值常量 ──
SOC_CRITICAL_RATIO_DEDUCT_THRESHOLD = 0.05
SOC_LOW_RATIO_DEDUCT_THRESHOLD = 0.20
SOC_OPTIMAL_LOWER = 40
SOC_OPTIMAL_UPPER = 80
SOC_OPTIMAL_BONUS = 3
MAX_SOC_DEDUCT = 15

# ── 电耗扣分阈值常量 ──
EXTREME_ENERGY_THRESHOLD = 13.8
HIGH_ENERGY_THRESHOLD = 12.0
MAX_NORMAL_BATTERY_CHANGE = 1
EXCESS_CHANGE_DEDUCT_PER_TIME = 4
MAX_CHANGE_DEDUCT = 8
MAX_ENERGY_DEDUCT = 12


def calc_monthly_score_v2(**kwargs):
    """
    包月友好评分计算（严格按照旧脚本逻辑，100分基准多维度扣分）
    
    Args:
        **kwargs: 包含所有评分参数的字典
        
    Returns:
        float: 0-100之间的评分
    """
    daily_distance = kwargs.get('daily_distance', 0)
    battery_change_cnt = kwargs.get('battery_change_cnt', 0)
    if daily_distance == 0 and battery_change_cnt == 0:
        return 20
    
    max_speed = kwargs.get('max_speed', 0)
    max_temp = kwargs.get('max_temp', 0)
    riding_avg_current = kwargs.get('riding_avg_current', 0)
    current_60a_count = kwargs.get('current_60a_count', 0)
    current_80a_count = kwargs.get('current_80a_count', 0)
    over100a_cont = kwargs.get('over100a_cont', 0)
    max_trip_current_cv = kwargs.get('max_trip_current_cv', 0)
    current_cv = kwargs.get('current_cv', 0)
    avg_riding_speed = kwargs.get('avg_riding_speed', 0)
    soc_data_valid = kwargs.get('soc_data_valid', False)
    has_real_ride = kwargs.get('has_real_ride', False)
    soc_below_10_ratio = kwargs.get('soc_below_10_ratio', 0)
    soc_below_20_ratio = kwargs.get('soc_below_20_ratio', 0)
    min_soc = kwargs.get('min_soc', 100)
    avg_soc = kwargs.get('avg_soc', 50)
    energy_data_valid = kwargs.get('energy_data_valid', False)
    energy_per_100km = kwargs.get('energy_per_100km', 0)
    battery_change_count = kwargs.get('battery_change_count', 0)
    
    score = 100.0

    # ── A. 速度维度（最大扣10分）──
    if max_speed >= 80:
        score -= 10
    elif max_speed >= 60:
        score -= 6
    elif max_speed >= 50:
        score -= 3

    # ── B. 温度维度（最大扣8分）──
    if max_temp >= 70:
        score -= 8
    elif max_temp >= 55:
        score -= 4

    # ── C. 电流维度（最大扣25分）──
    if riding_avg_current >= 40.4:
        score -= 15
    elif riding_avg_current >= 32.1:
        score -= 10
    elif riding_avg_current >= 28.3:
        score -= 5
    elif riding_avg_current >= 22.3:
        score -= 2

    deduct_80a = min(current_80a_count * 1.5, 8)
    score -= deduct_80a

    if over100a_cont > 0:
        score -= min(over100a_cont * 4, 12)

    if max_trip_current_cv >= 1.5:
        score -= 4
    elif current_cv >= 0.8:
        score -= 2 if avg_riding_speed <= 15 else 4
    elif current_cv >= 0.5:
        score -= 1

    # ── D. SOC 维度（最大扣15分）──
    soc_deduct = 0.0
    if soc_data_valid and has_real_ride:
        if soc_below_10_ratio >= SOC_CRITICAL_RATIO_DEDUCT_THRESHOLD:
            soc_deduct += min((soc_below_10_ratio - SOC_CRITICAL_RATIO_DEDUCT_THRESHOLD) / 0.01 * 2, 8)

        if soc_below_20_ratio >= SOC_LOW_RATIO_DEDUCT_THRESHOLD:
            soc_deduct += min((soc_below_20_ratio - SOC_LOW_RATIO_DEDUCT_THRESHOLD) / 0.02 * 1.5, 5)

        if min_soc <= 5:
            soc_deduct += 4
        elif min_soc <= 10:
            soc_deduct += 2

        if (SOC_OPTIMAL_LOWER <= avg_soc <= SOC_OPTIMAL_UPPER) and soc_below_20_ratio == 0:
            score += SOC_OPTIMAL_BONUS

    score -= min(soc_deduct, MAX_SOC_DEDUCT)

    # ── E. 电耗维度（最大扣12分）──
    energy_deduct = 0.0
    if energy_data_valid and has_real_ride:
        if energy_per_100km >= EXTREME_ENERGY_THRESHOLD:
            energy_deduct += 8
        elif energy_per_100km >= HIGH_ENERGY_THRESHOLD:
            energy_deduct += 4
        elif energy_per_100km >= 10.56:
            energy_deduct += 2

        excess = max(0, battery_change_count - MAX_NORMAL_BATTERY_CHANGE)
        energy_deduct += min(excess * EXCESS_CHANGE_DEDUCT_PER_TIME, MAX_CHANGE_DEDUCT)

    score -= min(energy_deduct, MAX_ENERGY_DEDUCT)

    return max(0, min(100, round(score)))


def determine_user_level_v2(**kwargs):
    """
    用户等级判定（严格按照旧脚本逻辑）
    
    Args:
        **kwargs: 包含所有判定参数的字典
        
    Returns:
        str: 用户等级名称
    """
    over100a_cont = kwargs.get('over100a_cont', 0)
    over100a_hours = kwargs.get('over100a_hours', 0)
    soc_below_10_ratio = kwargs.get('soc_below_10_ratio', 0)
    min_soc = kwargs.get('min_soc', 100)
    current_60a_count = kwargs.get('current_60a_count', 0)
    energy_per_100km = kwargs.get('energy_per_100km', 0)
    soc_below_20_ratio = kwargs.get('soc_below_20_ratio', 0)
    score = kwargs.get('score', 0)
    riding_avg_current = kwargs.get('riding_avg_current', 0)
    soc_data_valid = kwargs.get('soc_data_valid', False)
    energy_data_valid = kwargs.get('energy_data_valid', False)
    
    is_violent = False

    if (over100a_cont >= 2) or (over100a_hours >= 0.1):
        is_violent = True
    elif soc_data_valid and (soc_below_10_ratio >= 0.20) and (min_soc <= 5):
        is_violent = True

    is_high_loss = False

    if energy_data_valid and (energy_per_100km >= 17.0):
        is_high_loss = True
    elif (riding_avg_current >= 32.1) and soc_data_valid:
        if soc_below_20_ratio >= 0.10:
            is_high_loss = True
    elif soc_data_valid and (soc_below_20_ratio >= 0.20):
        is_high_loss = True
    elif (score < 30) and energy_data_valid:
        is_high_loss = True

    if is_violent:
        return "暴力"
    elif is_high_loss:
        return "高损耗用户"
    else:
        if score >= 75:
            return "优质用户"
        elif score >= 60:
            return "良好用户"
        elif score >= 45:
            return "普通用户"
        else:
            return "高损耗用户"
