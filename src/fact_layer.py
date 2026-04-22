# -*- coding: utf-8 -*-
"""
L1 Fact Layer - 客观事实层
职责：从清洁原始行计算可观测的物理事实，不含任何业务推断。

输入：raw/{date}.parquet
输出：fact/daily/{date}.parquet
"""

import os
import sys
import time
import math
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

warnings.filterwarnings('ignore')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import BASE_EXPORT_PATH, EXPORT_PATH_FACT_DAILY, EXPORT_PATH_RAW
from utils.geo import batch_gps_to_region

# ==============================================================================
# 全局配置参数（严格按照旧脚本）
# ==============================================================================
DISTANCE_UNIT_KM = True
GPS_FILTER = True
CSV_ENCODING = 'utf-8'
COLLECTION_CYCLE_MIN = 5
MAX_SPEED_KMH = 120
DRIFT_SPEED_THRESHOLD = 150
RADIUS_QUANTILES = [0.9, 0.95]
MIN_VALID_DISPLACEMENT_M = 20
STOP_SMOOTH_WINDOW = 5
MIN_RIDING_DURATION_MIN = 3
MAX_TRIP_GAP_MIN = 60
VALID_RIDE_MIN_HOUR = 0.1
VALID_RIDE_HOUR_FOR_PATTERN = 0.5
PATTERN_DOMINANT_RATIO = 0.5
# 地摊/储能场景专属判定参数（严格按照旧脚本）
STORAGE_MIN_VALID_GPS_POINTS = 10
STORAGE_MAX_DISTANCE_KM = 1.0
STORAGE_MAX_AVG_SPEED_KMH = 3.0
STORAGE_DISCHARGE_RATIO_THRESHOLD = 3.0
STORAGE_MIN_DISCHARGE_HOUR = 2.0
STORAGE_RIDE_DURATION_RATIO = 0.1

# 改装/超速判定阈值（严格按照旧脚本）
MODIFY_SPEED_THRESHOLD = 50
MODIFY_CURRENT_THRESHOLD = 24

# 数据有效性阈值（严格按照旧脚本）
MAX_VALID_CURRENT = 150.0
MAX_VALID_SPEED = 100.0
MIN_VALID_CURRENT = 0.1

# 电流异常判定阈值（严格按照旧脚本）
ABNORMAL_MAX_CURRENT = 80.0
ABNORMAL_CV = 1.5
ABNORMAL_AVG_CURRENT = 35.0
ABNORMAL_MIN_DISCHARGE_HOUR = 2.0

# 时段划分（严格按照旧脚本，使用list以支持+操作）
NOON_PEAK_HOURS = [11, 12, 13]
EVENING_PEAK_HOURS = [17, 18, 19]
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
OFFPEAK_HOURS = [h for h in range(24) if h not in NOON_PEAK_HOURS + EVENING_PEAK_HOURS + NIGHT_HOURS]

# 电流阈值体系（严格按照旧脚本）
NORMAL_CURRENT_THRESHOLD = 60
HIGH_CURRENT_THRESHOLD = 80
OVER_CURRENT_THRESHOLD = 100
OVER_CURRENT_CONTINUOUS = 2
OVER_CURRENT_MIN_HOUR = 0.1

# SOC判定阈值（严格按照旧脚本）
SOC_LOW_WARNING = 20
SOC_CRITICAL = 10

# 用户等级判定阈值（严格按照旧脚本）
VIOLENT_CURRENT_TIMES = 2
HIGH_LOSS_CURRENT_TIMES = 3

# 包月友好评分扣分阈值（严格按照旧脚本）
SOC_CRITICAL_RATIO_DEDUCT_THRESHOLD = 0.05
SOC_LOW_RATIO_DEDUCT_THRESHOLD = 0.10
SOC_OPTIMAL_LOWER = 30
SOC_OPTIMAL_UPPER = 80
SOC_OPTIMAL_BONUS = 3
MAX_SOC_DEDUCT = 15
EXTREME_ENERGY_THRESHOLD = 17.0
HIGH_ENERGY_THRESHOLD = 13.8
MAX_ENERGY_DEDUCT = 12
MAX_NORMAL_BATTERY_CHANGE = 3
EXCESS_CHANGE_DEDUCT_PER_TIME = 2
MAX_CHANGE_DEDUCT = 8

# ==============================================================================
# 辅助函数
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2, unit='km'):
    """Haversine公式计算地球表面两点间距离"""
    R = 6371 if unit == 'km' else 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def get_real_center(group):
    """计算真实中心点（去除异常GPS点）"""
    lat_data = group['纬度'].dropna()
    lon_data = group['经度'].dropna()
    if len(lat_data) < 3:
        return lat_data.mean() if len(lat_data) > 0 else np.nan, lon_data.mean() if len(lon_data) > 0 else np.nan
    
    lat_q1, lat_q3 = lat_data.quantile(0.25), lat_data.quantile(0.75)
    lon_q1, lon_q3 = lon_data.quantile(0.25), lon_data.quantile(0.75)
    lat_iqr, lon_iqr = lat_q3 - lat_q1, lon_q3 - lon_q1
    
    valid_mask = (
        (lat_data >= lat_q1 - 1.5 * lat_iqr) & (lat_data <= lat_q3 + 1.5 * lat_iqr) &
        (lon_data >= lon_q1 - 1.5 * lon_iqr) & (lon_data <= lon_q3 + 1.5 * lon_iqr)
    )
    
    if valid_mask.sum() > 0:
        return lat_data[valid_mask].mean(), lon_data[valid_mask].mean()
    return lat_data.mean(), lon_data.mean()


def _calc_monthly_score_v2(**kwargs):
    """包月友好评分计算（严格按照旧脚本逻辑，100分基准多维度扣分）"""
    # 沉默用户识别提前至基础计算层（修复Bug2：高分悖论）
    daily_distance = kwargs.get('daily_distance', 0)
    battery_change_cnt = kwargs.get('battery_change_cnt', 0)
    if daily_distance == 0 and battery_change_cnt == 0:
        return 20  # 沉默用户统一给20分，避免日级数据得满分、7天聚合得20分的冲突
    
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

    # 电流>80A次数扣分（排除>100A的部分）
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


def _determine_user_level_v2(**kwargs):
    """用户等级判定（严格按照旧脚本逻辑）"""
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
    
    # 暴力用户判定
    is_violent = False
    if (over100a_cont >= 2) or (over100a_hours >= 0.1):
        is_violent = True
    elif soc_data_valid and (soc_below_10_ratio >= 0.20) and (min_soc <= 5):
        is_violent = True

    # 高损耗用户判定
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


# ==============================================================================
# 数据预处理
# ==============================================================================
def preprocess_raw_data(df_raw):
    """数据预处理（严格按照旧脚本逻辑）"""
    df = df_raw.copy()
    
    # 字段类型转换
    for col in ['时间戳', '纬度', '经度', '电流', '温度', '速度']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 虚假速度过滤
    if '速度' in df.columns and '电流' in df.columns:
        df['速度'] = np.where(
            (df['电流'] < -20) & (df['速度'] > 5), 
            0, 
            df['速度']
        )
        df['速度'] = np.where(
            (df['电流'] > -5) & (df['电流'] <= 0) & (df['速度'] < 3), 
            0, 
            df['速度']
        )
    
    # 电流分层清洗
    df['电流_放电统计用'] = df['电流'].copy()
    df['电流_放电统计用'] = np.where(
        (df['电流_放电统计用'] > 0) & (df['电流_放电统计用'] <= 200), 
        df['电流_放电统计用'], 
        np.nan
    )
    df['电流_骑行判定用'] = np.where(
        (df['电流'] >= 1.0) & (df['电流'] <= MAX_VALID_CURRENT), 
        df['电流'], 
        np.nan
    )
    
    # 电流异常值过滤（中位数平滑）
    if '电流_骑行判定用' in df.columns:
        df = df.sort_values(['合约id', '用户id', '时间戳'])
        df['电流_前1'] = df.groupby(['合约id', '用户id'])['电流_骑行判定用'].shift(1)
        df['电流_后1'] = df.groupby(['合约id', '用户id'])['电流_骑行判定用'].shift(-1)
        df['电流_median'] = df[['电流_前1', '电流_骑行判定用', '电流_后1']].median(axis=1)
        df['电流_骑行判定用'] = np.where(
            abs(df['电流_骑行判定用'] - df['电流_median']) > 200, 
            np.nan, 
            df['电流_骑行判定用']
        )
        df = df.drop(columns=['电流_前1', '电流_后1', '电流_median'], errors='ignore')
    
    # 温度过滤
    if '温度' in df.columns:
        df = df[(df['温度'].isna()) | (df['温度'] <= 100)]
    
    # GPS过滤
    df = df.dropna(subset=['时间戳', '合约id', '用户id', '纬度', '经度']).copy()
    if GPS_FILTER:
        df = df[(df['纬度'] >= 3) & (df['纬度'] <= 54) & (df['经度'] >= 73) & (df['经度'] <= 136)]
    
    # 漂移速度过滤（严格按照旧脚本）
    df = df.sort_values(['合约id', '用户id', '时间戳'])
    df['漂移_前纬度'] = df.groupby(['合约id', '用户id'])['纬度'].shift(1)
    df['漂移_前经度'] = df.groupby(['合约id', '用户id'])['经度'].shift(1)
    df['漂移_前时间戳'] = df.groupby(['合约id', '用户id'])['时间戳'].shift(1)
    from utils.geo import haversine
    df['漂移_距离差_km'] = haversine(df['漂移_前纬度'], df['漂移_前经度'], df['纬度'], df['经度'], 'km')
    df['漂移_时间差_h'] = (df['时间戳'] - df['漂移_前时间戳']) / 3600
    df['漂移_瞬时速度_kmh'] = np.where(df['漂移_时间差_h'] > 0, df['漂移_距离差_km'] / df['漂移_时间差_h'], 0)
    df = df[(df['漂移_瞬时速度_kmh'] <= DRIFT_SPEED_THRESHOLD) | (df['漂移_瞬时速度_kmh'].isna())]
    df = df.drop(columns=['漂移_前纬度', '漂移_前经度', '漂移_前时间戳', '漂移_距离差_km', '漂移_时间差_h', '漂移_瞬时速度_kmh'], errors='ignore')
    
    df = df.sort_values(['合约id', '用户id', '时间戳']).reset_index(drop=True)
    
    return df


# ==============================================================================
# 合约日级指标计算（严格按照旧脚本逻辑）
# ==============================================================================
def calc_contract_metrics(df_sorted):
    """计算合约日级指标（完全对齐旧脚本）"""
    group_keys = ['合约id', '用户id']
    results = []
    groups = list(df_sorted.groupby(group_keys))
    
    for idx, ((contract_id, user_id), group) in enumerate(tqdm(groups, desc="合约计算进度", mininterval=2.0)):
        user_id = str(user_id) if pd.notna(user_id) else f"未知用户_{idx}"
        group = group.sort_values('时间戳').reset_index(drop=True)
        n = len(group)
        stat_date = group['统计日期'].iloc[0] if not group.empty else pd.Timestamp.now().strftime('%Y-%m-%d')
        
        res = {
            '统计日期': stat_date, '合约id': contract_id, '用户id': user_id,
            '核心活动省份': '', '核心活动城市': '', '核心活动区县': '',
            '行驶距离': 0.0, '最大出行距离': 0.0,
            'R90日常活动半径': 0.0, 'R95核心活动半径': 0.0, '凸包覆盖面积': 0.0,
            '中心纬度': np.nan, '中心经度': np.nan,
            '骑行次数': 0, '单次平均骑行时长(分钟)': 0.0, '骑行总耗时(小时)': 0.0,
            '最大速度': 0.0, '最大速度时间': pd.NA, '平均骑行速度': 0.0,
            '最大电流': 0.0, '最大电流时间': pd.NA, '骑行放电平均电流': 0.0,
            '全周期平均电流': 0.0, '电流标准差': 0.0, '电流变异系数': 0.0,
            '高峰电流标准差': 0.0, '高峰电流变异系数': 0.0,
            '平峰电流标准差': 0.0, '平峰电流变异系数': 0.0,
            '单行程最大电流标准差': 0.0, '单行程最大电流变异系数': 0.0,
            '总放电时长(小时)': 0.0, 
            '怠速放电时长(小时)': 0.0,
            '骑行-放电时长比': 0.0,
            '电流异常用户': '否',
            '最大温度': 0.0, '最大温度时间': pd.NA, '平均温度': 0.0,
            '客户形态': '数据不足', '包月友好评分': 0, '用户等级': '无效',
            '高峰骑行占比': 0.0, '单日骑行次数': 0, '当日是否出勤': 0,
            '电流>60A次数': 0, '电流>80A次数': 0,
            '超100A连续次数': 0, '超100A累计时长_h': 0.0,
            '总用电量(kWh)': 0.0, '百公里电耗(kWh)': 0.0, '换电次数': 0,
            '平均骑行SOC': 0.0, '最低SOC': 100.0, 'SOC低于20%时长占比': 0.0, 'SOC低于10%时长占比': 0.0,
            '午间高峰长时骑行次数': 0, '晚间高峰长时骑行次数': 0, '平峰长时骑行次数': 0, '夜间长时骑行次数': 0,
            '午间高峰骑行里程': 0.0, '晚间高峰骑行里程': 0.0, '平峰骑行里程': 0.0, '夜间骑行里程': 0.0,
            '工作时长覆盖(小时)': 0.0, '有效GPS点数': n,
            # 换电时段
            '换电总次数': 0, '平峰换电次数': 0, '深夜换电次数': 0, '高峰换电次数': 0,
            '平峰换电占比': 0.0, '深夜换电占比': 0.0, '偏好换电时段': '无换电',
            # SOC 换电画像
            '取电时平均SOC': np.nan, '还电时平均SOC': np.nan, '单次换电平均SOC消耗': np.nan,
            # 速度分位
            'P50骑行速度_kmh': 0.0, 'P90骑行速度_kmh': 0.0,
            '夜间骑行均速_kmh': 0.0, '高速骑行点数(>40kmh)': 0,
            # 骑行时刻
            '最早骑行时刻_h': -1, '最晚骑行时刻_h': -1, '主要骑行时段': '无数据',
        }

        total_riding_hours = 0.0
        idle_discharge_hours = 0.0
        total_discharge_hours = 0.0

        if n >= 2:
            # -------------------------- 1. 位移与骑行状态判定 --------------------------
            group['前纬度'] = group['纬度'].shift(1)
            group['前经度'] = group['经度'].shift(1)
            group['前时间戳'] = group['时间戳'].shift(1)
            group['相邻距离_km'] = haversine(group['前纬度'], group['前经度'], group['纬度'], group['经度'], 'km')
            group['相邻距离_m'] = group['相邻距离_km'] * 1000
            group['时间间隔_min'] = (group['时间戳'] - group['前时间戳']) / 60
            group['动态距离阈值_km'] = (MAX_SPEED_KMH * group['时间间隔_min'] / 60) * 1.1
            group['有效位移'] = (group['相邻距离_km'] <= group['动态距离阈值_km']) & (group['相邻距离_km'] > 0)

            # 骑行判定逻辑
            main_ride_condition = (
                (group['相邻距离_m'] >= MIN_VALID_DISPLACEMENT_M) &
                (group['速度'].fillna(0) > 0) &
                (group['速度'].fillna(0) <= MAX_VALID_SPEED) &
                (group['电流'] > -20)
            )
            group['初始骑行状态'] = main_ride_condition.rolling(
                window=STOP_SMOOTH_WINDOW, 
                min_periods=1
            ).max().fillna(0).astype(int)

            assist_ride_condition = (
                (group['相邻距离_m'] < MIN_VALID_DISPLACEMENT_M) &
                (group['速度'].fillna(0) >= 0) &
                (group['速度'].fillna(0) <= 5) &
                (group['电流_骑行判定用'] > 0) &
                (group['初始骑行状态'].shift(1).fillna(0) == 1)
            )
            group['核心有效'] = main_ride_condition | assist_ride_condition
            group['骑行状态'] = group['核心有效'].rolling(
                window=STOP_SMOOTH_WINDOW, 
                min_periods=1,
                center=True
            ).max().fillna(0).astype(int)
            group['有效里程_km'] = group['相邻距离_km'].where(
                (group['骑行状态'] == 1) & group['有效位移'] & main_ride_condition, 
                0.0
            )
            group = group.drop(columns=['初始骑行状态'])

            total_distance = group['有效里程_km'].sum() if DISTANCE_UNIT_KM else group['有效里程_km'].sum() * 1000
            res['行驶距离'] = round(total_distance, 2)

            # -------------------------- 2. 空间活动指标计算 --------------------------
            riding_points = group.loc[group['骑行状态'] == 1, ['纬度', '经度']].values
            center_lat, center_lon = get_real_center(group)
            res['中心纬度'] = round(center_lat, 6) if pd.notna(center_lat) else np.nan
            res['中心经度'] = round(center_lon, 6) if pd.notna(center_lon) else np.nan
            
            max_trip_distance = r90_radius = r95_radius = hull_area = 0.0
            if len(riding_points) >= 3:
                try:
                    CONVEXHULL_MAX_POINTS = 200
                    if len(riding_points) > CONVEXHULL_MAX_POINTS:
                        np.random.seed(42)
                        sample_indices = np.random.choice(len(riding_points), CONVEXHULL_MAX_POINTS, replace=False)
                        hull_input = riding_points[sample_indices]
                    else:
                        hull_input = riding_points
                    
                    lat_range = hull_input[:, 0].max() - hull_input[:, 0].min()
                    lon_range = hull_input[:, 1].max() - hull_input[:, 1].min()
                    aspect_ratio = max(lat_range, lon_range) / (min(lat_range, lon_range) + 1e-10)
                    
                    if aspect_ratio > 1000:
                        hull_area = 0.0
                        if len(hull_input) >= 2:
                            lat1 = hull_input[:, 0][:, np.newaxis]
                            lon1 = hull_input[:, 1][:, np.newaxis]
                            lat2 = hull_input[:, 0][np.newaxis, :]
                            lon2 = hull_input[:, 1][np.newaxis, :]
                            dlat = np.radians(lat2 - lat1)
                            dlon = np.radians(lon2 - lon1)
                            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                            dist_matrix = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                            upper_tri = np.triu(dist_matrix, k=1)
                            if upper_tri.size > 0:
                                max_trip_distance = np.max(upper_tri)
                        dist_to_center = haversine(center_lat, center_lon, riding_points[:,0], riding_points[:,1], 'km')
                        r90_radius = np.quantile(dist_to_center, RADIUS_QUANTILES[0])
                        r95_radius = np.quantile(dist_to_center, RADIUS_QUANTILES[1])
                    else:
                        hull = ConvexHull(hull_input)
                        hull_points = hull_input[hull.vertices]
                        if len(hull_points) >= 2:
                            lat1 = hull_points[:, 0][:, np.newaxis]
                            lon1 = hull_points[:, 1][:, np.newaxis]
                            lat2 = hull_points[:, 0][np.newaxis, :]
                            lon2 = hull_points[:, 1][np.newaxis, :]
                            dlat = np.radians(lat2 - lat1)
                            dlon = np.radians(lon2 - lon1)
                            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                            dist_matrix = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                            upper_tri = np.triu(dist_matrix, k=1)
                            if upper_tri.size > 0:
                                max_trip_distance = np.max(upper_tri)
                        dist_to_center = haversine(center_lat, center_lon, riding_points[:,0], riding_points[:,1], 'km')
                        r90_radius = np.quantile(dist_to_center, RADIUS_QUANTILES[0])
                        r95_radius = np.quantile(dist_to_center, RADIUS_QUANTILES[1])
                        lat_per_km = 1 / 111.0
                        lon_per_km = 1 / (111.0 * math.cos(math.radians(center_lat))) if pd.notna(center_lat) else 1 / 111.0
                        points_km = np.zeros_like(hull_points)
                        points_km[:,0] = (hull_points[:,0] - center_lat) / lat_per_km if pd.notna(center_lat) else 0
                        points_km[:,1] = (hull_points[:,1] - center_lon) / lon_per_km if pd.notna(center_lon) else 0
                        hull_km = ConvexHull(points_km)
                        hull_area = hull_km.area
                except:
                    hull_area = 0.0
            
            res['最大出行距离'] = round(max_trip_distance, 2)
            res['R90日常活动半径'] = round(r90_radius, 2)
            res['R95核心活动半径'] = round(r95_radius, 2)
            res['凸包覆盖面积'] = round(hull_area, 2)

            # -------------------------- 3. 行程拆分与骑行时长计算 --------------------------
            group['状态变化'] = group['骑行状态'].diff().fillna(0)
            group['断连标记'] = (group['时间间隔_min'] > MAX_TRIP_GAP_MIN).astype(int)
            group['行程ID'] = ((group['状态变化'] == 1) | (group['断连标记'] == 1)).cumsum()
            group.loc[group['骑行状态'] == 0, '行程ID'] = np.nan

            valid_trips = []
            noon_long_count = evening_long_count = offpeak_long_count = night_long_count = 0
            group['小时'] = (pd.to_datetime(group['时间戳'], unit='s', errors='coerce').dt.hour.fillna(0).astype(int))
            
            for trip_id, trip_group in group.groupby('行程ID', dropna=True):
                trip_duration_min = (trip_group['时间戳'].max() - trip_group['时间戳'].min()) / 60
                if trip_duration_min >= MIN_RIDING_DURATION_MIN:
                    valid_trips.append(trip_duration_min)
                    if trip_duration_min >= VALID_RIDE_HOUR_FOR_PATTERN * 60:
                        mid_idx = len(trip_group) // 2
                        trip_hour = trip_group['小时'].iloc[mid_idx] if mid_idx < len(trip_group) else trip_group['小时'].iloc[0]
                        if trip_hour in NOON_PEAK_HOURS: noon_long_count += 1
                        elif trip_hour in EVENING_PEAK_HOURS: evening_long_count += 1
                        elif trip_hour in NIGHT_HOURS: night_long_count += 1
                        else: offpeak_long_count += 1
            
            res['午间高峰长时骑行次数'] = noon_long_count
            res['晚间高峰长时骑行次数'] = evening_long_count
            res['平峰长时骑行次数'] = offpeak_long_count
            res['夜间长时骑行次数'] = night_long_count
            
            trip_count = len(valid_trips)
            total_riding_min = sum(valid_trips)
            avg_riding_min = round(total_riding_min / trip_count, 2) if trip_count > 0 else 0.0
            total_riding_hours = round(total_riding_min / 60, 2)
            
            res['骑行次数'] = trip_count
            res['单日骑行次数'] = trip_count
            res['单次平均骑行时长(分钟)'] = avg_riding_min
            res['骑行总耗时(小时)'] = total_riding_hours

            # -------------------------- 4. 分时段骑行里程统计 --------------------------
            group['是否午间高峰'] = group['小时'].isin(NOON_PEAK_HOURS)
            group['是否晚间高峰'] = group['小时'].isin(EVENING_PEAK_HOURS)
            group['是否夜间'] = group['小时'].isin(NIGHT_HOURS)
            group['是否平峰'] = group['小时'].isin(OFFPEAK_HOURS)
            
            noon_peak_miles = group[(group['骑行状态'] == 1) & group['是否午间高峰']]['有效里程_km'].sum()
            evening_peak_miles = group[(group['骑行状态'] == 1) & group['是否晚间高峰']]['有效里程_km'].sum()
            night_miles = group[(group['骑行状态'] == 1) & group['是否夜间']]['有效里程_km'].sum()
            offpeak_miles = group[(group['骑行状态'] == 1) & group['是否平峰']]['有效里程_km'].sum()
            
            res['午间高峰骑行里程'] = round(noon_peak_miles, 2)
            res['晚间高峰骑行里程'] = round(evening_peak_miles, 2)
            res['夜间骑行里程'] = round(night_miles, 2)
            res['平峰骑行里程'] = round(offpeak_miles, 2)

            # -------------------------- 5. 出勤与高峰占比计算 --------------------------
            total_riding_count = group[group['骑行状态'] == 1].shape[0]
            peak_riding_count = group[(group['骑行状态'] == 1) & (group['是否午间高峰'] | group['是否晚间高峰'])].shape[0]
            peak_riding_ratio = round(peak_riding_count / total_riding_count, 2) if total_riding_count > 0 else 0
            is_work_day = 1 if total_riding_hours >= 0.5 else 0
            res['高峰骑行占比'] = peak_riding_ratio
            res['当日是否出勤'] = is_work_day

            # -------------------------- 6. 速度指标计算 --------------------------
            riding_speed_data = group.loc[group['骑行状态'] == 1, ['速度', '时间戳']].dropna(subset=['速度'])
            max_speed = 0.0
            max_speed_time = pd.NA
            avg_riding_speed = 0.0
            if len(riding_speed_data) > 0:
                riding_speed_data = riding_speed_data[riding_speed_data['速度'] <= MAX_VALID_SPEED]
                if len(riding_speed_data) > 0:
                    max_speed = riding_speed_data['速度'].max()
                    max_speed_idx = riding_speed_data['速度'].idxmax()
                    max_speed_timestamp = riding_speed_data.loc[max_speed_idx, '时间戳']
                    max_speed_time = pd.to_datetime(max_speed_timestamp, unit='s', errors='coerce')
            if total_riding_hours > 0.01:
                avg_riding_speed = round(total_distance / total_riding_hours, 2)
            res['最大速度'] = round(max_speed, 2)
            res['最大速度时间'] = max_speed_time
            res['平均骑行速度'] = avg_riding_speed

            # 速度分位数 & 夜间速度
            speed_p50 = speed_p90 = night_avg_speed = 0.0
            high_speed_count = 0

            if len(riding_speed_data) > 0:
                speeds = riding_speed_data['速度'].values
                speed_p50 = round(np.percentile(speeds, 50), 2)
                speed_p90 = round(np.percentile(speeds, 90), 2)
                high_speed_count = int((speeds > 40).sum())

                riding_speed_data['小时_速度'] = pd.to_datetime(
                    riding_speed_data['时间戳'], unit='s', errors='coerce'
                ).dt.hour.fillna(0).astype(int)
                night_mask = riding_speed_data['小时_速度'].isin(NIGHT_HOURS)
                night_speeds = riding_speed_data.loc[night_mask, '速度']
                night_avg_speed = round(night_speeds.mean(), 2) if len(night_speeds) > 3 else 0.0

            res['P50骑行速度_kmh'] = speed_p50
            res['P90骑行速度_kmh'] = speed_p90
            res['夜间骑行均速_kmh'] = night_avg_speed
            res['高速骑行点数(>40kmh)'] = high_speed_count

            # -------------------------- 7. 放电时长计算 --------------------------
            current_col_for_discharge = '电流_放电统计用' if '电流_放电统计用' in group.columns else '电流'
            group['放电状态'] = (group[current_col_for_discharge] > 0).rolling(
                window=STOP_SMOOTH_WINDOW, 
                min_periods=1,
                center=True
            ).max().fillna(0).astype(int)
            
            total_discharge_hours = 0.0
            if '放电状态' in group.columns:
                group['放电块ID'] = (group['放电状态'] != group['放电状态'].shift()).cumsum()
                for block_id, block_group in group[group['放电状态'] == 1].groupby('放电块ID'):
                    if len(block_group) >= 1:
                        duration_h = len(block_group) * COLLECTION_CYCLE_MIN / 60
                        if duration_h >= 1/60:
                            total_discharge_hours += duration_h
            else:
                valid_discharge_mask = group[current_col_for_discharge] > 0
                total_discharge_hours = valid_discharge_mask.sum() * COLLECTION_CYCLE_MIN / 60
            
            total_discharge_hours = round(total_discharge_hours, 2)
            res['总放电时长(小时)'] = total_discharge_hours

            # -------------------------- 8. 怠速放电时长计算 --------------------------
            idle_discharge_hours = max(0.0, total_discharge_hours - total_riding_hours)
            res['怠速放电时长(小时)'] = round(idle_discharge_hours, 2)
            res['骑行-放电时长比'] = round(total_riding_hours / total_discharge_hours, 3) if total_discharge_hours > 0 else 0.0

            # -------------------------- 9. 电流指标计算 --------------------------
            current_col_for_ride = '电流_骑行判定用' if '电流_骑行判定用' in group.columns else '电流'
            riding_current_data = group.loc[group['骑行状态'] == 1, ['时间戳', current_col_for_ride]].dropna(subset=[current_col_for_ride])
            riding_current_data = riding_current_data.rename(columns={current_col_for_ride: '电流'})
            
            riding_current_data = riding_current_data[(riding_current_data['电流'] >= MIN_VALID_CURRENT) & (riding_current_data['电流'] <= MAX_VALID_CURRENT)]
            full_current_data = group[current_col_for_ride].fillna(0)
            
            max_current = 0.0
            max_current_time = pd.NA
            riding_avg_current = 0.0
            full_avg_current = 0.0
            current_std = 0.0
            current_cv = 0.0
            peak_current_std = 0.0
            peak_current_cv = 0.0
            offpeak_current_std = 0.0
            offpeak_current_cv = 0.0
            max_trip_current_std = 0.0
            max_trip_current_cv = 0.0
            is_current_abnormal = '否'

            if len(riding_current_data) > 0:
                max_current = riding_current_data['电流'].max()
                max_current_idx = riding_current_data['电流'].idxmax()
                max_current_timestamp = riding_current_data.loc[max_current_idx, '时间戳']
                max_current_time = pd.to_datetime(max_current_timestamp, unit='s', errors='coerce')
                riding_avg_current = riding_current_data['电流'].mean()
                current_std = riding_current_data['电流'].std() if len(riding_current_data) > 1 else 0.0
                current_cv = current_std / riding_avg_current if riding_avg_current > 0.1 else 0.0
                riding_current_data['小时'] = pd.to_datetime(riding_current_data['时间戳'], unit='s', errors='coerce').dt.hour.fillna(0).astype(int)
                riding_current_data['是否高峰'] = riding_current_data['小时'].isin(NOON_PEAK_HOURS + EVENING_PEAK_HOURS)
                peak_data = riding_current_data[riding_current_data['是否高峰']]
                if len(peak_data) > 1:
                    peak_avg = peak_data['电流'].mean()
                    peak_current_std = peak_data['电流'].std()
                    peak_current_cv = peak_current_std / peak_avg if peak_avg > 0.1 else 0.0
                offpeak_data = riding_current_data[~riding_current_data['是否高峰']]
                if len(offpeak_data) > 1:
                    offpeak_avg = offpeak_data['电流'].mean()
                    offpeak_current_std = offpeak_data['电流'].std()
                    offpeak_current_cv = offpeak_current_std / offpeak_avg if offpeak_avg > 0.1 else 0.0
                
                if '行程ID' in group.columns:
                    riding_current_data = riding_current_data.join(group[['行程ID']], on='时间戳', how='left')
                    trip_std_list = []
                    for trip_id, trip_group in riding_current_data.groupby('行程ID', dropna=True):
                        if len(trip_group) > 1:
                            trip_std = trip_group['电流'].std()
                            trip_avg = trip_group['电流'].mean()
                            trip_cv = trip_std / trip_avg if trip_avg > 0.1 else 0.0
                            trip_std_list.append((trip_std, trip_cv))
                    if trip_std_list:
                        max_trip_current_std, max_trip_current_cv = max(trip_std_list, key=lambda x: x[0])

            full_avg_current = full_current_data.mean()
            if (max_current >= ABNORMAL_MAX_CURRENT) or (current_cv >= ABNORMAL_CV):
                is_current_abnormal = '是'
            if (riding_avg_current >= ABNORMAL_AVG_CURRENT) and (total_discharge_hours >= ABNORMAL_MIN_DISCHARGE_HOUR):
                is_current_abnormal = '是'
            
            res['最大电流'] = round(max_current, 2)
            res['最大电流时间'] = max_current_time
            res['骑行放电平均电流'] = round(riding_avg_current, 2)
            res['全周期平均电流'] = round(full_avg_current, 2)
            res['电流标准差'] = round(current_std, 2)
            res['电流变异系数'] = round(current_cv, 3)
            res['高峰电流标准差'] = round(peak_current_std, 2)
            res['高峰电流变异系数'] = round(peak_current_cv, 3)
            res['平峰电流标准差'] = round(offpeak_current_std, 2)
            res['平峰电流变异系数'] = round(offpeak_current_cv, 3)
            res['单行程最大电流标准差'] = round(max_trip_current_std, 2)
            res['单行程最大电流变异系数'] = round(max_trip_current_cv, 3)
            res['电流异常用户'] = is_current_abnormal

            if not riding_current_data.empty:
                res['电流>60A次数'] = (riding_current_data['电流'] > NORMAL_CURRENT_THRESHOLD).sum()
                res['电流>80A次数'] = ((riding_current_data['电流'] > HIGH_CURRENT_THRESHOLD) & (riding_current_data['电流'] < OVER_CURRENT_THRESHOLD)).sum()
                riding_current_data['电流≥100A'] = (riding_current_data['电流'] >= OVER_CURRENT_THRESHOLD).astype(int)
                riding_current_data['连续超100A'] = riding_current_data['电流≥100A'].rolling(window=OVER_CURRENT_CONTINUOUS, min_periods=OVER_CURRENT_CONTINUOUS).sum()
                res['超100A连续次数'] = (riding_current_data['连续超100A'] >= OVER_CURRENT_CONTINUOUS).sum()
                res['超100A累计时长_h'] = round((riding_current_data['电流≥100A'].sum() * COLLECTION_CYCLE_MIN) / 60, 2)

            # -------------------------- 10. 温度指标计算 --------------------------
            temp_data = group[['温度', '时间戳']].dropna(subset=['温度'])
            max_temp = 0.0
            max_temp_time = pd.NA
            avg_temp = 0.0
            if len(temp_data) > 0:
                max_temp = temp_data['温度'].max()
                max_temp_idx = temp_data['温度'].idxmax()
                max_temp_timestamp = temp_data.loc[max_temp_idx, '时间戳']
                max_temp_time = pd.to_datetime(max_temp_timestamp, unit='s', errors='coerce')
                avg_temp = temp_data['温度'].mean()
            res['最大温度'] = round(max_temp, 2)
            res['最大温度时间'] = max_temp_time
            res['平均温度'] = round(avg_temp, 2)

            # -------------------------- 11. 电池能耗与换电次数计算 --------------------------
            total_energy_used = 0.0
            battery_change_count = 0
            energy_per_100km = 0.0
            riding_soc_data = pd.Series(dtype='float64')

            if '电池id' in group.columns:
                group['电池id'] = group['电池id'].fillna('未知电池').astype(str)
                group['电池切换标记'] = (group['电池id'] != group['电池id'].shift(1)).astype(int)
                battery_change_count = group['电池切换标记'].sum()
                if group['电池切换标记'].iloc[0] == 1:
                    battery_change_count -= 1
                battery_change_count = max(0, battery_change_count)
            res['换电次数'] = battery_change_count

            # 换电时段统计
            OFFPEAK_SWAP_HOURS = list(range(6, 11)) + list(range(14, 17))
            NIGHT_SWAP_HOURS   = list(range(20, 24)) + list(range(0, 6))

            swap_hours = []
            if '电池id' in group.columns:
                swap_mask = group['电池切换标记'] == 1
                if swap_mask.sum() > 0:
                    swap_times = pd.to_datetime(
                        group.loc[swap_mask, '时间戳'], unit='s', errors='coerce'
                    )
                    swap_hours = swap_times.dt.hour.dropna().tolist()

            total_swaps = len(swap_hours)
            offpeak_swaps = sum(1 for h in swap_hours if h in OFFPEAK_SWAP_HOURS)
            night_swaps   = sum(1 for h in swap_hours if h in NIGHT_SWAP_HOURS)
            peak_swaps    = total_swaps - offpeak_swaps - night_swaps

            res['换电总次数'] = total_swaps
            res['平峰换电次数'] = offpeak_swaps
            res['深夜换电次数'] = night_swaps
            res['高峰换电次数'] = peak_swaps
            res['平峰换电占比'] = round(offpeak_swaps / total_swaps, 3) if total_swaps > 0 else 0.0
            res['深夜换电占比'] = round(night_swaps / total_swaps, 3) if total_swaps > 0 else 0.0

            if swap_hours:
                from collections import Counter
                hour_dist = Counter(swap_hours)
                peak_hour = hour_dist.most_common(1)[0][0]
                if peak_hour in OFFPEAK_SWAP_HOURS:
                    res['偏好换电时段'] = '平峰'
                elif peak_hour in NIGHT_SWAP_HOURS:
                    res['偏好换电时段'] = '深夜'
                else:
                    res['偏好换电时段'] = '高峰'

            # 取电/还电 SOC 记录
            start_socs, end_socs = [], []

            if '电池度数' in group.columns and '电池id' in group.columns and '电池SOC' in group.columns:
                for battery_segment, batt_group in group.groupby(group['电池切换标记'].cumsum()):
                    batt_group = batt_group.dropna(subset=['电池度数'])
                    if len(batt_group) < 2: continue
                    start_energy = batt_group['电池度数'].iloc[0]
                    end_energy = batt_group['电池度数'].iloc[-1]
                    if start_energy > end_energy:
                        total_energy_used += (start_energy - end_energy)

                    soc_seg = batt_group['电池SOC'].dropna()
                    if len(soc_seg) >= 2:
                        start_socs.append(soc_seg.iloc[0])
                        end_socs.append(soc_seg.iloc[-1])

            else:
                if '电池度数' in group.columns and '电池id' in group.columns:
                    for battery_segment, batt_group in group.groupby(group['电池切换标记'].cumsum()):
                        batt_group = batt_group.dropna(subset=['电池度数'])
                        if len(batt_group) < 2: continue
                        start_energy = batt_group['电池度数'].iloc[0]
                        end_energy = batt_group['电池度数'].iloc[-1]
                        if start_energy > end_energy:
                            total_energy_used += (start_energy - end_energy)

            res['取电时平均SOC'] = round(np.mean(start_socs), 1) if start_socs else np.nan
            res['还电时平均SOC'] = round(np.mean(end_socs), 1) if end_socs else np.nan
            res['单次换电平均SOC消耗'] = round(
                np.mean(start_socs) - np.mean(end_socs), 1
            ) if start_socs and end_socs else np.nan

            res['总用电量(kWh)'] = round(total_energy_used, 2)

            if total_distance > 1.0:
                energy_per_100km = (total_energy_used / total_distance) * 100
            res['百公里电耗(kWh)'] = round(energy_per_100km, 2)

            # -------------------------- 12. SOC电池健康指标计算 --------------------------
            soc_data_valid = False
            if '电池SOC' in group.columns:
                riding_soc_data = group.loc[group['骑行状态'] == 1, '电池SOC'].dropna()
                if len(riding_soc_data) > 0:
                    soc_data_valid = True
                    res['平均骑行SOC'] = round(riding_soc_data.mean(), 1)
                    res['最低SOC'] = round(riding_soc_data.min(), 1)
                    total_riding_points = len(riding_soc_data)
                    res['SOC低于20%时长占比'] = round((riding_soc_data < SOC_LOW_WARNING).sum() / total_riding_points, 2)
                    res['SOC低于10%时长占比'] = round((riding_soc_data < SOC_CRITICAL).sum() / total_riding_points, 2)

            # -------------------------- 13. 工作时长覆盖计算 --------------------------
            riding_time_data = group.loc[group['骑行状态'] == 1, '时间戳']
            if len(riding_time_data) > 0:
                work_span_hours = (riding_time_data.max() - riding_time_data.min()) / 3600
                res['工作时长覆盖(小时)'] = round(work_span_hours, 1)

                # 骑行时刻分布
                riding_hours = pd.to_datetime(
                    group.loc[group['骑行状态'] == 1, '时间戳'], unit='s', errors='coerce'
                ).dt.hour.dropna()
                if len(riding_hours) > 0:
                    res['最早骑行时刻_h'] = int(riding_hours.min())
                    res['最晚骑行时刻_h'] = int(riding_hours.max())

                    main_hour = riding_hours.mode().iloc[0] if len(riding_hours.mode()) > 0 else -1
                    if main_hour in NOON_PEAK_HOURS:
                        res['主要骑行时段'] = '午间高峰'
                    elif main_hour in EVENING_PEAK_HOURS:
                        res['主要骑行时段'] = '晚间高峰'
                    elif main_hour in NIGHT_HOURS:
                        res['主要骑行时段'] = '夜间'
                    else:
                        res['主要骑行时段'] = '平峰'

            # -------------------------- 13.5. 数据有效性标记 --------------------------
            energy_data_valid = (energy_per_100km > 0)

            # -------------------------- 14. 客户形态判定 --------------------------
            has_real_ride = (total_riding_hours >= VALID_RIDE_MIN_HOUR) and (total_distance > 0)
            no_real_ride = not has_real_ride
            valid_gps_enough = (n >= STORAGE_MIN_VALID_GPS_POINTS)
            avg_speed = total_distance / total_riding_hours if total_riding_hours > 0.01 else 0

            very_short_distance = (total_distance < STORAGE_MAX_DISTANCE_KM)
            very_low_speed = (avg_speed < STORAGE_MAX_AVG_SPEED_KMH)

            discharge_much_longer = (
                (idle_discharge_hours > total_riding_hours * STORAGE_DISCHARGE_RATIO_THRESHOLD) & 
                (idle_discharge_hours >= STORAGE_MIN_DISCHARGE_HOUR) &
                (total_riding_hours < 0.5)
            )
            ride_ratio_low = (total_riding_hours / total_discharge_hours < STORAGE_RIDE_DURATION_RATIO) if total_discharge_hours > 0 else True

            is_storage_scene = no_real_ride and valid_gps_enough and very_short_distance and very_low_speed and discharge_much_longer and ride_ratio_low

            if max_speed >= MODIFY_SPEED_THRESHOLD and riding_avg_current >= MODIFY_CURRENT_THRESHOLD and has_real_ride:
                customer_type = "改装/超速车"
            elif is_storage_scene:
                customer_type = "地摊/储能"
            elif has_real_ride and total_riding_hours >= 4.0 and peak_riding_ratio >= 0.3:
                customer_type = "专送骑手"
            elif has_real_ride and total_riding_hours >= 2.0 and peak_riding_ratio >= 0.15:
                customer_type = "众包骑手"
            elif has_real_ride and total_distance <= 30 and total_riding_hours <= 2.0:
                customer_type = "标准骑手"
            elif has_real_ride:
                customer_type = "普通骑手"
            else:
                customer_type = "数据不足"
            res['客户形态'] = customer_type

            # -------------------------- 15. 包月友好评分体系计算 --------------------------
            score = _calc_monthly_score_v2(
                max_speed=max_speed,
                max_temp=max_temp,
                riding_avg_current=riding_avg_current,
                current_60a_count=res['电流>60A次数'],
                current_80a_count=res['电流>80A次数'],
                over100a_cont=res['超100A连续次数'],
                max_trip_current_cv=max_trip_current_cv,
                current_cv=current_cv,
                avg_riding_speed=avg_riding_speed,
                soc_data_valid=soc_data_valid,
                has_real_ride=has_real_ride,
                soc_below_10_ratio=res['SOC低于10%时长占比'],
                soc_below_20_ratio=res['SOC低于20%时长占比'],
                min_soc=res['最低SOC'],
                avg_soc=res['平均骑行SOC'],
                energy_data_valid=energy_data_valid,
                energy_per_100km=energy_per_100km,
                battery_change_count=battery_change_count,
                daily_distance=total_distance,
                battery_change_cnt=battery_change_count
            )
            if customer_type == "地摊/储能":
                score = min(100, score + 20)
            elif customer_type == "外卖高强度车":
                score -= 15
            elif customer_type == "改装/超速车":
                score -= 30
            score = max(0, min(100, round(score)))
            res['包月友好评分'] = score

            # -------------------------- 16. 用户等级判定 --------------------------
            user_level = _determine_user_level_v2(
                over100a_cont=res['超100A连续次数'],
                over100a_hours=res['超100A累计时长_h'],
                current_80a_count=0,
                soc_below_10_ratio=res['SOC低于10%时长占比'],
                min_soc=res['最低SOC'],
                current_60a_count=res['电流>60A次数'],
                energy_per_100km=energy_per_100km,
                soc_below_20_ratio=res['SOC低于20%时长占比'],
                score=score,
                riding_avg_current=riding_avg_current,
                soc_data_valid=soc_data_valid,
                energy_data_valid=energy_data_valid
            )
            res['用户等级'] = user_level

            # ==================================================================================
            # 🆕 【新增】单合约单日 动态说明生成
            # ==================================================================================
            # 1. 用户等级说明
            level_desc = ""
            current_level = res['用户等级']
            if current_level == "暴力用户（超量放电/电池滥用）":
                reasons = []
                if res['超100A连续次数'] >= VIOLENT_CURRENT_TIMES:
                    reasons.append(f"超100A连续放电{res['超100A连续次数']}次，触发阈值")
                if res['超100A累计时长_h'] >= OVER_CURRENT_MIN_HOUR:
                    reasons.append(f"超100A累计放电时长{res['超100A累计时长_h']}小时，触发阈值")
                if res['电流>60A次数'] >= VIOLENT_CURRENT_TIMES:
                    reasons.append(f"电流超60A共{res['电流>60A次数']}次，峰值达{res['最大电流']}A")
                if soc_data_valid and res['SOC低于10%时长占比'] >= 0.5 and res['最低SOC'] <= 5:
                    reasons.append(f"深度亏电严重（最低SOC{res['最低SOC']}%，SOC低于10%时长占比{res['SOC低于10%时长占比']*100:.0f}%）")
                level_desc = " | ".join(reasons) if reasons else "存在严重损害电池的行为"
            
            elif current_level == "高损耗用户":
                reasons = []
                if res['电流>60A次数'] >= HIGH_LOSS_CURRENT_TIMES:
                    reasons.append(f"中高电流使用频繁（电流超60A共{res['电流>60A次数']}次）")
                if energy_data_valid and energy_per_100km >= EXTREME_ENERGY_THRESHOLD:
                    reasons.append(f"能耗极高（百公里电耗{energy_per_100km}kWh）")
                if soc_data_valid and res['SOC低于20%时长占比'] >= 0.5:
                    reasons.append(f"长期低电量运行（SOC低于20%时长占比{res['SOC低于20%时长占比']*100:.0f}%）")
                if not reasons:
                    reasons.append(f"包月友好分较低（{score}分），电池损耗速度高于平均水平")
                level_desc = " | ".join(reasons)
            
            elif current_level == "优质用户":
                level_desc = f"电池使用习惯优秀（包月友好分{score}分），电流、速度、SOC均保持在健康区间"
            elif current_level == "良好用户":
                level_desc = f"电池使用习惯较好（包月友好分{score}分），整体负载可控"
            elif current_level == "普通用户":
                level_desc = f"电池使用行为一般（包月友好分{score}分），无明显过激使用行为"
            else:
                level_desc = "数据不足或无有效骑行数据，无法判定"
            res['用户等级说明'] = level_desc

            # 2. 客户形态说明
            type_desc = ""
            current_type = res['客户形态']
            if current_type == "改装/超速车":
                type_desc = f"行驶特征异常（最高速度{max_speed}km/h，平均骑行电流{riding_avg_current}A），远超普通两轮车水平，存在改装或超速嫌疑"
            elif current_type == "地摊/储能":
                type_desc = f"非移动用电特征明显（当日骑行{total_riding_hours}小时，怠速放电{idle_discharge_hours}小时），放电以静止状态为主，疑似地摊供电或储能场景"
            elif current_type == "专送骑手":
                type_desc = f"工作特征显著（当日骑行{total_riding_hours:.1f}小时，高峰骑行占比{peak_riding_ratio*100:.0f}%），工作时长稳定且午晚高峰高度活跃，符合专送骑手画像"
            elif current_type == "众包骑手":
                type_desc = f"具有兼职骑手特征（当日骑行{total_riding_hours:.1f}小时，高峰骑行占比{peak_riding_ratio*100:.0f}%），高峰时段有一定活跃度"
            elif current_type == "标准骑手":
                type_desc = f"骑行行为规律（当日行驶里程{total_distance}km，骑行时长{total_riding_hours:.1f}小时），属于标准日常使用场景"
            elif current_type == "普通骑手":
                type_desc = f"有常规骑行行为（当日行驶里程{total_distance}km，骑行时长{total_riding_hours:.1f}小时），不符合特定骑手标签特征"
            else:
                type_desc = "无有效骑行数据或数据量不足，无法判定具体使用场景"
            res['客户形态说明'] = type_desc
            # ==================================================================================

        results.append(res)

    return pd.DataFrame(results)


# ==============================================================================
# 主处理函数
# ==============================================================================
def process_fact_layer(target_date: Optional[str] = None, base_path: Optional[str] = None):
    """处理fact层数据"""
    print("\n" + "="*80)
    print("L1 Fact Layer - 客观事实层")
    print("="*80)
    
    if base_path is None:
        base_path = BASE_EXPORT_PATH
    
    raw_dir = os.path.join(base_path, EXPORT_PATH_RAW)
    output_dir = os.path.join(base_path, EXPORT_PATH_FACT_DAILY)
    os.makedirs(output_dir, exist_ok=True)
    
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.parquet')])
    if not raw_files:
        print("❌ 未找到原始数据文件")
        return {}
    
    processed = {}
    
    for filename in raw_files:
        date_str = filename.replace('.parquet', '')
        if target_date and date_str != target_date:
            continue
        
        output_path = os.path.join(output_dir, f"{date_str}.parquet")
        if os.path.exists(output_path):
            print(f"⏭️  L1 已存在，跳过: {output_path}")
            processed[date_str] = output_path
            continue
        
        print(f"\n📊 处理日期: {date_str}")
        raw_path = os.path.join(raw_dir, filename)
        
        try:
            df_raw = pd.read_parquet(raw_path)
        except Exception as e:
            print(f"⚠️ 读取原始数据失败 {raw_path}: {e}")
            continue
        
        print(f"   ✅ 加载完成：{len(df_raw):,} 行数据")
        
        # 数据预处理
        df_clean = preprocess_raw_data(df_raw)
        print(f"   ✅ 清洗完成：{len(df_clean):,} 行有效数据")
        
        # 计算合约指标
        df_result = calc_contract_metrics(df_clean)
        
        if df_result.empty:
            print("   ❌ 无有效合约指标")
            continue
        
        # GPS空间匹配
        print("🗺️  执行GPS空间匹配...")
        region_info = batch_gps_to_region(df_result, lat_col='中心纬度', lon_col='中心经度')
        for col in ['核心活动省份', '核心活动城市', '核心活动区县']:
            if col in region_info.columns:
                df_result[col] = region_info[col]
        
        df_result.to_parquet(output_path, engine='pyarrow', compression='snappy')
        print(f"   ✅ L1 保存: {output_path} ({len(df_result):,} 个合约日记录)")
        processed[date_str] = output_path
    
    print(f"\n✅ L1 Fact Layer 完成，处理 {len(processed)} 个日期")
    return processed


if __name__ == "__main__":
    process_fact_layer(target_date=None)
