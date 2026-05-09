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

from config.config import BASE_EXPORT_PATH, EXPORT_PATH_FACT_DAILY, EXPORT_PATH_BATTERY_STATUS_DAILY, EXPORT_FILE_BATTERY_CELL_VOLTAGE
from utils.geo import batch_gps_to_region, haversine
from score_common import (
    calc_monthly_score_v2, determine_user_level_v2,
    VIOLENT_CURRENT_TIMES, HIGH_LOSS_CURRENT_TIMES, OVER_CURRENT_MIN_HOUR,
    EXTREME_ENERGY_THRESHOLD,
)

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

# 用户等级判定阈值 & 包月友好评分扣分阈值
# 统一由 score_common.py 管理，fact_layer 通过导入函数间接使用

# ==============================================================================
# 辅助函数
# ==============================================================================

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
    df['漂移_距离差_km'] = haversine(df['漂移_前纬度'], df['漂移_前经度'], df['纬度'], df['经度'], 'km')
    df['漂移_时间差_h'] = (df['时间戳'] - df['漂移_前时间戳']) / 3600
    df['漂移_瞬时速度_kmh'] = np.where(df['漂移_时间差_h'] > 0, df['漂移_距离差_km'] / df['漂移_时间差_h'], 0)
    df = df[(df['漂移_瞬时速度_kmh'] <= DRIFT_SPEED_THRESHOLD) | (df['漂移_瞬时速度_kmh'].isna())]
    df = df.drop(columns=['漂移_前纬度', '漂移_前经度', '漂移_前时间戳', '漂移_距离差_km', '漂移_时间差_h', '漂移_瞬时速度_kmh'], errors='ignore')

    # 过滤离线状态数据
    if '是否在线' in df.columns:
        before = len(df)
        df = df[df['是否在线'] != 0]
        df = df[df['是否在线'] != '否']
        df = df[df['是否在线'] != False]
        removed = before - len(df)
        if removed > 0:
            print(f"   🧹 离线数据过滤：移除 {removed:,} 条")

    df = df.sort_values(['合约id', '用户id', '时间戳']).reset_index(drop=True)
    
    return df


# ==============================================================================
# 电池电压映射表加载
# ==============================================================================
def load_battery_voltage_map(battery_csv_path: str) -> dict:
    """加载电池id→标准电压(V)映射表"""
    if not os.path.exists(battery_csv_path):
        print(f"⚠️ 电池电压表不存在: {battery_csv_path}，将使用默认电压60V")
        return {}
    df = pd.read_csv(battery_csv_path)
    df['电池id'] = pd.to_numeric(df['电池id'], errors='coerce')
    df['标准电压'] = pd.to_numeric(df['标准电压'], errors='coerce')
    valid = df[(df['标准电压'] > 0) & df['电池id'].notna()]
    voltage_map = dict(zip(valid['电池id'].astype(int), valid['标准电压']))
    print(f"🔋 电池电压映射表加载完成：{len(voltage_map):,} 条有效记录")
    return voltage_map


# ==============================================================================
# 合约日级指标计算（严格按照旧脚本逻辑）
# ==============================================================================
def calc_contract_metrics(df_sorted, battery_voltage_map: dict = None):
    """计算合约日级指标（完全对齐旧脚本）"""
    if battery_voltage_map is None:
        battery_voltage_map = {}
    DEFAULT_VOLTAGE = 60.0
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
            # 原始维度字段备份
            '代理id': '', '电池id': '',
            '使用电池数': 0,
            # 时间字段
            '最早记录时间戳': 0, '最晚记录时间戳': 0, '记录时长_小时': 0.0,
            # 在线状态
            '在线率': 0.0, '在线时长_小时': 0.0,
            # 空间活动指标
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

            # -------------------------- 0. 原始维度字段统计（备份用）--------------------------
            res['最早记录时间戳'] = int(group['时间戳'].iloc[0]) if len(group) > 0 else 0
            res['最晚记录时间戳'] = int(group['时间戳'].iloc[-1]) if len(group) > 0 else 0
            time_span = (res['最晚记录时间戳'] - res['最早记录时间戳']) / 3600
            res['记录时长_小时'] = round(max(time_span, 0), 2)
            
            # 维度字段：取主要值（出现频率最高）
            if '代理id' in group.columns:
                mode_val = group['代理id'].mode()
                res['代理id'] = str(mode_val.iloc[0]) if len(mode_val) > 0 else ''
            if '电池id' in group.columns:
                mode_val = group['电池id'].mode()
                res['电池id'] = str(mode_val.iloc[0]) if len(mode_val) > 0 else ''
                res['使用电池数'] = group['电池id'].nunique()
            
            # 总用电量（SOC差法：标准电压 × 满容量 × SOC差）
            # 公式：E_kWh = Σ 标准电压_V × (容量/最早SOC) / 1000 × (最早SOC - 最晚SOC) / 100
            # 按电池id分组，每段取最早和最晚SOC计算耗电量
            if '电池id' in group.columns and '容量' in group.columns and '电池SOC' in group.columns and '时间戳' in group.columns:
                total_energy_kwh = 0.0
                group['容量_num'] = pd.to_numeric(group['容量'], errors='coerce')
                group['SOC_num'] = pd.to_numeric(group['电池SOC'], errors='coerce')
                group['时间戳_num'] = pd.to_numeric(group['时间戳'], errors='coerce')

                for bid, bgroup in group.groupby('电池id'):
                    bgroup = bgroup.sort_values('时间戳_num').reset_index(drop=True)
                    if len(bgroup) < 2:
                        continue

                    bid_int = int(bid) if pd.notna(bid) else -1
                    std_voltage = battery_voltage_map.get(bid_int, DEFAULT_VOLTAGE)

                    first_row = bgroup.iloc[0]
                    last_row = bgroup.iloc[-1]

                    first_cap = first_row['容量_num']
                    first_soc = first_row['SOC_num']
                    last_soc = last_row['SOC_num']

                    if pd.isna(first_cap) or pd.isna(first_soc) or pd.isna(last_soc):
                        continue
                    if first_soc <= 0:
                        continue

                    full_capacity_ah = first_cap / first_soc
                    soc_diff = first_soc - last_soc

                    if soc_diff > 0:
                        total_energy_kwh += std_voltage * full_capacity_ah * soc_diff / 100.0 / 1000.0

                group = group.drop(columns=['容量_num', 'SOC_num', '时间戳_num'], errors='ignore')
                res['总用电量(kWh)'] = round(total_energy_kwh, 3)
            
            # 在线状态统计
            if '是否在线' in group.columns:
                online_count = (group['是否在线'] == 1).sum() | (group['是否在线'] == '是').sum() | (group['是否在线'] == True).sum()
                res['在线率'] = round(online_count / n, 4) if n > 0 else 0
                # 在线时长估算（基于采样间隔）
                if n >= 2 and res['最晚记录时间戳'] > res['最早记录时间戳']:
                    res['在线时长_小时'] = round(online_count / n * time_span, 2)

            total_distance = group['有效里程_km'].sum() if DISTANCE_UNIT_KM else group['有效里程_km'].sum() * 1000
            res['行驶距离'] = round(total_distance, 2)
            
            # 百公里电耗计算（必须在行驶距离赋值之后）
            if res.get('总用电量(kWh)', 0) > 0 and res['行驶距离'] > 0:
                res['百公里电耗(kWh)'] = round(res['总用电量(kWh)'] / (res['行驶距离'] / 100), 3)

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
            riding_speed_data = group.loc[group['有效里程_km'] > 0, ['速度', '时间戳']].dropna(subset=['速度'])
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

            # -------------------------- 11. 换电次数与SOC计算 --------------------------
            battery_change_count = 0
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

            # 取电/还电 SOC 记录（仅保留SOC统计，用电量已由梯形积分法计算）
            start_socs, end_socs = [], []

            if '电池SOC' in group.columns and '电池id' in group.columns:
                for battery_segment, batt_group in group.groupby(group['电池切换标记'].cumsum()):
                    soc_seg = batt_group['电池SOC'].dropna()
                    if len(soc_seg) >= 2:
                        start_socs.append(soc_seg.iloc[0])
                        end_socs.append(soc_seg.iloc[-1])

            res['取电时平均SOC'] = round(np.mean(start_socs), 1) if start_socs else np.nan
            res['还电时平均SOC'] = round(np.mean(end_socs), 1) if end_socs else np.nan
            res['单次换电平均SOC消耗'] = round(
                np.mean(start_socs) - np.mean(end_socs), 1
            ) if start_socs and end_socs else np.nan

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

                # 骑行时刻分布（使用有效里程>0的点，与行驶距离逻辑一致）
                riding_hours = pd.to_datetime(
                    group.loc[group['有效里程_km'] > 0, '时间戳'], unit='s', errors='coerce'
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
            energy_data_valid = (res.get('总用电量(kWh)', 0) > 0)

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
            score = calc_monthly_score_v2(
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
                energy_per_100km=res.get('百公里电耗(kWh)', 0),
                battery_change_count=battery_change_count,
                daily_distance=total_distance,
                battery_change_cnt=battery_change_count
            )
            if customer_type == "地摊/储能":
                score = min(100, score + 20)
            elif customer_type == "改装/超速车":
                score -= 30
            score = max(0, min(100, round(score)))
            res['包月友好评分'] = score

            # -------------------------- 16. 用户等级判定 --------------------------
            user_level = determine_user_level_v2(
                over100a_cont=res['超100A连续次数'],
                over100a_hours=res['超100A累计时长_h'],
                current_80a_count=res['电流>80A次数'],
                soc_below_10_ratio=res['SOC低于10%时长占比'],
                min_soc=res['最低SOC'],
                current_60a_count=res['电流>60A次数'],
                energy_per_100km=res.get('百公里电耗(kWh)', 0),
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
            if current_level == "暴力":
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
                if energy_data_valid and res.get('百公里电耗(kWh)', 0) >= EXTREME_ENERGY_THRESHOLD:
                    reasons.append(f"能耗极高（百公里电耗{res['百公里电耗(kWh)']}kWh）")
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

    battery_voltage_map = load_battery_voltage_map(os.path.join(base_path, EXPORT_FILE_BATTERY_CELL_VOLTAGE))

    # 新数据源：直接从 battery_status_YYYY-MM-DD.parquet 读取
    raw_dir = os.path.join(base_path, EXPORT_PATH_BATTERY_STATUS_DAILY)
    output_dir = os.path.join(base_path, EXPORT_PATH_FACT_DAILY)
    os.makedirs(output_dir, exist_ok=True)
    
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.startswith('battery_status_') and f.endswith('.parquet')])
    if not raw_files:
        print("❌ 未找到原始数据文件")
        return {}
    
    # 过滤掉当天的数据（只计算到昨日）
    today_str = datetime.now().strftime('%Y-%m-%d')
    raw_files = [f for f in raw_files if f.replace('battery_status_', '').replace('.parquet', '') < today_str]
    if not raw_files:
        print("❌ 未找到需要处理的数据（当天数据已过滤）")
        return {}
    
    processed = {}
    
    for filename in raw_files:
        # 从文件名提取日期：battery_status_YYYY-MM-DD.parquet -> YYYY-MM-DD
        date_str = filename.replace('battery_status_', '').replace('.parquet', '')
        
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
        df_result = calc_contract_metrics(df_clean, battery_voltage_map)
        
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
