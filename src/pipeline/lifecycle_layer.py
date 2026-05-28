# -*- coding: utf-8 -*-
"""
L3 Lifecycle Layer - 用户生命周期层
职责：7天滚动聚合 + 动态阈值计算 + 用户等级 / 风险标签。
     唯一调用 dynamic_thresholds.py 的层。

输入：snapshot/user_daily.parquet（读取最近 30 天窗口）
输出：lifecycle/user_7d.parquet
"""

import os
import sys
import logging
import warnings
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import EXPORT_PATH_SNAPSHOT, EXPORT_PATH_LIFECYCLE_7D, EXPORT_FILE_CONTRACT_EARLY
from src.pipeline.dynamic_thresholds import DynamicBatteryAnalyzer

ROLLING_WINDOW_DAYS = 7
ROLLING_WEIGHTS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

LONG_TERM_OFFLINE_THRESHOLD_DAYS = 7

PATTERN_DOMINANT_RATIO = 0.5

STRATEGY_MAP = {
    '优质用户': '留存激励',
    '良好用户': '维持服务',
    '普通用户': '引导升级',
    '高损耗用户': '限制预警',
    '暴力': '清退处理',
    '观察期': '新手引导',
    '沉默用户': '激活唤醒',
}

LEVEL_RANK = {
    '暴力': 6, '高损耗用户': 4, '普通用户': 3, '观察期': 3,
    '良好用户': 2, '优质用户': 1, '未知': 3, '沉默用户': 5
}

# 全局合约信息缓存
_user_contract_map = None


def load_user_contract_info():
    """
    加载用户合约信息（首次入网日期、合约到期时间）
    
    Returns:
        dict: {user_id: {'join_date': 'YYYY-MM-DD', 'expire_date': 'YYYY-MM-DD'}}
    """
    global _user_contract_map
    if _user_contract_map is not None:
        return _user_contract_map
    
    if not os.path.exists(EXPORT_FILE_CONTRACT_EARLY):
        print(f"⚠️ 用户合约信息表不存在：{EXPORT_FILE_CONTRACT_EARLY}")
        _user_contract_map = {}
        return _user_contract_map
    
    print(f"📂 正在加载用户合约信息表...")
    try:
        df_contract = pd.read_csv(EXPORT_FILE_CONTRACT_EARLY, encoding='utf-8')
    except UnicodeDecodeError:
        df_contract = pd.read_csv(EXPORT_FILE_CONTRACT_EARLY, encoding='gbk')
    
    df_contract.columns = [str(col).strip() for col in df_contract.columns]
    
    if 'user_id' not in df_contract.columns:
        print(f"⚠️ 用户合约信息表缺少user_id列")
        _user_contract_map = {}
        return _user_contract_map
    
    df_contract['user_id'] = df_contract['user_id'].astype(str).str.strip()
    df_contract = df_contract[df_contract['user_id'].notna() & (df_contract['user_id'] != 'nan')]
    
    # 转换时间列为datetime
    if '入网时间' in df_contract.columns:
        df_contract['入网时间'] = pd.to_datetime(df_contract['入网时间'], errors='coerce')
    if '退网时间' in df_contract.columns:
        df_contract['退网时间'] = pd.to_datetime(df_contract['退网时间'], errors='coerce')
    
    # 使用groupby聚合：入网时间取min，退网时间取max
    agg_dict = {}
    if '入网时间' in df_contract.columns:
        agg_dict['join_date'] = ('入网时间', 'min')
    if '退网时间' in df_contract.columns:
        agg_dict['expire_date'] = ('退网时间', 'max')
    
    if agg_dict:
        g = df_contract.groupby('user_id').agg(**agg_dict)
        _user_contract_map = {}
        for uid, row in g.iterrows():
            join_date = row['join_date'].strftime('%Y-%m-%d') if 'join_date' in row and pd.notna(row['join_date']) else None
            expire_date = row['expire_date'].strftime('%Y-%m-%d') if 'expire_date' in row and pd.notna(row['expire_date']) else None
            _user_contract_map[uid] = {'join_date': join_date, 'expire_date': expire_date}
    else:
        _user_contract_map = {}
    
    print(f"✅ 用户合约信息表加载完成：共 {len(_user_contract_map)} 个用户")
    return _user_contract_map


def get_work_pattern(group):
    total_noon_long = group['近7d午间高峰长时骑行次数'].sum() if '近7d午间高峰长时骑行次数' in group.columns else 0
    total_evening_long = group['近7d晚间高峰长时骑行次数'].sum() if '近7d晚间高峰长时骑行次数' in group.columns else 0
    total_offpeak_long = group['近7d平峰长时骑行次数'].sum() if '近7d平峰长时骑行次数' in group.columns else 0
    total_night_long = group['近7d夜间长时骑行次数'].sum() if '近7d夜间长时骑行次数' in group.columns else 0
    total_long_all = total_noon_long + total_evening_long + total_offpeak_long + total_night_long
    
    if total_long_all > 0:
        period_counts = [
            ("习惯午间高峰", total_noon_long),
            ("习惯晚间高峰", total_evening_long),
            ("习惯非高峰期", total_offpeak_long),
            ("习惯晚上", total_night_long)
        ]
        period_counts.sort(key=lambda x: x[1], reverse=True)
        top_pattern, top_count = period_counts[0]
        if top_count / total_long_all >= PATTERN_DOMINANT_RATIO:
            return top_pattern
        else:
            return "全天"
    
    cols = ['近7d午间高峰总里程_km', '近7d晚间高峰总里程_km', '近7d平峰总里程_km', '近7d夜间总里程_km']
    if not all(c in group.columns for c in cols):
        return "数据不足"
    
    total_noon = group['近7d午间高峰总里程_km'].sum()
    total_evening = group['近7d晚间高峰总里程_km'].sum()
    total_offpeak = group['近7d平峰总里程_km'].sum()
    total_night = group['近7d夜间总里程_km'].sum()
    total_all = total_noon + total_evening + total_offpeak + total_night
    
    if total_all <= 0:
        return "数据不足"
    
    period_miles = [
        ("习惯午间高峰", total_noon),
        ("习惯晚间高峰", total_evening),
        ("习惯非高峰期", total_offpeak),
        ("习惯晚上", total_night)
    ]
    period_miles.sort(key=lambda x: x[1], reverse=True)
    top_pattern, top_miles = period_miles[0]
    if top_miles / total_all >= PATTERN_DOMINANT_RATIO:
        return top_pattern
    else:
        return "全天"


def _weighted_avg(series, weights):
    valid = pd.concat([series, weights], axis=1).dropna()
    if len(valid) == 0:
        return 0
    w_sum = valid[weights.name].sum()
    if w_sum == 0:
        return 0
    return (valid[series.name] * valid[weights.name]).sum() / w_sum


def _weighted_ratio(ratio_col, denominator_col, weights):
    valid = pd.concat([ratio_col, denominator_col, weights], axis=1).dropna()
    if len(valid) == 0:
        return 0
    weighted_numerator = (valid[ratio_col.name] * valid[denominator_col.name] * valid[weights.name]).sum()
    weighted_denominator = (valid[denominator_col.name] * valid[weights.name]).sum()
    if weighted_denominator == 0:
        return 0
    return weighted_numerator / weighted_denominator


def calc_rolling_7d_metrics(df_snapshot: pd.DataFrame) -> pd.DataFrame:
    print("计算7天滚动指标...")

    df = df_snapshot.copy()
    df['统计日期'] = pd.to_datetime(df['统计日期'])
    df = df.sort_values(['用户id', '统计日期']).reset_index(drop=True)

    users = df['用户id'].unique()
    results = []

    for user_id in tqdm(users, desc="计算用户7天滚动指标"):
        user_data = df[df['用户id'] == user_id].sort_values('统计日期')

        if len(user_data) == 0:
            continue

        latest_date = user_data['统计日期'].max()
        start_date = latest_date - pd.Timedelta(days=6)

        window_data = user_data[(user_data['统计日期'] >= start_date) & (user_data['统计日期'] <= latest_date)]

        if len(window_data) == 0:
            continue

        rolling_metrics = _calc_single_user_7d_rolling(window_data, user_id, latest_date)
        results.append(rolling_metrics)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def _calc_single_user_7d_rolling(window_data: pd.DataFrame, user_id: str, latest_date: pd.Timestamp) -> dict:
    group = window_data.sort_values('统计日期', ascending=False).copy()
    group['days_ago'] = (latest_date - group['统计日期']).dt.days
    group = group[group['days_ago'].between(0, ROLLING_WINDOW_DAYS - 1)].copy()
    group['weight'] = group['days_ago'].apply(lambda x: ROLLING_WEIGHTS[int(x)] if int(x) < len(ROLLING_WEIGHTS) else ROLLING_WEIGHTS[-1])

    total_data_days = len(group)

    if '数据完整性' in group.columns:
        complete_group = group[group['数据完整性'] != '不完整'].copy()
    else:
        complete_group = group.copy()

    valid_days = len(complete_group)

    res = {
        '用户id': user_id,
        '统计日期': latest_date.strftime('%Y-%m-%d'),
        '近7天有数据天数': total_data_days,
        '近7天有效数据天数': valid_days
    }

    latest_row = group[group['days_ago'] == 0].iloc[0] if not group[group['days_ago'] == 0].empty else group.iloc[0]
    for col in ['核心活动省份', '核心活动城市', '核心活动区县']:
        if col in latest_row:
            res[col] = latest_row[col]

    weighted_cols = [
        ('单合约日均行驶里程_km', '近7d单合约日均行驶里程_km'),
        ('单合约日均骑行时长_h', '近7d单合约日均骑行时长_h'),
        ('单合约日均放电时长_h', '近7d单合约日均放电时长_h'),
        ('单合约日均怠速放电_h', '近7d单合约日均怠速放电_h'),
        ('单合约日均用电量_kWh', '近7d单合约日均用电量_kWh'),
        ('平均包月友好分', '近7d平均包月友好分'),
        ('当日平均骑行SOC', '近7d平均骑行SOC'),
        ('当日平均骑行速度_kmh', '近7d平均骑行速度_kmh'),
        ('当日平均骑行电流_A', '近7d平均骑行电流_A'),
        ('当日百公里电耗_kWh', '近7d百公里电耗_kWh'),
        ('当日P50骑行速度_kmh', '近7d_P50骑行速度_kmh'),
        ('当日P90骑行速度_kmh', '近7d_P90骑行速度_kmh'),
        ('当日夜间骑行均速_kmh', '近7d夜间骑行均速_kmh'),
        ('取电时平均SOC_当日', '近7d取电时平均SOC'),
        ('还电时平均SOC_当日', '近7d还电时平均SOC'),
        ('单次换电平均SOC消耗_当日', '近7d单次换电SOC消耗'),
        ('平峰换电占比_当日', '近7d平峰换电占比'),
        ('深夜换电占比_当日', '近7d深夜换电占比'),
    ]
    for raw_col, roll_col in weighted_cols:
        if raw_col in complete_group.columns:
            res[roll_col] = round(_weighted_avg(complete_group[raw_col], complete_group['weight']), 2)
        else:
            res[roll_col] = 0

    if '高峰骑行占比' in complete_group.columns and '当日总骑行时长_h' in complete_group.columns:
        res['近7d高峰骑行占比'] = round(_weighted_ratio(complete_group['高峰骑行占比'], complete_group['当日总骑行时长_h'], complete_group['weight']), 2)
    else:
        res['近7d高峰骑行占比'] = 0

    soc20_col = '当日SOC低于20%时长占比' if '当日SOC低于20%时长占比' in complete_group.columns else 'SOC低于20%时长占比_当日'
    if soc20_col in complete_group.columns and '当日总骑行时长_h' in complete_group.columns:
        res['近7d_SOC低于20%时长占比'] = round(_weighted_ratio(complete_group[soc20_col], complete_group['当日总骑行时长_h'], complete_group['weight']), 4)
    else:
        res['近7d_SOC低于20%时长占比'] = 0

    soc10_col = '当日SOC低于10%时长占比' if '当日SOC低于10%时长占比' in complete_group.columns else 'SOC低于10%时长占比_当日'
    if soc10_col in complete_group.columns and '当日总骑行时长_h' in complete_group.columns:
        res['近7d_SOC低于10%时长占比'] = round(_weighted_ratio(complete_group[soc10_col], complete_group['当日总骑行时长_h'], complete_group['weight']), 4)
    else:
        res['近7d_SOC低于10%时长占比'] = 0

    max_cols = [
        ('当日最高速度_kmh', '近7d最高速度_kmh'),
        ('当日最大电流_A', '近7d最大电流_A'),
        ('当日最高温度_℃', '近7d最高温度_℃'),
        ('当日峰值功率_W', '近7d峰值功率_W'),
        ('最大单合约活动半径_km', '近7d最大活动半径_km'),
        ('R90活动半径_km', '近7d_R90活动半径_km'),
        ('最大凸包覆盖面积_km2', '近7d凸包覆盖面积_km2'),
        ('最大单次出行距离_km', '近7d最大单次出行距离_km'),
        ('当日跨区域转移次数', '近7d最大跨区域转移次数'),
    ]
    for raw_col, roll_col in max_cols:
        if raw_col in complete_group.columns:
            val = complete_group[raw_col].max()
            res[roll_col] = round(val, 2) if pd.notna(val) else 0
        else:
            res[roll_col] = 0

    # 新增特征聚合（文档标准：众包/专送判定增强）
    avg_cols_7d = [
        ('当日上线时间熵值', '近7d平均上线时间熵值'),
        ('当日骑行时段集中度', '近7d平均骑行时段集中度'),
        ('当日路线曲折系数', '近7d平均路线曲折系数'),
        ('当日速度变异系数', '近7d平均速度变异系数'),
        ('当日静止时长占比', '近7d平均静止时长占比'),
    ]
    for raw_col, roll_col in avg_cols_7d:
        if raw_col in complete_group.columns:
            val = complete_group[raw_col].mean()
            res[roll_col] = round(val, 3) if pd.notna(val) else 0
        else:
            res[roll_col] = 0

    if '最晚骑行时刻_h' in complete_group.columns:
        valid = complete_group['最晚骑行时刻_h'][complete_group['最晚骑行时刻_h'] >= 0]
        res['近7d最晚骑行时刻_h'] = round(valid.max(), 2) if len(valid) > 0 else -1
    else:
        res['近7d最晚骑行时刻_h'] = -1

    min_cols = [
        ('当日最低SOC', '近7d最低SOC'),
    ]
    for raw_col, roll_col in min_cols:
        if raw_col in complete_group.columns:
            val = complete_group[raw_col].min()
            res[roll_col] = round(val, 2) if pd.notna(val) else 100
        else:
            res[roll_col] = 100

    if '最早骑行时刻_h' in complete_group.columns:
        valid = complete_group['最早骑行时刻_h'][complete_group['最早骑行时刻_h'] >= 0]
        res['近7d最早骑行时刻_h'] = round(valid.min(), 2) if len(valid) > 0 else -1
    else:
        res['近7d最早骑行时刻_h'] = -1

    cumulative_and_avg_cols = [
        ('当日总骑行次数', '近7d总骑行次数', '近7d日均骑行次数'),
        ('当日总换电次数', '近7d总换电次数', '近7d日均换电次数'),
        ('总电流超60A次数', '近7d电流超60A总次数', '近7d日均电流超60A次数'),
        ('总电流超80A次数', '近7d电流超80A总次数', '近7d日均电流超80A次数'),
        ('总超100A连续次数', '近7d超100A连续总次数', '近7d日均超100A连续次数'),
        ('总超100A累计时长_h', '近7d超100A累计时长_h', '近7d日均超100A累计时长_h'),
        ('午间高峰长时骑行总次数', '近7d午间高峰长时骑行次数', '近7d日均午间高峰长时骑行次数'),
        ('晚间高峰长时骑行总次数', '近7d晚间高峰长时骑行次数', '近7d日均晚间高峰长时骑行次数'),
        ('平峰长时骑行总次数', '近7d平峰长时骑行次数', '近7d日均平峰长时骑行次数'),
        ('夜间长时骑行总次数', '近7d夜间长时骑行次数', '近7d日均夜间长时骑行次数'),
        ('午间高峰总里程_km', '近7d午间高峰总里程_km', '近7d日均午间高峰总里程_km'),
        ('晚间高峰总里程_km', '近7d晚间高峰总里程_km', '近7d日均晚间高峰总里程_km'),
        ('平峰总里程_km', '近7d平峰总里程_km', '近7d日均平峰总里程_km'),
        ('夜间总里程_km', '近7d夜间总里程_km', '近7d日均夜间总里程_km'),
        ('当日平峰换电次数', '近7d平峰换电总次数', '近7d日均平峰换电次数'),
        ('当日深夜换电次数', '近7d深夜换电总次数', '近7d日均深夜换电次数'),
        ('当日高峰换电次数', '近7d高峰换电总次数', '近7d日均高峰换电次数'),
        ('当日高速骑行点数', '近7d高速骑行总点数', '近7d日均高速骑行点数'),
        ('当日总行驶距离_km', '近7d总行驶距离_km', '近7d日均行驶距离_km'),
    ]

    for raw_col, cum_col, avg_col in cumulative_and_avg_cols:
        if raw_col in complete_group.columns:
            cum_val = complete_group[raw_col].sum()
            res[cum_col] = round(cum_val, 2) if pd.notna(cum_val) else 0
            res[avg_col] = round(res[cum_col] / valid_days, 2) if valid_days > 0 else 0
        else:
            res[cum_col] = 0
            res[avg_col] = 0

    if '当日是否出勤' in complete_group.columns:
        attendance_sum = complete_group['当日是否出勤'].sum()
        res['近7d出勤天数'] = round(attendance_sum, 2) if pd.notna(attendance_sum) else 0
        res['近7d出勤率'] = round(res['近7d出勤天数'] / valid_days, 2) if valid_days > 0 else 0
    else:
        res['近7d出勤天数'] = 0
        res['近7d出勤率'] = 0

    pattern_group = pd.DataFrame([{
        '近7d午间高峰长时骑行次数': res['近7d午间高峰长时骑行次数'],
        '近7d晚间高峰长时骑行次数': res['近7d晚间高峰长时骑行次数'],
        '近7d平峰长时骑行次数': res['近7d平峰长时骑行次数'],
        '近7d夜间长时骑行次数': res['近7d夜间长时骑行次数'],
        '近7d午间高峰总里程_km': res['近7d午间高峰总里程_km'],
        '近7d晚间高峰总里程_km': res['近7d晚间高峰总里程_km'],
        '近7d平峰总里程_km': res['近7d平峰总里程_km'],
        '近7d夜间总里程_km': res['近7d夜间总里程_km'],
    }])
    res['近7d工作特点'] = get_work_pattern(pattern_group)

    # -------------------------- 车辆形态_7d --------------------------
    if '车辆形态_综合' in complete_group.columns:
        vt_values = complete_group['车辆形态_综合'].dropna()
        if len(vt_values) > 0:
            vt_unique = vt_values.unique()
            if '改装/超速车' in vt_unique:
                res['车辆形态_7d'] = '改装/超速车'
            elif '地摊/储能' in vt_unique:
                res['车辆形态_7d'] = '地摊/储能'
            else:
                res['车辆形态_7d'] = vt_values.value_counts().index[0]
        else:
            res['车辆形态_7d'] = "数据不足"
    else:
        res['车辆形态_7d'] = "数据不足"

    vt_desc_7d = ""
    vt_7d = res.get('车辆形态_7d', '数据不足')
    max_speed_7d = res.get('近7d最高速度_kmh', 0)
    avg_current_7d = res.get('近7d平均骑行电流_A', 0)
    peak_power_7d = res.get('近7d峰值功率_W', 0)
    peak_power_str = f"{float(peak_power_7d):.0f}" if peak_power_7d is not None else "0"

    if vt_7d == "改装/超速车":
        vt_desc_7d = f"行驶特征异常（近7天最高速度{max_speed_7d}km/h，平均骑行电流{avg_current_7d}A，峰值功率{peak_power_str}W），远超正常电动车水平，存在改装或超速嫌疑"
    elif vt_7d == "电动自行车":
        vt_desc_7d = f"符合电动自行车国标（最大车速≤25km/h），近7天实测最高{max_speed_7d}km/h"
    elif vt_7d == "电动轻便摩托车":
        vt_desc_7d = f"符合电动轻便摩托车国标（最大车速≤50km/h），近7天实测最高{max_speed_7d}km/h"
    elif vt_7d == "电动摩托车":
        vt_desc_7d = f"符合电动摩托车国标（最大车速>50km/h），近7天实测最高{max_speed_7d}km/h"
    elif vt_7d == "地摊/储能":
        vt_desc_7d = "非移动用电场景，车辆形态不适用"
    else:
        vt_desc_7d = "无有效骑行数据，无法判定车辆形态"
    res['车辆形态_7d说明'] = vt_desc_7d

    # -------------------------- 客户形态_7d（含空间维度 + 行为特征众包/专送重定义）--------------------------
    hull_area_7d = res.get('近7d凸包覆盖面积_km2', 0)
    r90_7d = res.get('近7d_R90活动半径_km', 0)
    avg_ride_hour_7d = res.get('近7d单合约日均骑行时长_h', 0)
    peak_ratio_7d = res.get('近7d高峰骑行占比', 0)
    avg_mileage_7d = res.get('近7d单合约日均行驶里程_km', 0)
    avg_idle_discharge_7d = res.get('近7d单合约日均怠速放电_h', 0)

    # 新增：文档标准特征
    time_entropy_7d = res.get('近7d平均上线时间熵值', 0)
    hour_concentration_7d = res.get('近7d平均骑行时段集中度', 0)
    detour_ratio_7d = res.get('近7d平均路线曲折系数', 0)
    speed_cv_7d = res.get('近7d平均速度变异系数', 0)
    cross_region_7d = res.get('近7d最大跨区域转移次数', 0)
    idle_ratio_7d = res.get('近7d平均静止时长占比', 0)

    # 空间集中度阈值（专送骑手活动范围小且集中，众包骑手跨区域大范围移动）
    SPATIAL_CONCENTRATED_R90 = 5.0
    SPATIAL_CONCENTRATED_HULL = 20.0

    # 行为特征阈值（文档标准）
    TIME_ENTROPY_LOW = 3.5  # 专送上线时间熵值低（规律性强）
    DETOUR_RATIO_LOW = 2.0  # 专送路线曲折系数低（顺路）
    SPEED_CV_LOW = 0.4  # 专送速度变异系数低（稳定）
    CROSS_REGION_LOW = 3  # 专送跨区域转移少
    IDLE_RATIO_LOW = 0.5  # 专送静止时长占比低（持续派单）

    # 调试日志：记录判定输入特征
    logger.debug(f"[客户形态判定] 用户{user_id} 输入特征: "
                 f"日均骑行时长={avg_ride_hour_7d:.2f}h, 高峰占比={peak_ratio_7d:.2f}, "
                 f"R90半径={r90_7d:.2f}km, 凸包面积={hull_area_7d:.2f}km², "
                 f"时间熵={time_entropy_7d:.3f}, 路线曲折系数={detour_ratio_7d:.2f}, "
                 f"速度变异系数={speed_cv_7d:.3f}, 跨区域转移={cross_region_7d}次, "
                 f"静止占比={idle_ratio_7d:.3f}")

    if '客户形态_综合' in complete_group.columns:
        type_values = complete_group['客户形态_综合'].dropna()
        if len(type_values) > 0:
            unique_types = type_values.unique()
            if '地摊/储能' in unique_types:
                res['客户形态_综合_7d'] = '地摊/储能'
                logger.info(f"[客户形态判定] 用户{user_id} → 地摊/储能 (日级数据包含地摊/储能标签)")
            else:
                daily_modal_type = type_values.value_counts().index[0]
                logger.debug(f"[客户形态判定] 用户{user_id} 日级众数类型={daily_modal_type}")
                # 基于7天聚合数据重新判定众包/专送（加入空间维度 + 行为特征）
                if daily_modal_type == '专送骑手':
                    is_spatial_concentrated = (hull_area_7d > 0 and hull_area_7d <= SPATIAL_CONCENTRATED_HULL) or \
                                              (r90_7d > 0 and r90_7d <= SPATIAL_CONCENTRATED_R90)
                    is_behavior_concentrated = (time_entropy_7d <= TIME_ENTROPY_LOW) and \
                                               (detour_ratio_7d <= DETOUR_RATIO_LOW) and \
                                               (speed_cv_7d <= SPEED_CV_LOW)
                    
                    logger.debug(f"[客户形态判定] 用户{user_id} 专送候选判定: "
                                 f"空间集中={is_spatial_concentrated}(凸包{hull_area_7d:.1f}≤{SPATIAL_CONCENTRATED_HULL}, R90{r90_7d:.1f}≤{SPATIAL_CONCENTRATED_R90}), "
                                 f"行为集中={is_behavior_concentrated}(熵{time_entropy_7d:.2f}≤{TIME_ENTROPY_LOW}, 曲折{detour_ratio_7d:.2f}≤{DETOUR_RATIO_LOW}, 速度CV{speed_cv_7d:.3f}≤{SPEED_CV_LOW})")
                    
                    if avg_ride_hour_7d >= 4.0 and peak_ratio_7d >= 0.3 and is_spatial_concentrated and is_behavior_concentrated:
                        res['客户形态_综合_7d'] = '专送骑手'
                        logger.info(f"[客户形态判定] 用户{user_id} → 专送骑手 (日均{avg_ride_hour_7d:.1f}h, 高峰{peak_ratio_7d:.0%}, 空间集中, 行为规律)")
                    elif avg_ride_hour_7d >= 2.0 and peak_ratio_7d >= 0.15:
                        res['客户形态_综合_7d'] = '众包骑手'
                        logger.info(f"[客户形态判定] 用户{user_id} → 众包骑手 (日均{avg_ride_hour_7d:.1f}h, 高峰{peak_ratio_7d:.0%}, 空间/行为未达标)")
                    else:
                        res['客户形态_综合_7d'] = '普通骑手'
                        logger.info(f"[客户形态判定] 用户{user_id} → 普通骑手 (日均{avg_ride_hour_7d:.1f}h<2h 或 高峰{peak_ratio_7d:.0%}<15%)")
                elif daily_modal_type == '众包骑手':
                    is_spatial_dispersed = (hull_area_7d <= 0 or hull_area_7d > SPATIAL_CONCENTRATED_HULL) or \
                                           (r90_7d <= 0 or r90_7d > SPATIAL_CONCENTRATED_R90)
                    is_behavior_dispersed = (time_entropy_7d > TIME_ENTROPY_LOW) or \
                                            (detour_ratio_7d > DETOUR_RATIO_LOW) or \
                                            (cross_region_7d > CROSS_REGION_LOW)
                    
                    logger.debug(f"[客户形态判定] 用户{user_id} 众包候选判定: "
                                 f"空间分散={is_spatial_dispersed}(凸包{hull_area_7d:.1f}>{SPATIAL_CONCENTRATED_HULL}或R90{r90_7d:.1f}>{SPATIAL_CONCENTRATED_R90}), "
                                 f"行为分散={is_behavior_dispersed}(熵{time_entropy_7d:.2f}>{TIME_ENTROPY_LOW}, 曲折{detour_ratio_7d:.2f}>{DETOUR_RATIO_LOW}, 转移{cross_region_7d}>{CROSS_REGION_LOW})")
                    
                    if avg_ride_hour_7d >= 4.0 and peak_ratio_7d >= 0.3 and not is_spatial_dispersed and not is_behavior_dispersed:
                        res['客户形态_综合_7d'] = '专送骑手'
                        logger.info(f"[客户形态判定] 用户{user_id} → 专送骑手 (日级众包但7天特征符合专送: 日均{avg_ride_hour_7d:.1f}h, 高峰{peak_ratio_7d:.0%}, 空间集中, 行为规律)")
                    elif avg_ride_hour_7d >= 2.0 and peak_ratio_7d >= 0.15:
                        res['客户形态_综合_7d'] = '众包骑手'
                        logger.info(f"[客户形态判定] 用户{user_id} → 众包骑手 (日均{avg_ride_hour_7d:.1f}h, 高峰{peak_ratio_7d:.0%}, 空间/行为分散)")
                    else:
                        res['客户形态_综合_7d'] = '普通骑手'
                        logger.info(f"[客户形态判定] 用户{user_id} → 普通骑手 (日均{avg_ride_hour_7d:.1f}h<2h 或 高峰{peak_ratio_7d:.0%}<15%)")
                else:
                    res['客户形态_综合_7d'] = daily_modal_type
        else:
            res['客户形态_综合_7d'] = "数据不足"
            logger.debug(f"[客户形态判定] 用户{user_id} → 数据不足 (无有效日级客户形态数据)")
    else:
        res['客户形态_综合_7d'] = "数据不足"
        logger.debug(f"[客户形态判定] 用户{user_id} → 数据不足 (complete_group无客户形态_综合列)")

    type_desc_7d = ""
    current_type_7d = res.get('客户形态_综合_7d', '数据不足')

    if current_type_7d == "地摊/储能":
        type_desc_7d = f"非移动用电特征明显（近7天日均骑行{avg_ride_hour_7d}小时，日均怠速放电{avg_idle_discharge_7d}小时），放电以静止状态为主，疑似地摊供电或储能场景"
    elif current_type_7d == "专送骑手":
        type_desc_7d = f"工作特征显著（近7天日均骑行{avg_ride_hour_7d:.1f}小时，高峰骑行占比{peak_ratio_7d*100:.0f}%），活动范围集中（R90半径{r90_7d:.1f}km，凸包面积{hull_area_7d:.1f}km²），行为规律（时间熵{time_entropy_7d:.2f}，路线曲折系数{detour_ratio_7d:.2f}），符合专送骑手画像"
    elif current_type_7d == "众包骑手":
        type_desc_7d = f"具有兼职骑手特征（近7天日均骑行{avg_ride_hour_7d:.1f}小时，高峰骑行占比{peak_ratio_7d*100:.0f}%），活动范围分散（R90半径{r90_7d:.1f}km，凸包面积{hull_area_7d:.1f}km²），行为灵活（时间熵{time_entropy_7d:.2f}，跨区域转移{cross_region_7d}次），符合众包骑手画像"
    elif current_type_7d == "标准骑手":
        type_desc_7d = f"骑行行为规律（近7天日均里程{avg_mileage_7d}km，日均骑行{avg_ride_hour_7d:.1f}小时），属于标准日常使用场景"
    elif current_type_7d == "普通骑手":
        type_desc_7d = f"有常规骑行行为（近7天日均里程{avg_mileage_7d}km，日均骑行{avg_ride_hour_7d:.1f}小时），但不符合特定骑手标签特征"
    else:
        type_desc_7d = "无有效骑行数据或数据量不足，无法判定具体使用场景"
    
    res['客户形态_综合_7d说明'] = type_desc_7d

    latest_record_date = pd.to_datetime(group['统计日期'].max())
    
    # 以数据窗口最新日期为基准，而非 today，避免离线天数随报告日期自动增长
    days_since_last_seen = (latest_date - latest_record_date).days
    
    if days_since_last_seen >= LONG_TERM_OFFLINE_THRESHOLD_DAYS:
        res['设备状态监控'] = f"异常(离线{days_since_last_seen}天)"
    elif days_since_last_seen >= 1:
        res['设备状态监控'] = f"暂离线({days_since_last_seen}天)"
    else:
        res['设备状态监控'] = "正常在线"

    return res


def log_type_distribution_summary(df_results: pd.DataFrame):
    """输出客户形态分布统计摘要，方便排查分类比例异常"""
    if '客户形态_综合_7d' in df_results.columns:
        type_counts = df_results['客户形态_综合_7d'].value_counts()
        logger.info(f"[客户形态分布统计] 总用户数={len(df_results)}")
        for ctype, cnt in type_counts.items():
            logger.info(f"  {ctype}: {cnt}人 ({cnt/len(df_results)*100:.1f}%)")
        
        # 输出特征均值对比
        if '近7d平均上线时间熵值' in df_results.columns:
            zhuan = df_results[df_results['客户形态_综合_7d'] == '专送骑手']
            zhong = df_results[df_results['客户形态_综合_7d'] == '众包骑手']
            if len(zhuan) > 0 and len(zhong) > 0:
                logger.info(f"[特征对比] 专送 vs 众包:")
                logger.info(f"  时间熵: 专送={zhuan['近7d平均上线时间熵值'].mean():.3f}, 众包={zhong['近7d平均上线时间熵值'].mean():.3f}")
                logger.info(f"  路线曲折系数: 专送={zhuan['近7d平均路线曲折系数'].mean():.2f}, 众包={zhong['近7d平均路线曲折系数'].mean():.2f}")
                logger.info(f"  速度变异系数: 专送={zhuan['近7d平均速度变异系数'].mean():.3f}, 众包={zhong['近7d平均速度变异系数'].mean():.3f}")
                logger.info(f"  跨区域转移次数: 专送={zhuan['近7d最大跨区域转移次数'].mean():.1f}, 众包={zhong['近7d最大跨区域转移次数'].mean():.1f}")
                logger.info(f"  静止时长占比: 专送={zhuan['近7d平均静止时长占比'].mean():.3f}, 众包={zhong['近7d平均静止时长占比'].mean():.3f}")


# 月度用电预估参数
MIN_RECORD_DAYS_FOR_ESTIMATE = 7
FULL_RECORD_DAYS_FOR_CUMULATIVE = 30
DEFAULT_MONTH_DAYS = 26
MONTH_CALENDAR_DAYS = 30


def calc_user_monthly_attendance(df_rolling: pd.DataFrame, df_snapshot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于用户历史全量数据计算月度用电预估

    有效日判定：用电量>0 即算出勤。仅排除数据不完整的日期。

    分三档：
    - 完整记录天数 < 7：兜底值 26 天
    - 7 <= 完整记录天数 < 30：出勤率 × 30 天
    - 完整记录天数 >= 30：出勤率 × 30 天

    返回：
        (df_rolling, df_attendance_detail)
    """
    attendance_stats = {}
    attendance_details = []

    for uid, group in df_snapshot.groupby('用户id'):
        group_unique = group.drop_duplicates(subset=['统计日期']).sort_values('统计日期')

        incomplete_days = 0
        if '数据完整性' in group_unique.columns:
            incomplete_mask = group_unique['数据完整性'] == '不完整'
            incomplete_days = incomplete_mask.sum()
            group_valid = group_unique[~incomplete_mask].copy()
        else:
            group_valid = group_unique.copy()

        total_record_days = len(group_valid)
        total_attendance_days = group_valid['当日是否出勤'].sum() if '当日是否出勤' in group_valid.columns else 0
        attendance_rate = round(total_attendance_days / total_record_days, 4) if total_record_days > 0 else 0

        if '单合约日均用电量_kWh' in group_valid.columns:
            avg_single_contract_energy = group_valid['单合约日均用电量_kWh'].mean()
        else:
            total_cum_energy = group_valid['当日总用电量_kWh'].sum() if '当日总用电量_kWh' in group_valid.columns else 0
            avg_single_contract_energy = total_cum_energy / total_record_days if total_record_days > 0 else 0

        if total_record_days < MIN_RECORD_DAYS_FOR_ESTIMATE:
            month_estimate_days = DEFAULT_MONTH_DAYS
            estimate_type = "兜底值"
        elif total_record_days < FULL_RECORD_DAYS_FOR_CUMULATIVE:
            month_estimate_days = round(attendance_rate * MONTH_CALENDAR_DAYS, 1)
            estimate_type = "实际出勤率估算"
        else:
            month_estimate_days = round(attendance_rate * MONTH_CALENDAR_DAYS, 1)
            estimate_type = "真实累计计算"

        final_month_energy = round(avg_single_contract_energy * month_estimate_days, 2)

        attendance_stats[uid] = {
            '月度预估工作天数': month_estimate_days,
            '单合约月度用电度数预估_kWh': final_month_energy,
        }

        attendance_details.append({
            '用户id': uid,
            '完整记录天数': total_record_days,
            '排除的不完整天数': int(incomplete_days),
            '实际出勤总天数': total_attendance_days,
            '历史出勤率': attendance_rate,
            '历史单合约日均用电量_kWh': round(avg_single_contract_energy, 2),
            '月度预估工作天数': month_estimate_days,
            '单合约月度用电度数预估_kWh': final_month_energy,
            '预估类型': estimate_type,
            '最新统计日期': group_unique['统计日期'].max()
        })
    
    # 合并到7天滚动数据
    df = df_rolling.copy()
    df['月度预估工作天数'] = df['用户id'].map(lambda x: attendance_stats.get(x, {}).get('月度预估工作天数', DEFAULT_MONTH_DAYS))
    df['单合约月度用电度数预估_kWh'] = df['用户id'].map(lambda x: attendance_stats.get(x, {}).get('单合约月度用电度数预估_kWh', 0))
    
    # 生成明细表
    df_attendance_detail = pd.DataFrame(attendance_details)
    
    return df, df_attendance_detail


def determine_lifecycle_state(row: dict, current_date: pd.Timestamp) -> str:
    """
    判断用户生命周期状态
    
    状态优先级：已到期 > 新用户 > 流失 > 沉默 > 轻度活跃 > 活跃
    """
    MIN_DAYS_FOR_ACTIVE = 5
    MIN_DATA_DAYS_TO_AVOID_CHURN = 1
    
    expire_date = pd.to_datetime(row.get('合约到期时间'), errors='coerce')
    if pd.notna(expire_date) and current_date > expire_date:
        return '已到期'
    
    join_date = pd.to_datetime(row.get('首次入网日期'), errors='coerce')
    if pd.notna(join_date):
        days_since_join = (current_date - join_date).days
        if days_since_join <= 3:
            return '新用户'
    
    attendance_days = row.get('近7d出勤天数', 0)
    has_data_days = row.get('近7天有效数据天数', 0)
    
    if attendance_days == 0:
        if has_data_days <= MIN_DATA_DAYS_TO_AVOID_CHURN:
            return '流失'
        else:
            return '沉默'
    elif attendance_days >= MIN_DAYS_FOR_ACTIVE:
        return '活跃'
    else:
        return '轻度活跃'


def _add_derived_portrait_metrics(df: pd.DataFrame) -> pd.DataFrame:

    r90 = df.get('近7d_R90活动半径_km', pd.Series(0, index=df.index)).fillna(0)
    r95 = df.get('近7d最大活动半径_km', pd.Series(1, index=df.index)).replace(0, 1)
    df['近7d活动集中度'] = (r90 / r95).clip(0, 1).round(3)

    offpeak_ratio = df.get('近7d平峰换电占比', pd.Series(0, index=df.index)).fillna(0)
    night_ratio   = df.get('近7d深夜换电占比', pd.Series(0, index=df.index)).fillna(0)
    
    total_swaps   = df.get('近7d总换电次数', pd.Series(0, index=df.index)).fillna(0)

    swap_friendly = np.where(total_swaps == 0, 0.0, (offpeak_ratio * 0.6 + night_ratio * 0.4) * 20)
    df['近7d换电友好分'] = np.clip(swap_friendly, 0, 20).round(1)

    def _swap_period_label(row):
        offpeak = row.get('近7d平峰换电占比', 0) or 0
        night   = row.get('近7d深夜换电占比', 0) or 0
        peak    = 1 - offpeak - night
        if offpeak >= 0.5:  return '偏好平峰换电'
        if night   >= 0.4:  return '偏好深夜换电'
        if offpeak + night >= 0.5: return '平峰+深夜混合'
        if peak >= 0.6:     return '高峰换电为主'
        return '换电时段分散'
    df['换电时段偏好_7d'] = df.apply(_swap_period_label, axis=1)

    attendance_rate = df.get('近7d出勤率', pd.Series(0, index=df.index)).fillna(0)
    df['近7d骑行规律性评分'] = (attendance_rate * 10).clip(0, 10).round(1)

    soc_consume = df.get('近7d单次换电SOC消耗', pd.Series(np.nan, index=df.index))
    df['近7d单次换电SOC消耗'] = soc_consume.fillna(0).round(1)

    night_speed = df.get('近7d夜间骑行均速_kmh', pd.Series(0, index=df.index)).fillna(0)
    df['夜间高速风险'] = np.select(
        [night_speed >= 45, night_speed >= 35],
        ['高', '中'],
        default='低'
    )

    return df


def _generate_full_portrait(df: pd.DataFrame) -> pd.DataFrame:

    portraits = []
    for _, row in df.iterrows():
        parts = []

        level     = row.get('用户等级_动态', '未知')
        ctype     = row.get('客户形态_综合_7d', '未知')
        lifecycle = row.get('用户生命周期状态_7d', '未知')
        parts.append(f"【类型】{ctype} | {level} | 生命周期:{lifecycle}")

        earliest = row.get('近7d最早骑行时刻_h', -1)
        latest   = row.get('近7d最晚骑行时刻_h', -1)
        pattern  = row.get('近7d工作特点', '未知')
        attend   = row.get('近7d出勤率', 0)
        ride_h   = row.get('近7d单合约日均骑行时长_h', 0)
        if earliest >= 0 and latest >= 0:
            def _fmt_hm(h):
                hh = int(h)
                mm = int(round((h - hh) * 60))
                return f"{hh:02d}:{mm:02d}"
            parts.append(
                f"【骑行时间】惯用时段:{pattern} | "
                f"首次上路:{_fmt_hm(earliest)} / 最晚收车:{_fmt_hm(latest)} | "
                f"出勤率:{attend:.0%} | 日均骑行:{ride_h:.1f}h"
            )
        else:
            parts.append(
                f"【骑行时间】惯用时段:无 | "
                f"首次上路:无 / 最晚收车:无 | "
                f"出勤率:{attend:.0%} | 日均骑行:{ride_h:.1f}h"
            )

        swap_pref   = row.get('换电时段偏好_7d', '未知')
        swap_total  = row.get('近7d总换电次数', 0)
        swap_friend = row.get('近7d换电友好分', 0)
        offpeak_r   = row.get('近7d平峰换电占比', 0)
        night_r     = row.get('近7d深夜换电占比', 0)
        if swap_total > 0:
            parts.append(
                f"【换电习惯】{swap_pref} | 7天共换电{swap_total:.0f}次 | "
                f"平峰占比:{offpeak_r:.0%} / 深夜占比:{night_r:.0%} | "
                f"换电友好分:{swap_friend:.1f}/20"
            )
        else:
            parts.append("【换电习惯】近7天无换电记录")

        province = row.get('核心活动省份', '')
        city     = row.get('核心活动城市', '')
        district = row.get('核心活动区县', '')
        r90      = row.get('近7d_R90活动半径_km', 0)
        r95      = row.get('近7d最大活动半径_km', 0)
        hull     = row.get('近7d凸包覆盖面积_km2', 0)
        max_dist = row.get('近7d最大单次出行距离_km', 0)
        conc     = row.get('近7d活动集中度', 0)
        location = ' '.join(filter(None, [province, city, district]))
        conc_label = '高度集中' if conc > 0.8 else ('较集中' if conc > 0.6 else '活动范围广')
        parts.append(
            f"【活动区域】{location} | "
            f"日常半径(R90):{r90:.1f}km / 极限半径(R95):{r95:.1f}km | "
            f"覆盖面积:{hull:.1f}km² | 最远单次出行:{max_dist:.1f}km | "
            f"集中度:{conc_label}"
        )

        total_km  = row.get('近7d总行驶距离_km', 0)
        daily_km  = row.get('近7d单合约日均行驶里程_km', 0)
        energy100 = row.get('近7d百公里电耗_kWh', 0)
        monthly_e = row.get('单合约月度用电度数预估_kWh', 0)
        parts.append(
            f"【骑行距离】7天:{total_km:.0f}km / 日均:{daily_km:.1f}km | "
            f"百公里电耗:{energy100:.1f}kWh | 月预估用电:{monthly_e:.0f}度"
        )

        max_spd  = row.get('近7d最高速度_kmh', 0)
        p50_spd  = row.get('近7d_P50骑行速度_kmh', 0)
        p90_spd  = row.get('近7d_P90骑行速度_kmh', 0)
        night_spd = row.get('近7d夜间骑行均速_kmh', 0)
        night_risk = row.get('夜间高速风险', '低')
        spd_line = (
            f"【速度画像】P50:{p50_spd:.0f}km/h / P90:{p90_spd:.0f}km/h / 峰值:{max_spd:.0f}km/h"
        )
        if night_spd > 0:
            spd_line += f" | 夜间均速:{night_spd:.0f}km/h（风险:{night_risk}）"
        parts.append(spd_line)

        avg_cur = row.get('近7d平均骑行电流_A', 0)
        max_cur = row.get('近7d最大电流_A', 0)
        over80  = row.get('近7d电流超80A总次数', 0)
        over100 = row.get('近7d超100A连续总次数', 0)
        cur_line = f"【电流画像】均值:{avg_cur:.1f}A / 峰值:{max_cur:.0f}A"
        if over80 > 0:
            cur_line += f" | 超80A:{over80:.0f}次"
        if over100 > 0:
            cur_line += f" | 超100A连续:{over100:.0f}次"
        parts.append(cur_line)

        min_soc    = row.get('近7d最低SOC', 100)
        avg_soc    = row.get('近7d平均骑行SOC', 0)
        soc20      = row.get('近7d_SOC低于20%时长占比', 0)
        get_soc    = row.get('近7d取电时平均SOC', np.nan)
        ret_soc    = row.get('近7d还电时平均SOC', np.nan)
        soc_consume = row.get('近7d单次换电SOC消耗', 0)
        bat_line = f"【电池状况】最低SOC:{min_soc:.0f}% / 均值SOC:{avg_soc:.0f}%"
        if soc20 > 0:
            bat_line += f" | 低电量时长占比:{soc20:.1%}"
        if not np.isnan(get_soc) and not np.isnan(ret_soc):
            bat_line += f" | 取电SOC:{get_soc:.0f}% → 还电SOC:{ret_soc:.0f}%（消耗{soc_consume:.0f}%）"
        parts.append(bat_line)

        risk   = row.get('风险标签', '正常')
        strat  = row.get('策略建议', '维持服务')
        score  = row.get('重评分数', 0)
        parts.append(f"【风险&策略】{risk} | {strat} | 综合评分:{score:.1f}/100")

        portraits.append('\n'.join(parts))

    df['用户完全体画像'] = portraits
    return df


def process_lifecycle_layer(target_date: Optional[str] = None) -> str:
    print("\n" + "=" * 80)
    print("L3 Lifecycle Layer - 用户生命周期层")
    print("=" * 80)

    snapshot_path = EXPORT_PATH_SNAPSHOT
    output_path = EXPORT_PATH_LIFECYCLE_7D
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(snapshot_path):
        print("快照数据文件不存在，请先运行 L2 Snapshot Layer")
        return ""

    print("\n[1/6] 加载快照数据...")
    df_snapshot = pd.read_parquet(snapshot_path)
    print(f"   加载完成：{len(df_snapshot):,} 条记录")

    if target_date:
        # 统一类型：将target_date转为datetime.date以匹配统计日期列
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        df_snapshot = df_snapshot[df_snapshot['统计日期'] <= target_date]

    print("\n[2/6] 计算7天滚动指标...")
    df_rolling = calc_rolling_7d_metrics(df_snapshot)
    if df_rolling.empty:
        print("无有效7天滚动数据")
        return ""
    print(f"   完成：{len(df_rolling):,} 个用户")

    # 输出客户形态分布统计摘要
    log_type_distribution_summary(df_rolling)

    print("\n[3/6] 计算月度用电预估...")
    df_rolling, df_attendance_detail = calc_user_monthly_attendance(df_rolling, df_snapshot)
    
    # 保存月度出勤预估明细表
    attendance_output = os.path.join(os.path.dirname(output_path), 'user_monthly_attendance_detail.csv')
    df_attendance_detail.to_csv(attendance_output, index=False, encoding='utf-8-sig')
    print(f"   ✅ 月度出勤预估明细已保存: {attendance_output} ({len(df_attendance_detail):,} 用户)")

    print("\n[3.5/6] 加载合约信息...")
    contract_map = load_user_contract_info()
    
    def get_contract_info(uid):
        info = contract_map.get(uid, {})
        return pd.Series([info.get('join_date', None), info.get('expire_date', None)])
    
    df_rolling[['首次入网日期', '合约到期时间']] = df_rolling['用户id'].apply(get_contract_info)
    
    current_dt = pd.Timestamp.now()
    join_dates = pd.to_datetime(df_rolling['首次入网日期'], errors='coerce')
    df_rolling['入网天数'] = (current_dt - join_dates).dt.days.fillna(999).astype(int)
    
    print(f"   ✅ 合约信息注入完成")

    print("\n[4/6] 动态阈值计算与用户评分...")
    baseline_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'thresholds_baseline.json')
    analyzer = DynamicBatteryAnalyzer(
        baseline_path=baseline_path,
        use_ema_update=True,  # 启用EMA自动更新基准文件
        ema_alpha=0.3
    )
    thresholds, baseline = analyzer.run(df_rolling)

    from src.pipeline.score_layer import score_and_classify
    df_scored = score_and_classify(df_rolling, thresholds, baseline)
    print(f"   评分分类完成")

    print("\n[4.5/6] 计算完全体画像衍生指标...")
    df_scored = _add_derived_portrait_metrics(df_scored)

    print("\n[5/6] 生命周期状态判定...")
    current_date = pd.Timestamp.now()
    df_scored['用户生命周期状态_7d'] = df_scored.apply(
        lambda row: determine_lifecycle_state(row, current_date), axis=1
    )

    print("\n[5.5/6] 生成完全体画像文本...")
    df_scored = _generate_full_portrait(df_scored)

    print("\n[6/6] 保存结果...")
    df_scored.to_parquet(output_path, engine='pyarrow', compression='zstd')
    print(f"   L3 保存: {output_path} ({len(df_scored):,} 用户)")

    return output_path


if __name__ == "__main__":
    process_lifecycle_layer(target_date=None)
