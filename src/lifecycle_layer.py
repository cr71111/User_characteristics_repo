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
import warnings
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import BASE_EXPORT_PATH, EXPORT_PATH_SNAPSHOT, EXPORT_PATH_LIFECYCLE_7D
from dynamic_thresholds import DynamicBatteryAnalyzer

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

    valid_days = len(group)

    res = {
        '用户id': user_id,
        '统计日期': latest_date.strftime('%Y-%m-%d'),
        '近7天有数据天数': valid_days
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
        if raw_col in group.columns:
            res[roll_col] = round(_weighted_avg(group[raw_col], group['weight']), 2)
        else:
            res[roll_col] = 0

    if '高峰骑行占比' in group.columns and '当日总骑行时长_h' in group.columns:
        res['近7d高峰骑行占比'] = round(_weighted_ratio(group['高峰骑行占比'], group['当日总骑行时长_h'], group['weight']), 2)
    else:
        res['近7d高峰骑行占比'] = 0

    soc20_col = '当日SOC低于20%时长占比' if '当日SOC低于20%时长占比' in group.columns else 'SOC低于20%时长占比_当日'
    if soc20_col in group.columns and '当日总骑行时长_h' in group.columns:
        res['近7d_SOC低于20%时长占比'] = round(_weighted_ratio(group[soc20_col], group['当日总骑行时长_h'], group['weight']), 4)
    else:
        res['近7d_SOC低于20%时长占比'] = 0

    soc10_col = '当日SOC低于10%时长占比' if '当日SOC低于10%时长占比' in group.columns else 'SOC低于10%时长占比_当日'
    if soc10_col in group.columns and '当日总骑行时长_h' in group.columns:
        res['近7d_SOC低于10%时长占比'] = round(_weighted_ratio(group[soc10_col], group['当日总骑行时长_h'], group['weight']), 4)
    else:
        res['近7d_SOC低于10%时长占比'] = 0

    max_cols = [
        ('当日最高速度_kmh', '近7d最高速度_kmh'),
        ('当日最大电流_A', '近7d最大电流_A'),
        ('当日最高温度_℃', '近7d最高温度_℃'),
        ('最大单合约活动半径_km', '近7d最大活动半径_km'),
        ('R90活动半径_km', '近7d_R90活动半径_km'),
        ('最大凸包覆盖面积_km2', '近7d凸包覆盖面积_km2'),
        ('最大单次出行距离_km', '近7d最大单次出行距离_km'),
        ('最晚骑行时刻_h', '近7d最晚骑行时刻_h'),
    ]
    for raw_col, roll_col in max_cols:
        if raw_col in group.columns:
            val = group[raw_col].max()
            res[roll_col] = round(val, 2) if pd.notna(val) else 0
        else:
            res[roll_col] = 0

    min_cols = [
        ('当日最低SOC', '近7d最低SOC'),
        ('最早骑行时刻_h', '近7d最早骑行时刻_h'),
    ]
    for raw_col, roll_col in min_cols:
        if raw_col in group.columns:
            val = group[raw_col].min()
            res[roll_col] = round(val, 2) if pd.notna(val) else 100
        else:
            res[roll_col] = 100

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
        if raw_col in group.columns:
            cum_val = group[raw_col].sum()
            res[cum_col] = round(cum_val, 2) if pd.notna(cum_val) else 0
            res[avg_col] = round(res[cum_col] / valid_days, 2) if valid_days > 0 else 0
        else:
            res[cum_col] = 0
            res[avg_col] = 0

    if '当日是否出勤' in group.columns:
        attendance_sum = group['当日是否出勤'].sum()
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

    if '客户形态_综合' in group.columns:
        type_counts = group['客户形态_综合'].value_counts()
        if len(type_counts) > 0:
            res['客户形态_综合_7d'] = type_counts.index[0]
        else:
            res['客户形态_综合_7d'] = "数据不足"
    else:
        res['客户形态_综合_7d'] = "数据不足"

    type_desc_7d = ""
    current_type_7d = res.get('客户形态_综合_7d', '数据不足')
    avg_speed_7d = res.get('近7d平均骑行速度_kmh', 0)
    max_speed_7d = res.get('近7d最高速度_kmh', 0)
    avg_current_7d = res.get('近7d平均骑行电流_A', 0)
    avg_ride_hour_7d = res.get('近7d单合约日均骑行时长_h', 0)
    peak_ratio_7d = res.get('近7d高峰骑行占比', 0)
    avg_mileage_7d = res.get('近7d单合约日均行驶里程_km', 0)
    avg_idle_discharge_7d = res.get('近7d单合约日均怠速放电_h', 0)

    if current_type_7d == "改装/超速车":
        type_desc_7d = f"行驶特征异常（近7天最高速度{max_speed_7d}km/h，平均骑行电流{avg_current_7d}A），远超普通两轮车水平，存在改装或超速嫌疑"
    elif current_type_7d == "地摊/储能":
        type_desc_7d = f"非移动用电特征明显（近7天日均骑行{avg_ride_hour_7d}小时，日均怠速放电{avg_idle_discharge_7d}小时），放电以静止状态为主，疑似地摊供电或储能场景"
    elif current_type_7d == "专送骑手":
        type_desc_7d = f"工作特征显著（近7天日均骑行{avg_ride_hour_7d:.1f}小时，高峰骑行占比{peak_ratio_7d*100:.0f}%），工作时长稳定且午晚高峰高度活跃，符合专送骑手画像"
    elif current_type_7d == "众包骑手":
        type_desc_7d = f"具有兼职骑手特征（近7天日均骑行{avg_ride_hour_7d:.1f}小时，高峰骑行占比{peak_ratio_7d*100:.0f}%），高峰时段有一定活跃度"
    elif current_type_7d == "标准骑手":
        type_desc_7d = f"骑行行为规律（近7天日均里程{avg_mileage_7d}km，日均骑行{avg_ride_hour_7d:.1f}小时），属于标准日常使用场景"
    elif current_type_7d == "普通骑手":
        type_desc_7d = f"有常规骑行行为（近7天日均里程{avg_mileage_7d}km，日均骑行{avg_ride_hour_7d:.1f}小时），但不符合特定骑手标签特征"
    else:
        type_desc_7d = "无有效骑行数据或数据量不足，无法判定具体使用场景"
    
    res['客户形态_综合_7d说明'] = type_desc_7d

    latest_record_date = pd.to_datetime(group['统计日期'].max())
    days_since_last_seen = (latest_date - latest_record_date).days
    
    if days_since_last_seen >= LONG_TERM_OFFLINE_THRESHOLD_DAYS:
        res['设备状态监控'] = f"异常(离线{days_since_last_seen}天)"
    elif days_since_last_seen >= 1:
        res['设备状态监控'] = f"暂离线({days_since_last_seen}天)"
    else:
        res['设备状态监控'] = "正常在线"

    return res


def calc_user_monthly_attendance(df_rolling: pd.DataFrame) -> pd.DataFrame:
    df = df_rolling.copy()
    df['月度预估工作天数'] = (df['近7d出勤天数'] * 30 / 7).round(0)
    df['单合约月度用电度数预估_kWh'] = (
        df['近7d单合约日均骑行时长_h'] * df['近7d百公里电耗_kWh'] * 0.5 * (30 / 7)
    ).round(2)
    return df


def determine_lifecycle_state(row: dict, current_date: pd.Timestamp) -> str:
    expire_date = pd.to_datetime(row.get('合约到期时间'), errors='coerce')
    if pd.notna(expire_date) and current_date > expire_date:
        return '已到期'
    
    join_date = pd.to_datetime(row.get('首次入网日期'), errors='coerce')
    if pd.notna(join_date):
        days_since_join = (current_date - join_date).days
        if days_since_join <= 3:
            return '新用户'
    
    attendance_days = row.get('近7d出勤天数', 0)
    has_data_days = row.get('近7天有数据天数', 0)
    
    if attendance_days == 0:
        if has_data_days <= (7 - 7):
            return '流失'
        else:
            return '沉默'
    elif attendance_days >= 5:
        return '活跃'
    else:
        return '轻度活跃'


def _add_derived_portrait_metrics(df: pd.DataFrame) -> pd.DataFrame:

    r90 = df.get('近7d_R90活动半径_km', pd.Series(0, index=df.index)).fillna(0)
    r95 = df.get('近7d最大活动半径_km', pd.Series(1, index=df.index)).replace(0, 1)
    df['近7d活动集中度'] = (r90 / r95).clip(0, 1).round(3)

    offpeak_ratio = df.get('近7d平峰换电占比', pd.Series(0, index=df.index)).fillna(0)
    night_ratio   = df.get('近7d深夜换电占比', pd.Series(0, index=df.index)).fillna(0)

    swap_friendly = (offpeak_ratio * 0.6 + night_ratio * 0.4) * 20
    df['近7d换电友好分'] = swap_friendly.clip(0, 20).round(1)

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
            parts.append(
                f"【骑行时间】惯用时段:{pattern} | "
                f"首次上路:{earliest:02d}:xx / 最晚收车:{latest:02d}:xx | "
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


def process_lifecycle_layer(target_date: Optional[str] = None, base_path: str = BASE_EXPORT_PATH) -> str:
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
        df_snapshot = df_snapshot[df_snapshot['统计日期'] <= target_date]

    print("\n[2/6] 计算7天滚动指标...")
    df_rolling = calc_rolling_7d_metrics(df_snapshot)
    if df_rolling.empty:
        print("无有效7天滚动数据")
        return ""
    print(f"   完成：{len(df_rolling):,} 个用户")

    print("\n[3/6] 计算月度用电预估...")
    df_rolling = calc_user_monthly_attendance(df_rolling)

    print("\n[4/6] 动态阈值计算与用户评分...")
    baseline_path = os.path.join(os.path.dirname(__file__), 'thresholds_baseline.json')
    analyzer = DynamicBatteryAnalyzer(baseline_path=baseline_path)
    thresholds, baseline = analyzer.run(df_rolling)

    from score_layer import score_and_classify
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
