# -*- coding: utf-8 -*-
"""
===============================================================================
参数自动校准器 v1.0 — AI辅助参数调优
===============================================================================
职责：基于历史数据分布，自动分析并建议最优阈值，输出校准报告供人工审核。

设计原则：
  1. 只建议，不自动修改 — 所有建议需人工确认后手动应用
  2. 基于分位数 — 阈值从数据分布中来，而非拍脑袋
  3. 可追溯 — 每次校准生成带时间戳的报告，记录变更理由
  4. 渐进式 — 支持逐项审核，不必一次性全部采纳

使用方式：
    python auto_calibrate.py --days 30
    python auto_calibrate.py --days 90 --output ./calibration_report.json

校准维度：
  1. 数据完整性阈值（时间跨度、小时覆盖）
  2. 时段划分（高峰时段自动聚类）
  3. 电流异常阈值（基于全量电流分布）
  4. 改装/超速判定阈值
  5. 储能/地摊判定阈值
  6. SOC预警阈值
  7. 评分体系阈值（温度、速度、用电量）
  8. 生命周期判定阈值
===============================================================================
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import EXPORT_PATH_FACT_DAILY, DATA_OUTPUT_ROOT

CALIBRATION_OUTPUT_DIR = os.path.join(DATA_OUTPUT_ROOT, 'calibration')


def load_fact_data(days: int = 30) -> pd.DataFrame:
    """加载最近N天的fact层日数据"""
    fact_dir = EXPORT_PATH_FACT_DAILY
    if not os.path.exists(fact_dir):
        print(f"⚠️  fact数据目录不存在: {fact_dir}")
        return pd.DataFrame()

    parquet_files = sorted([
        f for f in os.listdir(fact_dir) if f.endswith('.parquet')
    ], reverse=True)

    if not parquet_files:
        print(f"⚠️  fact数据目录为空: {fact_dir}")
        return pd.DataFrame()

    files_to_load = parquet_files[:days]
    print(f"📂 加载最近 {len(files_to_load)} 天的fact数据...")

    dfs = []
    for f in files_to_load:
        try:
            df_day = pd.read_parquet(os.path.join(fact_dir, f))
            dfs.append(df_day)
        except Exception as e:
            print(f"  ⚠️  跳过 {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    print(f"  ✅ 加载完成: {len(df)} 条记录, {df['用户id'].nunique()} 个用户")
    return df


def calibrate_data_completeness(df: pd.DataFrame) -> Dict:
    """校准数据完整性阈值"""
    result = {
        'category': '数据完整性校验',
        'current': {
            'MIN_TIME_SPAN_HOURS': 20,
            'MIN_HOUR_COVERAGE': 20,
        }
    }

    if '时间跨度_h' not in df.columns or '小时覆盖数' not in df.columns:
        result['status'] = 'skipped'
        result['reason'] = '缺少必要字段（时间跨度_h、小时覆盖数）'
        return result

    time_spans = df['时间跨度_h'].dropna()
    hour_coverages = df['小时覆盖数'].dropna()

    if len(time_spans) == 0:
        result['status'] = 'skipped'
        result['reason'] = '无有效数据'
        return result

    result['distribution'] = {
        '时间跨度_h': {
            'P5': round(float(time_spans.quantile(0.05)), 1),
            'P10': round(float(time_spans.quantile(0.10)), 1),
            'P25': round(float(time_spans.quantile(0.25)), 1),
            'P50': round(float(time_spans.quantile(0.50)), 1),
            'mean': round(float(time_spans.mean()), 1),
            '样本数': len(time_spans),
        },
        '小时覆盖数': {
            'P5': round(float(hour_coverages.quantile(0.05)), 1),
            'P10': round(float(hour_coverages.quantile(0.10)), 1),
            'P25': round(float(hour_coverages.quantile(0.25)), 1),
            'P50': round(float(hour_coverages.quantile(0.50)), 1),
            'mean': round(float(hour_coverages.mean()), 1),
            '样本数': len(hour_coverages),
        }
    }

    suggested_span = max(18, round(float(time_spans.quantile(0.05)), 0))
    suggested_coverage = max(18, round(float(hour_coverages.quantile(0.05)), 0))

    result['suggested'] = {
        'MIN_TIME_SPAN_HOURS': int(suggested_span),
        'MIN_HOUR_COVERAGE': int(suggested_coverage),
    }

    result['analysis'] = (
        f"当前阈值（时间跨度≥20h, 小时覆盖≥20）会排除约 "
        f"{round(float((time_spans < 20).mean()) * 100, 1)}% 的时间跨度数据 和 "
        f"{round(float((hour_coverages < 20).mean()) * 100, 1)}% 的小时覆盖数据。"
        f"建议基于P5分位数调整，可减少误排除。"
    )
    result['status'] = 'ready'
    return result


def calibrate_peak_hours(df: pd.DataFrame) -> Dict:
    """校准高峰时段划分（基于骑行时间密度）"""
    result = {
        'category': '时段划分',
        'current': {
            '午间高峰': [11, 12, 13],
            '晚间高峰': [17, 18, 19],
            '夜间': [20, 21, 22, 23, 0, 1, 2, 3, 4, 5],
        }
    }

    hour_cols = [c for c in df.columns if '骑行时长' in c and '小时' in c and 'h' in c]
    riding_hour_col = None
    for col in ['骑行时长_小时', '总骑行时长(小时)', '骑行时长(h)']:
        if col in df.columns:
            riding_hour_col = col
            break

    if riding_hour_col is None:
        result['status'] = 'skipped'
        result['reason'] = '缺少骑行时长字段，无法按小时分析骑行密度'
        return result

    riding_data = df[riding_hour_col].dropna()
    if len(riding_data) == 0:
        result['status'] = 'skipped'
        result['reason'] = '无有效骑行数据'
        return result

    result['distribution'] = {
        '骑行时长_小时': {
            'P25': round(float(riding_data.quantile(0.25)), 2),
            'P50': round(float(riding_data.quantile(0.50)), 2),
            'P75': round(float(riding_data.quantile(0.75)), 2),
            'P90': round(float(riding_data.quantile(0.90)), 2),
            'mean': round(float(riding_data.mean()), 2),
            '样本数': len(riding_data),
        }
    }

    result['suggested'] = result['current'].copy()
    result['analysis'] = (
        "时段划分依赖小时级骑行密度数据，当前fact层未输出逐小时骑行分布。"
        "建议在fact层增加'骑行时长_逐小时'字段后重新校准。"
        "当前阈值保持不变。"
    )
    result['status'] = 'partial'
    return result


def calibrate_current_thresholds(df: pd.DataFrame) -> Dict:
    """校准电流异常阈值"""
    result = {
        'category': '电流阈值体系',
        'current': {
            'NORMAL_CURRENT_THRESHOLD': 60,
            'HIGH_CURRENT_THRESHOLD': 80,
            'OVER_CURRENT_THRESHOLD': 100,
            'ABNORMAL_MAX_CURRENT': 80,
            'ABNORMAL_CV': 1.5,
            'ABNORMAL_AVG_CURRENT': 35,
        }
    }

    current_cols = [c for c in df.columns if '电流' in c and ('平均' in c or 'avg' in c.lower())]
    max_current_cols = [c for c in df.columns if '最大电流' in c or 'max_current' in c.lower()]

    avg_current_col = None
    for col in ['平均电流', '骑行平均电流', 'avg_current']:
        if col in df.columns:
            avg_current_col = col
            break
    if avg_current_col is None and current_cols:
        avg_current_col = current_cols[0]

    max_current_col = None
    for col in ['最大电流', '骑行最大电流', 'max_current']:
        if col in df.columns:
            max_current_col = col
            break
    if max_current_col is None and max_current_cols:
        max_current_col = max_current_cols[0]

    if avg_current_col is None and max_current_col is None:
        result['status'] = 'skipped'
        result['reason'] = '缺少电流相关字段'
        return result

    distributions = {}
    if avg_current_col:
        avg_data = df[avg_current_col].dropna()
        avg_data = avg_data[(avg_data > 0) & (avg_data < 200)]
        distributions['平均电流'] = {
            'P50': round(float(avg_data.quantile(0.50)), 1),
            'P75': round(float(avg_data.quantile(0.75)), 1),
            'P90': round(float(avg_data.quantile(0.90)), 1),
            'P95': round(float(avg_data.quantile(0.95)), 1),
            'P99': round(float(avg_data.quantile(0.99)), 1),
            'mean': round(float(avg_data.mean()), 1),
            '样本数': len(avg_data),
        }

    if max_current_col:
        max_data = df[max_current_col].dropna()
        max_data = max_data[(max_data > 0) & (max_data < 200)]
        distributions['最大电流'] = {
            'P50': round(float(max_data.quantile(0.50)), 1),
            'P75': round(float(max_data.quantile(0.75)), 1),
            'P90': round(float(max_data.quantile(0.90)), 1),
            'P95': round(float(max_data.quantile(0.95)), 1),
            'P99': round(float(max_data.quantile(0.99)), 1),
            'mean': round(float(max_data.mean()), 1),
            '样本数': len(max_data),
        }

    result['distribution'] = distributions

    if max_current_col:
        max_data = df[max_current_col].dropna()
        max_data = max_data[(max_data > 0) & (max_data < 200)]
        p90 = float(max_data.quantile(0.90))
        p95 = float(max_data.quantile(0.95))
        p99 = float(max_data.quantile(0.99))

        result['suggested'] = {
            'NORMAL_CURRENT_THRESHOLD': int(round(p90, -1)),
            'HIGH_CURRENT_THRESHOLD': int(round(p95, -1)),
            'OVER_CURRENT_THRESHOLD': max(80, int(round(p99, -1))),
            'ABNORMAL_MAX_CURRENT': int(round(p95, -1)),
            'ABNORMAL_CV': 1.5,
            'ABNORMAL_AVG_CURRENT': int(round(float(avg_data.quantile(0.90)), -1)) if avg_current_col else 35,
        }

        over_pct = round(float((max_data >= 100).mean()) * 100, 2)
        result['analysis'] = (
            f"当前超100A阈值下，约 {over_pct}% 的记录被标记为过流。"
            f"最大电流P90={p90:.0f}A, P95={p95:.0f}A, P99={p99:.0f}A。"
            f"建议将NORMAL阈值对齐P90，HIGH对齐P95，OVER对齐P99。"
        )
    else:
        result['suggested'] = result['current'].copy()
        result['analysis'] = "缺少最大电流字段，无法校准，保持当前阈值。"

    result['status'] = 'ready'
    return result


def calibrate_modify_thresholds(df: pd.DataFrame) -> Dict:
    """校准改装/超速判定阈值"""
    result = {
        'category': '改装/超速判定',
        'current': {
            'MODIFY_SPEED_THRESHOLD': 50,
            'MODIFY_CURRENT_THRESHOLD': 24,
        }
    }

    speed_col = None
    for col in ['最大速度', '骑行最大速度', 'max_speed']:
        if col in df.columns:
            speed_col = col
            break

    current_col = None
    for col in ['骑行平均电流', '平均电流', 'avg_current']:
        if col in df.columns:
            current_col = col
            break

    if speed_col is None and current_col is None:
        result['status'] = 'skipped'
        result['reason'] = '缺少速度和电流字段'
        return result

    distributions = {}
    if speed_col:
        speed_data = df[speed_col].dropna()
        speed_data = speed_data[(speed_data > 0) & (speed_data < 150)]
        distributions['最大速度'] = {
            'P50': round(float(speed_data.quantile(0.50)), 1),
            'P75': round(float(speed_data.quantile(0.75)), 1),
            'P90': round(float(speed_data.quantile(0.90)), 1),
            'P95': round(float(speed_data.quantile(0.95)), 1),
            'P99': round(float(speed_data.quantile(0.99)), 1),
            'mean': round(float(speed_data.mean()), 1),
            '样本数': len(speed_data),
        }

    if current_col:
        cur_data = df[current_col].dropna()
        cur_data = cur_data[(cur_data > 0) & (cur_data < 200)]
        distributions['平均电流'] = {
            'P50': round(float(cur_data.quantile(0.50)), 1),
            'P75': round(float(cur_data.quantile(0.75)), 1),
            'P90': round(float(cur_data.quantile(0.90)), 1),
            'P95': round(float(cur_data.quantile(0.95)), 1),
            'mean': round(float(cur_data.mean()), 1),
            '样本数': len(cur_data),
        }

    result['distribution'] = distributions

    suggested_speed = 50
    suggested_current = 24
    if speed_col:
        speed_data = df[speed_col].dropna()
        speed_data = speed_data[(speed_data > 0) & (speed_data < 150)]
        suggested_speed = int(round(float(speed_data.quantile(0.90)), -1))
        suggested_speed = max(40, min(70, suggested_speed))

    if current_col:
        cur_data = df[current_col].dropna()
        cur_data = cur_data[(cur_data > 0) & (cur_data < 200)]
        suggested_current = int(round(float(cur_data.quantile(0.75)), -1))
        suggested_current = max(20, min(40, suggested_current))

    result['suggested'] = {
        'MODIFY_SPEED_THRESHOLD': suggested_speed,
        'MODIFY_CURRENT_THRESHOLD': suggested_current,
    }

    if speed_col:
        speed_data = df[speed_col].dropna()
        speed_data = speed_data[(speed_data > 0) & (speed_data < 150)]
        over_pct = round(float((speed_data >= 50).mean()) * 100, 2)
        result['analysis'] = (
            f"当前速度阈值50km/h下，约 {over_pct}% 的记录被标记为超速。"
            f"最大速度P90={float(speed_data.quantile(0.90)):.0f}km/h。"
            f"建议将速度阈值对齐P90={suggested_speed}km/h。"
        )
    else:
        result['analysis'] = "缺少速度字段，保持当前阈值。"

    result['status'] = 'ready'
    return result


def calibrate_storage_thresholds(df: pd.DataFrame) -> Dict:
    """校准储能/地摊判定阈值"""
    result = {
        'category': '储能/地摊判定',
        'current': {
            'STORAGE_MAX_DISTANCE_KM': 1.0,
            'STORAGE_MAX_AVG_SPEED_KMH': 3.0,
            'STORAGE_DISCHARGE_RATIO_THRESHOLD': 3.0,
            'STORAGE_MIN_DISCHARGE_HOUR': 2.0,
        }
    }

    distance_col = None
    for col in ['行驶距离', '总行驶距离', 'total_distance']:
        if col in df.columns:
            distance_col = col
            break

    speed_col = None
    for col in ['平均速度', '骑行平均速度', 'avg_speed']:
        if col in df.columns:
            speed_col = col
            break

    discharge_col = None
    for col in ['总放电时长(小时)', '放电时长', 'discharge_hours']:
        if col in df.columns:
            discharge_col = col
            break

    if distance_col is None:
        result['status'] = 'skipped'
        result['reason'] = '缺少行驶距离字段'
        return result

    dist_data = df[distance_col].dropna()
    dist_data = dist_data[dist_data >= 0]

    distributions = {
        '行驶距离_km': {
            'P5': round(float(dist_data.quantile(0.05)), 2),
            'P10': round(float(dist_data.quantile(0.10)), 2),
            'P25': round(float(dist_data.quantile(0.25)), 2),
            'P50': round(float(dist_data.quantile(0.50)), 2),
            'mean': round(float(dist_data.mean()), 2),
            '样本数': len(dist_data),
        }
    }

    if speed_col:
        spd_data = df[speed_col].dropna()
        spd_data = spd_data[spd_data >= 0]
        distributions['平均速度_kmh'] = {
            'P5': round(float(spd_data.quantile(0.05)), 1),
            'P10': round(float(spd_data.quantile(0.10)), 1),
            'P25': round(float(spd_data.quantile(0.25)), 1),
            'P50': round(float(spd_data.quantile(0.50)), 1),
            'mean': round(float(spd_data.mean()), 1),
            '样本数': len(spd_data),
        }

    if discharge_col:
        dis_data = df[discharge_col].dropna()
        dis_data = dis_data[dis_data >= 0]
        distributions['放电时长_h'] = {
            'P25': round(float(dis_data.quantile(0.25)), 1),
            'P50': round(float(dis_data.quantile(0.50)), 1),
            'P75': round(float(dis_data.quantile(0.75)), 1),
            'P90': round(float(dis_data.quantile(0.90)), 1),
            'mean': round(float(dis_data.mean()), 1),
            '样本数': len(dis_data),
        }

    result['distribution'] = distributions

    p5_dist = float(dist_data.quantile(0.05))
    suggested_dist = round(max(0.5, p5_dist * 0.5), 1)

    result['suggested'] = {
        'STORAGE_MAX_DISTANCE_KM': suggested_dist,
        'STORAGE_MAX_AVG_SPEED_KMH': 3.0,
        'STORAGE_DISCHARGE_RATIO_THRESHOLD': 3.0,
        'STORAGE_MIN_DISCHARGE_HOUR': 2.0,
    }

    result['analysis'] = (
        f"行驶距离P5={p5_dist:.2f}km。储能/地摊用户特征为极短距离+极低速度，"
        f"建议MAX_DISTANCE={suggested_dist}km（P5的一半）。"
        f"速度和放电比阈值暂保持不变（需结合地摊用户标注数据校准）。"
    )
    result['status'] = 'ready'
    return result


def calibrate_soc_thresholds(df: pd.DataFrame) -> Dict:
    """校准SOC预警阈值"""
    result = {
        'category': 'SOC预警',
        'current': {
            'SOC_LOW_WARNING': 20,
            'SOC_CRITICAL': 10,
        }
    }

    soc_cols = [c for c in df.columns if 'SOC' in c.upper() or 'soc' in c.lower()]
    min_soc_col = None
    for col in ['最低SOC', 'min_soc', '最小SOC']:
        if col in df.columns:
            min_soc_col = col
            break
    if min_soc_col is None and soc_cols:
        for col in soc_cols:
            if '最低' in col or 'min' in col.lower():
                min_soc_col = col
                break

    if min_soc_col is None:
        result['status'] = 'skipped'
        result['reason'] = '缺少SOC相关字段'
        return result

    soc_data = df[min_soc_col].dropna()
    soc_data = soc_data[(soc_data >= 0) & (soc_data <= 100)]

    if len(soc_data) == 0:
        result['status'] = 'skipped'
        result['reason'] = '无有效SOC数据'
        return result

    result['distribution'] = {
        '最低SOC': {
            'P5': round(float(soc_data.quantile(0.05)), 1),
            'P10': round(float(soc_data.quantile(0.10)), 1),
            'P25': round(float(soc_data.quantile(0.25)), 1),
            'P50': round(float(soc_data.quantile(0.50)), 1),
            'mean': round(float(soc_data.mean()), 1),
            '样本数': len(soc_data),
        }
    }

    p10 = float(soc_data.quantile(0.10))
    p5 = float(soc_data.quantile(0.05))

    result['suggested'] = {
        'SOC_LOW_WARNING': max(15, int(round(p10, -1))),
        'SOC_CRITICAL': max(5, int(round(p5, -1))),
    }

    below_20 = round(float((soc_data < 20).mean()) * 100, 2)
    below_10 = round(float((soc_data < 10).mean()) * 100, 2)
    result['analysis'] = (
        f"当前阈值下，约 {below_20}% 的记录SOC<20（低电量预警），"
        f"{below_10}% 的记录SOC<10（严重低电量）。"
        f"最低SOC P10={p10:.0f}%, P5={p5:.0f}%。"
    )
    result['status'] = 'ready'
    return result


def calibrate_score_thresholds(df: pd.DataFrame) -> Dict:
    """校准评分体系阈值"""
    result = {
        'category': '评分体系阈值',
        'current': {
            'MONTHLY_ENERGY_LOSS_THRESHOLD': 150,
            'MONTHLY_ENERGY_VIOLENT_THRESHOLD': 250,
            'TEMP_HIGH_THRESHOLD': 55,
            'TEMP_EXTREME_THRESHOLD': 70,
            'SPEED_HIGH_THRESHOLD': 50,
            'SPEED_EXTREME_THRESHOLD': 60,
            'SPEED_VIOLENT_THRESHOLD': 80,
        }
    }

    energy_col = None
    for col in ['单合约月度用电度数预估_kWh', '月度预估用电', 'monthly_energy']:
        if col in df.columns:
            energy_col = col
            break

    temp_col = None
    for col in ['最高温度', 'max_temp', '最高温度_7d']:
        if col in df.columns:
            temp_col = col
            break

    speed_col = None
    for col in ['最大速度_7d', '骑行最大速度_7d', 'max_speed_7d']:
        if col in df.columns:
            speed_col = col
            break

    distributions = {}
    suggestions = result['current'].copy()

    if energy_col:
        energy_data = df[energy_col].dropna()
        energy_data = energy_data[energy_data >= 0]
        distributions['月度预估用电_kWh'] = {
            'P50': round(float(energy_data.quantile(0.50)), 1),
            'P75': round(float(energy_data.quantile(0.75)), 1),
            'P90': round(float(energy_data.quantile(0.90)), 1),
            'P95': round(float(energy_data.quantile(0.95)), 1),
            'P99': round(float(energy_data.quantile(0.99)), 1),
            'mean': round(float(energy_data.mean()), 1),
            '样本数': len(energy_data),
        }
        suggestions['MONTHLY_ENERGY_LOSS_THRESHOLD'] = int(round(float(energy_data.quantile(0.75)), -1))
        suggestions['MONTHLY_ENERGY_VIOLENT_THRESHOLD'] = int(round(float(energy_data.quantile(0.95)), -1))

    if temp_col:
        temp_data = df[temp_col].dropna()
        temp_data = temp_data[(temp_data > 0) & (temp_data < 150)]
        distributions['最高温度'] = {
            'P50': round(float(temp_data.quantile(0.50)), 1),
            'P75': round(float(temp_data.quantile(0.75)), 1),
            'P90': round(float(temp_data.quantile(0.90)), 1),
            'P95': round(float(temp_data.quantile(0.95)), 1),
            'mean': round(float(temp_data.mean()), 1),
            '样本数': len(temp_data),
        }
        suggestions['TEMP_HIGH_THRESHOLD'] = int(round(float(temp_data.quantile(0.90)), -1))
        suggestions['TEMP_EXTREME_THRESHOLD'] = int(round(float(temp_data.quantile(0.95)), -1))

    if speed_col:
        spd_data = df[speed_col].dropna()
        spd_data = spd_data[(spd_data > 0) & (spd_data < 150)]
        distributions['最大速度_7d'] = {
            'P50': round(float(spd_data.quantile(0.50)), 1),
            'P75': round(float(spd_data.quantile(0.75)), 1),
            'P90': round(float(spd_data.quantile(0.90)), 1),
            'P95': round(float(spd_data.quantile(0.95)), 1),
            'mean': round(float(spd_data.mean()), 1),
            '样本数': len(spd_data),
        }
        suggestions['SPEED_HIGH_THRESHOLD'] = int(round(float(spd_data.quantile(0.75)), -1))
        suggestions['SPEED_EXTREME_THRESHOLD'] = int(round(float(spd_data.quantile(0.90)), -1))
        suggestions['SPEED_VIOLENT_THRESHOLD'] = int(round(float(spd_data.quantile(0.95)), -1))

    result['distribution'] = distributions
    result['suggested'] = suggestions

    if distributions:
        result['analysis'] = (
            "评分阈值基于全量用户分布的分位数建议。"
            "LOSS线对齐P75（约75%用户在此之下），VIOLENT线对齐P95（极端用户）。"
            "温度和速度阈值同理，HIGH=P75, EXTREME=P90, VIOLENT=P95。"
        )
    else:
        result['analysis'] = "缺少评分相关字段，保持当前阈值。"

    result['status'] = 'ready' if distributions else 'skipped'
    return result


def calibrate_lifecycle_thresholds(df: pd.DataFrame) -> Dict:
    """校准生命周期判定阈值"""
    result = {
        'category': '生命周期判定',
        'current': {
            'LONG_TERM_OFFLINE_THRESHOLD_DAYS': 7,
            'MIN_DAYS_FOR_ACTIVE': 5,
            'MIN_RECORD_DAYS_FOR_ESTIMATE': 7,
            'FULL_RECORD_DAYS_FOR_CUMULATIVE': 30,
            'DEFAULT_MONTH_DAYS': 26,
        }
    }

    attendance_col = None
    for col in ['近7d出勤天数', '出勤天数_7d', 'attendance_days']:
        if col in df.columns:
            attendance_col = col
            break

    if attendance_col is None:
        result['status'] = 'skipped'
        result['reason'] = '缺少出勤天数字段'
        return result

    att_data = df[attendance_col].dropna()
    att_data = att_data[(att_data >= 0) & (att_data <= 7)]

    result['distribution'] = {
        '近7d出勤天数': {
            '0天占比': round(float((att_data == 0).mean()) * 100, 1),
            '1-2天占比': round(float(((att_data >= 1) & (att_data <= 2)).mean()) * 100, 1),
            '3-4天占比': round(float(((att_data >= 3) & (att_data <= 4)).mean()) * 100, 1),
            '5-7天占比': round(float((att_data >= 5).mean()) * 100, 1),
            'mean': round(float(att_data.mean()), 2),
            '样本数': len(att_data),
        }
    }

    result['suggested'] = result['current'].copy()
    result['analysis'] = (
        f"近7天出勤天数分布：均值={att_data.mean():.1f}天。"
        f"当前活跃阈值≥5天，离线阈值≥7天。"
        f"建议根据业务需求调整（如需更严格则提高活跃阈值）。"
    )
    result['status'] = 'ready'
    return result


def generate_calibration_report(
    results: List[Dict],
    df_meta: Dict,
    output_path: str = None
) -> str:
    """生成校准报告JSON"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_path is None:
        os.makedirs(CALIBRATION_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            CALIBRATION_OUTPUT_DIR,
            f'calibration_{timestamp}.json'
        )

    ready_count = sum(1 for r in results if r.get('status') == 'ready')
    partial_count = sum(1 for r in results if r.get('status') == 'partial')
    skipped_count = sum(1 for r in results if r.get('status') == 'skipped')

    report = {
        'meta': {
            'version': 'v1.0',
            'generated_at': datetime.now().isoformat(),
            'data_summary': df_meta,
            'total_categories': len(results),
            'ready': ready_count,
            'partial': partial_count,
            'skipped': skipped_count,
        },
        'results': results,
        'instructions': {
            'review_process': [
                '1. 逐项查看 suggested 建议值',
                '2. 对比 current 当前值与 distribution 数据分布',
                '3. 阅读 analysis 分析说明，理解建议依据',
                '4. 决定采纳/调整/拒绝，手动修改对应源码文件',
                '5. 修改后重新运行管线验证效果',
            ],
            'safety_notes': [
                '所有建议仅供参考，不自动修改代码',
                '阈值调整后建议先在测试环境验证',
                '极端阈值（如P99）可能导致大量用户被误判',
                '建议优先调整数据完整性和评分阈值，电流/速度阈值谨慎调整',
            ]
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return output_path


def print_summary(results: List[Dict], report_path: str):
    """打印校准摘要"""
    print("\n" + "=" * 70)
    print("📊 参数校准报告摘要")
    print("=" * 70)

    for r in results:
        status_icon = {'ready': '✅', 'partial': '⚠️', 'skipped': '⏭️'}.get(r.get('status'), '❓')
        category = r['category']
        status = r.get('status', 'unknown')

        print(f"\n{status_icon} [{category}] (状态: {status})")

        if status == 'skipped':
            print(f"   原因: {r.get('reason', '未知')}")
            continue

        current = r.get('current', {})
        suggested = r.get('suggested', {})

        changes = []
        for key in current:
            if key in suggested and current[key] != suggested[key]:
                changes.append(f"  {key}: {current[key]} → {suggested[key]}")

        if changes:
            print("  建议调整:")
            for c in changes:
                print(c)
        else:
            print("  当前阈值合理，无需调整")

        analysis = r.get('analysis', '')
        if analysis:
            print(f"  分析: {analysis[:120]}...")

    print(f"\n📁 完整报告: {report_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='参数自动校准器')
    parser.add_argument('--days', type=int, default=30,
                        help='加载最近N天的fact数据 (默认: 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='校准报告输出路径 (默认: auto)')
    parser.add_argument('--categories', type=str, default='all',
                        help='校准类别，逗号分隔 (默认: all)')
    args = parser.parse_args()

    print("=" * 70)
    print("🔧 参数自动校准器 v1.0")
    print(f"   数据范围: 最近 {args.days} 天")
    print(f"   校准类别: {args.categories}")
    print("=" * 70)

    df = load_fact_data(days=args.days)
    if df.empty:
        print("❌ 无法加载数据，校准终止")
        return

    df_meta = {
        'total_records': len(df),
        'unique_users': int(df['用户id'].nunique()),
        'date_range': f"{df['统计日期'].min()} ~ {df['统计日期'].max()}" if '统计日期' in df.columns else 'N/A',
        'days_loaded': args.days,
    }

    requested = set(args.categories.split(',')) if args.categories != 'all' else {'all'}

    calibrators = [
        ('data_completeness', calibrate_data_completeness),
        ('peak_hours', calibrate_peak_hours),
        ('current', calibrate_current_thresholds),
        ('modify', calibrate_modify_thresholds),
        ('storage', calibrate_storage_thresholds),
        ('soc', calibrate_soc_thresholds),
        ('score', calibrate_score_thresholds),
        ('lifecycle', calibrate_lifecycle_thresholds),
    ]

    results = []
    for name, func in calibrators:
        if 'all' not in requested and name not in requested:
            continue
        print(f"\n🔍 校准 [{name}] ...")
        try:
            r = func(df)
            results.append(r)
        except Exception as e:
            print(f"  ❌ 校准失败: {e}")
            results.append({
                'category': name,
                'status': 'error',
                'reason': str(e),
            })

    report_path = generate_calibration_report(results, df_meta, args.output)
    print_summary(results, report_path)

    print(f"\n✅ 校准完成！报告已保存至: {report_path}")
    print("⚠️  请人工审核报告后再修改源码中的阈值参数。")


if __name__ == '__main__':
    main()