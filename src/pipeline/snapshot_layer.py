# -*- coding: utf-8 -*-
"""
L2 Snapshot Layer - 用户日级快照层
职责：将合约日级事实聚合为用户日级快照，并做日级评分和客户形态判定。

输入：fact/daily/{date}.parquet
输出：snapshot/user_daily.parquet（追加写入，保留历史）
"""

import os
import sys
import warnings
from typing import Optional

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import EXPORT_PATH_FACT_DAILY, EXPORT_PATH_SNAPSHOT
from src.pipeline.score_common import (
    SOC_CRITICAL_RATIO_DEDUCT_THRESHOLD, SOC_LOW_RATIO_DEDUCT_THRESHOLD,
    SOC_OPTIMAL_LOWER, SOC_OPTIMAL_UPPER, SOC_OPTIMAL_BONUS,
    MAX_SOC_DEDUCT, MAX_ENERGY_DEDUCT,
    EXTREME_ENERGY_THRESHOLD, HIGH_ENERGY_THRESHOLD,
    MAX_NORMAL_BATTERY_CHANGE, EXCESS_CHANGE_DEDUCT_PER_TIME, MAX_CHANGE_DEDUCT,
    VIOLENT_CURRENT_TIMES, HIGH_LOSS_CURRENT_TIMES, OVER_CURRENT_MIN_HOUR,
)

VALID_RIDE_MIN_HOUR = 0.1
STORAGE_MIN_VALID_GPS_POINTS = 10
STORAGE_MAX_DISTANCE_KM = 1.0
STORAGE_MAX_AVG_SPEED_KMH = 3.0
STORAGE_DISCHARGE_RATIO_THRESHOLD = 3.0
STORAGE_MIN_DISCHARGE_HOUR = 2.0
STORAGE_RIDE_DURATION_RATIO = 0.1
MODIFY_SPEED_THRESHOLD = 50
MODIFY_CURRENT_THRESHOLD = 24
PATTERN_DOMINANT_RATIO = 0.5

SOC_LOW_WARNING = 20
SOC_CRITICAL = 10

NOON_PEAK_HOURS = set(range(11, 14))
EVENING_PEAK_HOURS = set(range(17, 20))


def determine_customer_type(total_riding_hours, total_distance, idle_discharge_hours, total_discharge_hours, 
                           max_speed, riding_avg_current, peak_riding_ratio, n):
    """客户形态判定（严格按照旧脚本逻辑）"""
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
        return "改装/超速车"
    elif is_storage_scene:
        return "地摊/储能"
    elif has_real_ride and total_riding_hours >= 4.0 and peak_riding_ratio >= 0.3:
        return "专送骑手"
    elif has_real_ride and total_riding_hours >= 2.0 and peak_riding_ratio >= 0.15:
        return "众包骑手"
    elif has_real_ride and total_distance <= 30 and total_riding_hours <= 2.0:
        return "标准骑手"
    elif has_real_ride:
        return "普通骑手"
    else:
        return "数据不足"


def _get_user_type(group):
    """获取用户客户形态（按优先级）"""
    priority = ["改装/超速车", "地摊/储能"]
    for t in priority:
        if t in list(group['客户形态']):
            return t
    
    contract_type_duration = group.groupby('客户形态')['骑行总耗时(小时)'].sum()
    total_duration = contract_type_duration.sum()
    if total_duration > 0 and len(contract_type_duration) > 0:
        return contract_type_duration.idxmax()
    return "数据不足"


def _get_user_level(levels):
    """获取用户等级（按优先级）"""
    priority = ["暴力", "高损耗用户", "普通用户", "良好用户", "优质用户", "无效"]
    for l in priority:
        if l in list(levels): return l
    return "无效"


def _get_level_desc(levels, descs):
    """获取用户等级说明"""
    priority = ["暴力", "高损耗用户", "普通用户", "良好用户", "优质用户", "无效"]
    temp_df = pd.DataFrame({'level': levels, 'desc': descs})
    for l in priority:
        if l in temp_df['level'].values:
            return temp_df[temp_df['level'] == l]['desc'].iloc[0]
    return "数据不足"


def _get_type_desc(types, descs):
    """获取客户形态说明"""
    priority = ["改装/超速车", "地摊/储能", "专送骑手", "众包骑手", "标准骑手", "普通骑手", "数据不足"]
    temp_df = pd.DataFrame({'type': types, 'desc': descs})
    for t in priority:
        if t in temp_df['type'].values:
            return temp_df[temp_df['type'] == t]['desc'].iloc[0]
    return "数据不足"


def _generate_level_desc(row):
    """生成用户等级说明"""
    current_level = row['用户等级']
    if current_level == "暴力":
        reasons = []
        if row['超100A连续次数'] >= VIOLENT_CURRENT_TIMES:
            reasons.append(f"超100A连续放电{row['超100A连续次数']}次，触发阈值")
        if row['超100A累计时长_h'] >= OVER_CURRENT_MIN_HOUR:
            reasons.append(f"超100A累计放电时长{row['超100A累计时长_h']}小时，触发阈值")
        if row['电流>60A次数'] >= VIOLENT_CURRENT_TIMES:
            reasons.append(f"电流超60A共{row['电流>60A次数']}次，峰值达{row['最大电流']}A")
        if row['SOC数据有效'] and row['SOC低于10%时长占比'] >= 0.5 and row['最低SOC'] <= 5:
            reasons.append(f"深度亏电严重（最低SOC{row['最低SOC']}%，SOC低于10%时长占比{row['SOC低于10%时长占比']*100:.0f}%）")
        return " | ".join(reasons) if reasons else "存在严重损害电池的行为"
    
    elif current_level == "高损耗用户":
        reasons = []
        if row['电流>60A次数'] >= HIGH_LOSS_CURRENT_TIMES:
            reasons.append(f"中高电流使用频繁（电流超60A共{row['电流>60A次数']}次）")
        if row['能量数据有效'] and row['百公里电耗(kWh)'] >= EXTREME_ENERGY_THRESHOLD:
            reasons.append(f"能耗极高（百公里电耗{row['百公里电耗(kWh)']}kWh）")
        if row['SOC数据有效'] and row['SOC低于20%时长占比'] >= 0.5:
            reasons.append(f"长期低电量运行（SOC低于20%时长占比{row['SOC低于20%时长占比']*100:.0f}%）")
        if not reasons:
            reasons.append(f"包月友好分较低（{row['包月友好评分']}分），电池损耗速度高于平均水平")
        return " | ".join(reasons)
    
    elif current_level == "优质用户":
        return f"电池使用习惯优秀（包月友好分{row['包月友好评分']}分），电流、速度、SOC均保持在健康区间"
    elif current_level == "良好用户":
        return f"电池使用习惯较好（包月友好分{row['包月友好评分']}分），整体负载可控"
    elif current_level == "普通用户":
        return f"电池使用行为一般（包月友好分{row['包月友好评分']}分），无明显过激使用行为"
    else:
        return "数据不足或无有效骑行数据，无法判定"


def _generate_type_desc(row):
    """生成客户形态说明"""
    current_type = row['客户形态']
    if current_type == "改装/超速车":
        return f"行驶特征异常（最高速度{row['最大速度']}km/h，平均骑行电流{row['骑行放电平均电流']}A），远超普通两轮车水平，存在改装或超速嫌疑"
    elif current_type == "地摊/储能":
        return f"非移动用电特征明显（当日骑行{row['骑行总耗时(小时)']}小时，怠速放电{row['怠速放电时长(小时)']}小时），放电以静止状态为主，疑似地摊供电或储能场景"
    elif current_type == "专送骑手":
        return f"工作特征显著（当日骑行{row['骑行总耗时(小时)']:.1f}小时，高峰骑行占比{row['高峰骑行占比']*100:.0f}%），工作时长稳定且午晚高峰高度活跃，符合专送骑手画像"
    elif current_type == "众包骑手":
        return f"具有兼职骑手特征（当日骑行{row['骑行总耗时(小时)']:.1f}小时，高峰骑行占比{row['高峰骑行占比']*100:.0f}%），高峰时段有一定活跃度"
    elif current_type == "标准骑手":
        return f"骑行行为规律（当日行驶里程{row['行驶距离']}km，骑行时长{row['骑行总耗时(小时)']:.1f}小时），属于标准日常使用场景"
    elif current_type == "普通骑手":
        return f"有常规骑行行为（当日行驶里程{row['行驶距离']}km，骑行时长{row['骑行总耗时(小时)']:.1f}小时），不符合特定骑手标签特征"
    else:
        return "无有效骑行数据或数据量不足，无法判定具体使用场景"


def append_to_snapshot(df_new: pd.DataFrame, path: str) -> None:
    """追加写入快照文件，按 [用户id, 统计日期] 去重"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table_new = pa.Table.from_pandas(df_new)
    if os.path.exists(path):
        table_old = pq.read_table(path)
        
        # 对齐新旧表的列，删除旧表中多余列（如__index_level_0__）
        old_cols = set(table_old.column_names)
        new_cols = set(table_new.column_names)
        
        # 从旧表中移除新表不存在的列
        cols_to_remove = old_cols - new_cols
        if cols_to_remove:
            table_old = table_old.drop([c for c in cols_to_remove])
        
        # 从新表中添加旧表有但新表没有的列（填充None）
        cols_to_add = new_cols - old_cols
        if cols_to_add:
            for col in cols_to_add:
                table_old = table_old.append_column(col, pa.array([None] * len(table_old)))
        
        # 修复类型不一致：将旧表中null类型的列转换为string
        for col in table_old.column_names:
            if pa.types.is_null(table_old.schema.field(col).type):
                table_old = table_old.set_column(
                    table_old.column_names.index(col),
                    col,
                    pa.array([None] * len(table_old), type=pa.string())
                )

        # 修复类型不一致：统一 string / large_string，以旧表类型为准
        for col in table_old.column_names:
            old_type = table_old.schema.field(col).type
            new_type = table_new.schema.field(col).type
            if old_type != new_type:
                if pa.types.is_string(old_type) or pa.types.is_large_string(old_type):
                    target_type = pa.string()
                    table_old = table_old.set_column(
                        table_old.column_names.index(col), col,
                        table_old.column(col).cast(target_type)
                    )
                    table_new = table_new.set_column(
                        table_new.column_names.index(col), col,
                        table_new.column(col).cast(target_type)
                    )

        df = pa.concat_tables([table_old, table_new]).to_pandas()
        df = df.drop_duplicates(subset=['用户id', '统计日期'], keep='last')
        table_new = pa.Table.from_pandas(df)
    pq.write_table(table_new, path, compression='zstd')


def process_snapshot_layer(target_date: Optional[str] = None, target_dates: Optional[set] = None) -> str:
    """
    L2 Snapshot Layer 主入口

    Args:
        target_date: 目标日期 YYYY-MM-DD，None 则处理所有日期
        target_dates: 目标日期集合（增量模式），优先级高于 target_date

    Returns:
        str: 快照文件路径
    """
    print("\n" + "=" * 80)
    print("L2 Snapshot Layer - 用户日级快照层")
    print("=" * 80)

    fact_dir = EXPORT_PATH_FACT_DAILY
    output_path = EXPORT_PATH_SNAPSHOT
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(fact_dir):
        print("❌ Fact 数据目录不存在，请先运行 L1 Fact Layer")
        return ""

    fact_files = [f for f in os.listdir(fact_dir) if f.endswith('.parquet')]
    if not fact_files:
        print("❌ 未找到 Fact Parquet 文件")
        return ""

    all_user_rows = []

    for fact_file in fact_files:
        date_str = fact_file.replace('.parquet', '')
        if target_dates and date_str not in target_dates:
            continue
        if target_date and date_str != target_date:
            continue

        fact_path = os.path.join(fact_dir, fact_file)
        print(f"\n📊 处理日期: {date_str}")

        try:
            df_fact = pd.read_parquet(fact_path)
        except Exception as e:
            print(f"⚠️ 读取 Fact 数据失败 {fact_path}: {e}")
            continue

        if '用户id' not in df_fact.columns:
            print("   ❌ 缺少用户id字段")
            continue

        df_fact = df_fact.copy()
        df_fact['用户id'] = df_fact['用户id'].astype(str).fillna('未知用户')
        
        # 兼容两种字段名
        current_col = '骑行放电平均电流' if '骑行放电平均电流' in df_fact.columns else '平均骑行电流'
        df_fact['weighted_current'] = df_fact[current_col] * df_fact['骑行总耗时(小时)']

        group_key = ['统计日期', '用户id']
        g = df_fact.groupby(group_key)

        df_agg = pd.DataFrame()
        df_agg['关联合约数'] = g['合约id'].nunique()
        
        df_agg['当日总行驶距离_km'] = g['行驶距离'].sum()
        df_agg['当日总骑行时长_h'] = g['骑行总耗时(小时)'].sum()
        df_agg['当日总放电时长_h'] = g['总放电时长(小时)'].sum()
        df_agg['当日总怠速放电时长_h'] = g['怠速放电时长(小时)'].sum()
        df_agg['当日平均骑行放电比'] = g['骑行-放电时长比'].mean()
        
        df_agg['当日总用电量_kWh'] = g['总用电量(kWh)'].sum()
        df_agg['当日总骑行次数'] = g['骑行次数'].sum()
        df_agg['当日总换电次数'] = g['换电次数'].sum()
        
        df_agg['单合约日均行驶里程_km'] = (df_agg['当日总行驶距离_km'] / df_agg['关联合约数'].clip(lower=1)).round(2)
        df_agg['单合约日均骑行时长_h'] = (df_agg['当日总骑行时长_h'] / df_agg['关联合约数'].clip(lower=1)).round(2)
        df_agg['单合约日均放电时长_h'] = (df_agg['当日总放电时长_h'] / df_agg['关联合约数'].clip(lower=1)).round(2)
        df_agg['单合约日均怠速放电_h'] = (df_agg['当日总怠速放电时长_h'] / df_agg['关联合约数'].clip(lower=1)).round(2)
        df_agg['单合约日均用电量_kWh'] = (df_agg['当日总用电量_kWh'] / df_agg['关联合约数'].clip(lower=1)).round(2)
        
        df_agg['午间高峰长时骑行总次数'] = g['午间高峰长时骑行次数'].sum()
        df_agg['晚间高峰长时骑行总次数'] = g['晚间高峰长时骑行次数'].sum()
        df_agg['平峰长时骑行总次数'] = g['平峰长时骑行次数'].sum()
        df_agg['夜间长时骑行总次数'] = g['夜间长时骑行次数'].sum()
        
        df_agg['午间高峰总里程_km'] = g['午间高峰骑行里程'].sum()
        df_agg['晚间高峰总里程_km'] = g['晚间高峰骑行里程'].sum()
        df_agg['平峰总里程_km'] = g['平峰骑行里程'].sum()
        df_agg['夜间总里程_km'] = g['夜间骑行里程'].sum()
        
        df_agg['总电流超60A次数'] = g['电流>60A次数'].sum()
        df_agg['总电流超80A次数'] = g['电流>80A次数'].sum()
        df_agg['总超100A连续次数'] = g['超100A连续次数'].sum()
        df_agg['总超100A累计时长_h'] = g['超100A累计时长_h'].sum()
        
        df_agg['当日最高速度_kmh'] = g['最大速度'].max()
        df_agg['当日最大电流_A'] = g['最大电流'].max()
        df_agg['当日最高温度_℃'] = g['最大温度'].max()
        df_agg['当日最低SOC'] = g['最低SOC'].min()
        df_agg['最低包月友好分'] = g['包月友好评分'].min()
        df_agg['最大单合约活动半径_km'] = g['R95核心活动半径'].max()
        df_agg['R90活动半径_km'] = g['R90日常活动半径'].max()
        df_agg['最大凸包覆盖面积_km2'] = g['凸包覆盖面积'].max()
        df_agg['最大单次出行距离_km'] = g['最大出行距离'].max()
        df_agg['当日是否出勤'] = g['当日是否出勤'].max()
        
        df_agg['平均包月友好分'] = g['包月友好评分'].mean().round(1)
        df_agg['当日平均骑行SOC'] = g['平均骑行SOC'].mean().round(1)
        df_agg['当日平均骑行速度_kmh'] = g['平均骑行速度'].mean().round(2)
        df_agg['_weighted_current_sum'] = g['weighted_current'].sum()
        df_agg['当日平均骑行电流_A'] = np.where(
            df_agg['当日总骑行时长_h'] > 0,
            (df_agg['_weighted_current_sum'] / df_agg['当日总骑行时长_h']).round(2),
            0.0
        )
        df_agg['当日百公里电耗_kWh'] = g['百公里电耗(kWh)'].mean().round(2)

        def _weighted_aggregations(x):
            h = x['骑行总耗时(小时)']
            total_h = h.sum()
            return pd.Series({
                '_weighted_peak_ratio': (x['高峰骑行占比'] * h).sum(),
                '_weighted_soc20_ratio': (x['SOC低于20%时长占比'] * h).sum(),
                '_weighted_soc10_ratio': (x['SOC低于10%时长占比'] * h).sum(),
                '_total_riding_hour': total_h,
            })
        
        df_weighted = g.apply(_weighted_aggregations)
        df_agg = pd.concat([df_agg, df_weighted], axis=1)
        
        df_agg['高峰骑行占比'] = np.where(
            df_agg['_total_riding_hour'] > 0, 
            (df_agg['_weighted_peak_ratio'] / df_agg['_total_riding_hour']).round(2), 
            0
        )
        df_agg['SOC低于20%时长占比_当日'] = np.where(
            df_agg['_total_riding_hour'] > 0, 
            (df_agg['_weighted_soc20_ratio'] / df_agg['_total_riding_hour']).round(2), 
            0
        )
        df_agg['SOC低于10%时长占比_当日'] = np.where(
            df_agg['_total_riding_hour'] > 0,
            (df_agg['_weighted_soc10_ratio'] / df_agg['_total_riding_hour']).round(2),
            0
        )

        # 换电时段聚合
        df_agg['当日平峰换电次数'] = g['平峰换电次数'].sum()
        df_agg['当日深夜换电次数'] = g['深夜换电次数'].sum()
        df_agg['当日高峰换电次数'] = g['高峰换电次数'].sum()

        total_swap = g['换电总次数'].sum().replace(0, np.nan)
        df_agg['平峰换电占比_当日'] = (g['平峰换电次数'].sum() / total_swap).fillna(0).round(3)
        df_agg['深夜换电占比_当日'] = (g['深夜换电次数'].sum() / total_swap).fillna(0).round(3)

        # SOC 换电画像聚合
        df_agg['取电时平均SOC_当日'] = g['取电时平均SOC'].mean().round(1)
        df_agg['还电时平均SOC_当日'] = g['还电时平均SOC'].mean().round(1)
        df_agg['单次换电平均SOC消耗_当日'] = g['单次换电平均SOC消耗'].mean().round(1)

        # 速度分位聚合
        df_agg['当日P50骑行速度_kmh'] = g['P50骑行速度_kmh'].mean().round(2)
        df_agg['当日P90骑行速度_kmh'] = g['P90骑行速度_kmh'].max().round(2)
        df_agg['当日夜间骑行均速_kmh'] = g['夜间骑行均速_kmh'].max().round(2)
        df_agg['当日高速骑行点数'] = g['高速骑行点数(>40kmh)'].sum()

        # 骑行时刻聚合
        df_agg['最早骑行时刻_h'] = g['最早骑行时刻_h'].min()
        df_agg['最晚骑行时刻_h'] = g['最晚骑行时刻_h'].max()

        df_custom = pd.DataFrame()
        df_custom['核心活动省份'] = g['核心活动省份'].apply(lambda x: x.value_counts().index[0] if len(x.dropna())>0 else "")
        df_custom['核心活动城市'] = g['核心活动城市'].apply(lambda x: x.value_counts().index[0] if len(x.dropna())>0 else "")
        df_custom['核心活动区县'] = g['核心活动区县'].apply(lambda x: x.value_counts().index[0] if len(x.dropna())>0 else "")
        df_custom['风险标签_电流异常'] = g['电流异常用户'].apply(lambda x: '是' if '是' in list(x) else '否')
        df_custom['客户形态_综合'] = g.apply(_get_user_type)
        df_custom['用户等级_综合'] = g.apply(_get_user_level)

        def _dominant_swap_period(series):
            counts = series.value_counts()
            for period in ['平峰', '深夜', '高峰', '无换电']:
                if period in counts.index:
                    return period
            return '无换电'

        df_custom['偏好换电时段_综合'] = g['偏好换电时段'].apply(_dominant_swap_period)

        def _dominant_period(series):
            counts = series[series != '无数据'].value_counts()
            return counts.index[0] if len(counts) > 0 else '无数据'

        df_custom['主要骑行时段_综合'] = g['主要骑行时段'].apply(_dominant_period)

        if '用户等级说明' in df_fact.columns:
            df_custom['用户等级_综合说明'] = g.apply(lambda x: _get_level_desc(x['用户等级'], x['用户等级说明']))
        if '客户形态说明' in df_fact.columns:
            df_custom['客户形态_综合说明'] = g.apply(lambda x: _get_type_desc(x['客户形态'], x['客户形态说明']))

        df_user = pd.concat([df_agg, df_custom], axis=1).reset_index()
        df_user = df_user.drop(columns=['_weighted_peak_ratio', '_total_riding_hour', '_weighted_soc20_ratio', '_weighted_soc10_ratio', '_weighted_current_sum'], errors='ignore')
        
        all_user_rows.append(df_user)

    if not all_user_rows:
        print("❌ 无有效用户快照数据")
        return ""

    df_snapshot = pd.concat(all_user_rows, ignore_index=True)

    append_to_snapshot(df_snapshot, output_path)
    print(f"✅ L2 保存: {output_path} ({len(df_snapshot)} 用户日记录)")

    return output_path


if __name__ == "__main__":
    process_snapshot_layer(target_date=None)
