# -*- coding: utf-8 -*-
"""
L4 Report Layer - 报告输出层
职责：从生命周期 Parquet 生成各类面向业务的输出文件，不做任何计算。

输入：lifecycle/user_7d.parquet
输出：reports/ 目录下各类报告文件（固定目录，每次覆盖最新）
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import EXPORT_PATH_LIFECYCLE_7D, EXPORT_PATH_REPORTS


def generate_user_detail(df: pd.DataFrame, output_dir: str) -> str:
    """生成用户明细 CSV"""
    output_path = os.path.join(output_dir, "user_detail.csv")

    keep_cols = [
        '用户id', '统计日期', '核心活动省份', '核心活动城市', '核心活动区县',
        '近7d总行驶距离_km', '近7d总骑行时长_h', '近7d平均骑行电流_A',
        '近7d最大电流_A', '近7d百公里电耗_kWh', '近7d最低SOC',
        '近7d出勤天数', '近7d出勤率',
        '客户形态_综合_7d', '近7d平均包月友好分',
        '重评分数', '用户等级_动态', '风险标签', '策略建议', '用户等级_动态说明',
        '用户生命周期状态_7d', '设备状态监控'
    ]

    available_cols = [c for c in keep_cols if c in df.columns]
    df[available_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 用户明细: {output_path}")
    return output_path


def generate_user_7d_full(df: pd.DataFrame, output_dir: str) -> str:
    """生成完整用户7天滚动数据 CSV（全量字段）"""
    output_path = os.path.join(output_dir, "user_7d_full.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 用户7天滚动全量数据: {output_path} ({len(df):,} 用户, {len(df.columns)} 字段)")
    return output_path


def generate_level_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """生成用户等级分布统计 CSV"""
    output_path = os.path.join(output_dir, "level_distribution.csv")

    # 等级分布
    level_dist = df['用户等级_动态'].value_counts().reset_index()
    level_dist.columns = ['用户等级', '用户数']
    level_dist['占比'] = (level_dist['用户数'] / len(df) * 100).round(2)

    # 客户形态分布
    customer_dist = df['客户形态_综合_7d'].value_counts().reset_index()
    customer_dist.columns = ['客户形态', '用户数']
    customer_dist['占比'] = (customer_dist['用户数'] / len(df) * 100).round(2)

    # 策略分布
    strategy_dist = df['策略建议'].value_counts().reset_index()
    strategy_dist.columns = ['策略建议', '用户数']
    strategy_dist['占比'] = (strategy_dist['用户数'] / len(df) * 100).round(2)

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write("用户等级分布\n")
        level_dist.to_csv(f, index=False, encoding='utf-8-sig')
        f.write("\n客户形态分布\n")
        customer_dist.to_csv(f, index=False, encoding='utf-8-sig')
        f.write("\n策略建议分布\n")
        strategy_dist.to_csv(f, index=False, encoding='utf-8-sig')

    print(f"   ✅ 等级分布: {output_path}")
    return output_path


def generate_risk_alerts(df: pd.DataFrame, output_dir: str) -> str:
    """生成风险用户告警列表 CSV"""
    output_path = os.path.join(output_dir, "risk_alerts.csv")

    # 暴力用户 + 高损耗用户
    risk_users = df[df['用户等级_动态'].isin(['暴力', '高损耗用户'])].copy()

    keep_cols = [
        '用户id', '统计日期', '用户等级_动态', '风险标签', '策略建议',
        '近7d最大电流_A', '近7d平均骑行电流_A', '近7d百公里电耗_kWh',
        '近7d超100A连续总次数', '单合约月度用电度数预估_kWh'
    ]

    available_cols = [c for c in keep_cols if c in risk_users.columns]
    risk_users[available_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 风险告警: {output_path} ({len(risk_users)} 用户)")
    return output_path


def generate_shift_report(df: pd.DataFrame, output_dir: str) -> str:
    """生成分布漂移检测报告 JSON"""
    output_path = os.path.join(output_dir, "shift_report.json")

    report = {
        'generated_at': datetime.now().isoformat(),
        'total_users': len(df),
        'key_metrics': {
            'avg_current': {
                'mean': round(df['近7d平均骑行电流_A'].mean(), 2),
                'median': round(df['近7d平均骑行电流_A'].median(), 2),
                'P90': round(df['近7d平均骑行电流_A'].quantile(0.9), 2),
            },
            'max_current': {
                'mean': round(df['近7d最大电流_A'].mean(), 2),
                'median': round(df['近7d最大电流_A'].median(), 2),
                'P90': round(df['近7d最大电流_A'].quantile(0.9), 2),
            },
            'energy_per_100km': {
                'mean': round(df['近7d百公里电耗_kWh'].mean(), 2),
                'median': round(df['近7d百公里电耗_kWh'].median(), 2),
                'P90': round(df['近7d百公里电耗_kWh'].quantile(0.9), 2),
            },
        },
        'level_distribution': df['用户等级_动态'].value_counts().to_dict(),
        'risk_tag_distribution': df['风险标签'].value_counts().head(20).to_dict(),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 漂移报告: {output_path}")
    return output_path


def generate_summary(df: pd.DataFrame, output_dir: str) -> str:
    """生成文本摘要报告"""
    output_path = os.path.join(output_dir, "summary.txt")

    total_users = len(df)
    level_dist = df['用户等级_动态'].value_counts()
    risk_dist = df['风险标签'].value_counts().head(10)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  用户生命周期分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总用户数: {total_users:,}\n\n")

        f.write("用户等级分布:\n")
        for level, count in level_dist.items():
            pct = count / total_users * 100
            f.write(f"  {level:<12}: {count:>6,} 人  ({pct:.1f}%)\n")

        f.write("\n风险标签 TOP 10:\n")
        for tag, count in risk_dist.items():
            pct = count / total_users * 100
            f.write(f"  {tag:<15}: {count:>6,} 人  ({pct:.1f}%)\n")

        f.write("\n核心指标:\n")
        f.write(f"  平均骑行电流: {df['近7d平均骑行电流_A'].mean():.2f} A\n")
        f.write(f"  平均百公里电耗: {df['近7d百公里电耗_kWh'].mean():.2f} kWh\n")
        f.write(f"  平均出勤率: {df['近7d出勤率'].mean():.2%}\n")

    print(f"   ✅ 文本摘要: {output_path}")
    return output_path


def generate_attendance_detail(df: pd.DataFrame, output_dir: str) -> str:
    """生成用户月度出勤预估明细 CSV"""
    output_path = os.path.join(output_dir, "user_monthly_attendance_detail.csv")
    
    attendance_cols = [
        '用户id', '统计日期',
        '近7d出勤天数', '近7d出勤率', '近7天有数据天数',
        '单合约月度用电度数预估_kWh', '近7d总用电量_kWh',
        '近7d总行驶距离_km', '客户形态_综合_7d', '用户等级_动态'
    ]
    
    available_cols = [c for c in attendance_cols if c in df.columns]
    df_attendance = df[available_cols].copy()
    
    df_attendance.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 月度出勤预估明细: {output_path} ({len(df_attendance):,} 用户)")
    return output_path


def process_report_layer(target_date: Optional[str] = None) -> str:
    """
    L4 Report Layer 主入口

    Args:
        target_date: 目标日期 YYYY-MM-DD，用于报告目录命名

    Returns:
        str: 报告目录路径
    """
    print("\n" + "=" * 80)
    print("L4 Report Layer - 报告输出层")
    print("=" * 80)

    lifecycle_path = EXPORT_PATH_LIFECYCLE_7D

    if not os.path.exists(lifecycle_path):
        print("❌ 生命周期数据文件不存在，请先运行 L3 Lifecycle Layer")
        return ""

    print("\n[1/1] 加载生命周期数据...")
    df = pd.read_parquet(lifecycle_path)
    print(f"   ✅ 加载完成：{len(df):,} 个用户")

    # 报告目录（固定为latest，每次覆盖最新报告）
    output_dir = EXPORT_PATH_REPORTS
    os.makedirs(output_dir, exist_ok=True)

    print("\n生成报告文件...")
    generate_user_detail(df, output_dir)
    generate_user_7d_full(df, output_dir)
    generate_level_distribution(df, output_dir)
    generate_risk_alerts(df, output_dir)
    generate_attendance_detail(df, output_dir)
    generate_shift_report(df, output_dir)
    generate_summary(df, output_dir)

    print(f"\n✅ L4 Report Layer 完成，报告目录: {output_dir}")
    return output_dir


if __name__ == "__main__":
    process_report_layer(target_date=None)
