# -*- coding: utf-8 -*-
"""验证输出数据质量"""
import pandas as pd
import numpy as np
import os

# 加载数据
fact_path = r"E:\test\data\fact\daily\2026-04-15.parquet"
snapshot_path = r"E:\test\data\snapshot\user_daily.parquet"
lifecycle_path = r"E:\test\data\lifecycle\user_7d.parquet"

print("=" * 80)
print("数据质量验证")
print("=" * 80)

# 1. Fact Layer验证
print("\n【Fact Layer验证】")
df_fact = pd.read_parquet(fact_path)
print(f"总记录数: {len(df_fact):,}")
print(f"唯一用户数: {df_fact['用户id'].nunique():,}")
print(f"唯一合约数: {df_fact['合约id'].nunique():,}")

# 检查关键字段的空值和零值比例
fact_key_cols = [
    '客户形态', '包月友好评分', '用户等级',
    '行驶距离', '骑行总耗时(小时)', '骑行放电平均电流',
    '最大速度', '最大电流', '最大温度',
    '电流>60A次数', '电流>80A次数', '超100A连续次数',
    '平均骑行SOC', '最低SOC', 'SOC低于20%时长占比', 'SOC低于10%时长占比',
    '百公里电耗(kWh)', '换电次数',
]

print("\nFact Layer关键字段统计:")
for col in fact_key_cols:
    if col in df_fact.columns:
        non_zero = (df_fact[col] != 0).sum()
        non_null = df_fact[col].notna().sum()
        total = len(df_fact)
        print(f"  {col}: 非零={non_zero:,}({non_zero/total*100:.1f}%), 非空={non_null:,}({non_null/total*100:.1f}%)")

# 客户形态分布
print("\n客户形态分布:")
print(df_fact['客户形态'].value_counts())

# 用户等级分布
print("\n用户等级分布:")
print(df_fact['用户等级'].value_counts())

# 包月友好评分分布
print("\n包月友好评分分布:")
print(df_fact['包月友好评分'].describe())

# 2. Snapshot Layer验证
print("\n" + "=" * 80)
print("【Snapshot Layer验证】")
df_snap = pd.read_parquet(snapshot_path)
print(f"总记录数: {len(df_snap):,}")
print(f"唯一用户数: {df_snap['用户id'].nunique():,}")

snap_key_cols = [
    '客户形态_综合', '用户等级_综合',
    '当日总行驶距离_km', '当日总骑行时长_h',
    '平均包月友好分', '当日平均骑行SOC', '当日平均骑行速度_kmh',
    '当日平均骑行电流_A', '当日百公里电耗_kWh',
]

print("\nSnapshot Layer关键字段统计:")
for col in snap_key_cols:
    if col in df_snap.columns:
        non_zero = (df_snap[col] != 0).sum()
        non_null = df_snap[col].notna().sum()
        total = len(df_snap)
        print(f"  {col}: 非零={non_zero:,}({non_zero/total*100:.1f}%), 非空={non_null:,}({non_null/total*100:.1f}%)")

print("\n客户形态_综合分布:")
print(df_snap['客户形态_综合'].value_counts())

print("\n用户等级_综合分布:")
print(df_snap['用户等级_综合'].value_counts())

# 3. Lifecycle Layer验证
print("\n" + "=" * 80)
print("【Lifecycle Layer验证】")
df_life = pd.read_parquet(lifecycle_path)
print(f"总记录数: {len(df_life):,}")
print(f"唯一用户数: {df_life['用户id'].nunique():,}")

life_key_cols = [
    '近7d平均包月友好分', '近7d平均骑行SOC', '近7d平均骑行速度_kmh',
    '近7d平均骑行电流_A', '近7d百公里电耗_kWh',
    '近7d最高速度_kmh', '近7d最大电流_A', '近7d最低SOC',
    '近7d总骑行次数', '近7d总换电次数',
    '客户形态_综合_7d', '用户等级_动态', '风险标签',
]

print("\nLifecycle Layer关键字段统计:")
for col in life_key_cols:
    if col in df_life.columns:
        non_zero = (df_life[col] != 0).sum()
        non_null = df_life[col].notna().sum()
        total = len(df_life)
        print(f"  {col}: 非零={non_zero:,}({non_zero/total*100:.1f}%), 非空={non_null:,}({non_null/total*100:.1f}%)")

if '客户形态_综合_7d' in df_life.columns:
    print("\n客户形态_综合_7d分布:")
    print(df_life['客户形态_综合_7d'].value_counts())

if '用户等级_动态' in df_life.columns:
    print("\n用户等级_动态分布:")
    print(df_life['用户等级_动态'].value_counts())

if '风险标签' in df_life.columns:
    print("\n风险标签分布(前20):")
    print(df_life['风险标签'].value_counts().head(20))

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)
