"""全流程验证脚本: 完整L3生命周期层处理 + 本地输出"""
import sys, os, json
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = 'temp/pipeline_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'lifecycle'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'reports'), exist_ok=True)

print("=" * 70)
print("L3 Lifecycle Layer - 全流程验证 (本地输出模式)")
print("=" * 70)

# ========== 1. 加载快照 ==========
SNAP_PATH = r'E:\OneDrive\DataBase\DataBase\用户行为习惯\用户特征画像\snapshot\user_daily.parquet'
print(f'\n[1/5] 加载快照数据...')
df_snapshot = pd.read_parquet(SNAP_PATH)
print(f'   完成：{len(df_snapshot):,} 条记录, {df_snapshot["用户id"].nunique():,} 用户')

# ========== 2. 计算7天滚动指标 ==========
from pipeline.lifecycle_layer import (
    calc_rolling_7d_metrics,
    _add_derived_portrait_metrics,
)
print(f'\n[2/5] 计算7天滚动指标...')
df_rolling = calc_rolling_7d_metrics(df_snapshot)
if df_rolling.empty:
    print('   无有效数据!')
    sys.exit(1)
print(f'   完成：{len(df_rolling):,} 个用户')

# 保存原始滚动指标
df_rolling.to_parquet(os.path.join(OUTPUT_DIR, 'lifecycle', 'user_7d_raw.parquet'), engine='pyarrow', compression='zstd')
print(f'   已保存: {OUTPUT_DIR}/lifecycle/user_7d_raw.parquet')

# ========== 3. 添加衍生指标 ==========
print(f'\n[3/5] 计算衍生指标...')
df_scored = _add_derived_portrait_metrics(df_rolling)
print(f'   完成')

# ========== 4. 用户形态分布统计 ==========
print(f'\n[4/5] 用户形态分布:')
vc = df_scored['用户形态_综合_7d'].value_counts()
total = len(df_scored)
for k in ['专送骑手', '众包骑手', '地摊/储能', '数据不足']:
    v = vc.get(k, 0)
    pct = v / total * 100
    bar = '█' * int(pct / 2)
    print(f'   {k:>8}: {v:>6}人 ({pct:>5.1f}%) {bar}')

zhongbao = vc.get('专送骑手', 0) + vc.get('众包骑手', 0)
print(f'   {"合计":>8}: {zhongbao:>6}人 ({zhongbao/total*100:>5.1f}%) [专送+众包]')

# ========== 5. 输出报告 ==========
print(f'\n[5/5] 生成报告...')

# CSV完整报告
report_cols = [
    '用户id', '统计日期', '近7天有数据天数', '近7天有效数据天数',
    '核心活动省份', '核心活动城市', '核心活动区县',
    '近7d单合约日均骑行时长_h', '近7d高峰骑行占比',
    '近7d总行驶距离_km', '近7d日均行驶距离_km',
    '近7d凸包覆盖面积_km2', '近7d_R90活动半径_km',
    '近7d平均上线时间熵值', '近7d平均路线曲折系数', '近7d平均速度变异系数',
    '用户形态_综合_7d', '车辆形态_7d',
    '近7d出勤率', '近7d工作特点',
]
available_cols = [c for c in report_cols if c in df_scored.columns]
df_report = df_scored[available_cols].copy()
csv_path = os.path.join(OUTPUT_DIR, 'reports', 'user_7d_full.csv')
df_report.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'   [OK] 用户画像CSV: {csv_path} ({len(df_report):,} 用户)')

# Parquet完整数据
parquet_path = os.path.join(OUTPUT_DIR, 'lifecycle', 'user_7d.parquet')
df_scored.to_parquet(parquet_path, engine='pyarrow', compression='zstd')
print(f'   [OK] L3 Parquet: {parquet_path}')

# 分布统计
dist_df = pd.DataFrame({
    '用户形态': vc.index,
    '人数': vc.values,
    '占比': (vc.values / total * 100).round(1),
})
dist_path = os.path.join(OUTPUT_DIR, 'reports', 'level_distribution.csv')
dist_df.to_csv(dist_path, index=False, encoding='utf-8-sig')
print(f'   [OK] 分布统计: {dist_path}')

# 日级 vs L3 对比
snap_mode = df_snapshot.groupby('用户id')['用户形态_综合'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '数据不足'
)
cross = pd.DataFrame({'日级': snap_mode, 'L3': df_scored.set_index('用户id')['用户形态_综合_7d']})
cross = cross.dropna()
cross_table = pd.crosstab(cross['日级'], cross['L3'])
cross_path = os.path.join(OUTPUT_DIR, 'reports', 'daily_vs_l3_cross.csv')
cross_table.to_csv(cross_path, encoding='utf-8-sig')
print(f'   [OK] 日级/L3交叉表: {cross_path}')
print('\n' + str(cross_table))

print(f'\n{"=" * 70}')
print(f'[DONE] 全流程完成! 所有结果保存在: {OUTPUT_DIR}/')
print(f'{"=" * 70}')