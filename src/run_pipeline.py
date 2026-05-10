# -*- coding: utf-8 -*-
"""
Run Pipeline - 数据流水线主入口
职责：统一调度 L1-L4 各层，支持多种执行模式。

执行模式：
- full:         清除历史数据 → L1 → L4，全量重建
- incremental:  L1 → L4，仅处理新日期（已有数据自动跳过）
- refresh:      L2 → L4，调整聚合逻辑后重跑
- recalculate:  L3 → L4，调整评级规则后重跑
- report_only:  L4，仅重新生成报告文件

使用示例：
    python run_pipeline.py --mode incremental --date 2026-04-19
    python run_pipeline.py --mode full
    python run_pipeline.py --mode recalculate
    python run_pipeline.py --mode report_only --date 2026-04-19
"""

import os
import sys
import argparse
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 确保src目录也在path中（支持 python -m src.run_pipeline 方式运行）
src_dir = os.path.dirname(__file__)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config.config import DATA_OUTPUT_ROOT

# 使用兼容两种运行方式的导入
try:
    from fact_layer import process_fact_layer
    from snapshot_layer import process_snapshot_layer
    from lifecycle_layer import process_lifecycle_layer
    from report_layer import process_report_layer
except ImportError:
    # 如果上述导入失败，尝试从src包导入
    from src.fact_layer import process_fact_layer
    from src.snapshot_layer import process_snapshot_layer
    from src.lifecycle_layer import process_lifecycle_layer
    from src.report_layer import process_report_layer


def run_full_pipeline(target_date: str = None):
    """full 模式：清除历史数据 → L1 → L4，全量重建"""
    print("\n" + "=" * 80)
    print("🚀 执行模式: FULL (全量重建)")
    print("=" * 80)

    start_time = datetime.now()

    # Step 0: 清除历史数据（使用实际数据目录，而非base_path根目录）
    import shutil
    layers_to_clean = ['fact', 'snapshot', 'lifecycle', 'reports']
    cleaned_count = 0
    for layer in layers_to_clean:
        layer_dir = os.path.join(DATA_OUTPUT_ROOT, layer)
        if os.path.exists(layer_dir):
            try:
                shutil.rmtree(layer_dir)
                os.makedirs(layer_dir, exist_ok=True)
                cleaned_count += 1
                print(f"   🗑️  已清除 {layer}/ 目录")
            except Exception as e:
                print(f"   ⚠️ 清除 {layer}/ 失败: {e}")
    
    if cleaned_count > 0:
        print(f"   ✅ 已清除 {cleaned_count} 个历史数据目录，开始全量重算\n")

    # L1: Fact层（全量处理所有日期）
    process_fact_layer(target_date=None)

    # L2: Snapshot层
    process_snapshot_layer(target_date=None)

    # L3: Lifecycle层
    process_lifecycle_layer(target_date=None)

    # L4: Report层
    process_report_layer(target_date=None)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  全量流水线完成，总耗时: {elapsed:.2f} 秒")


def run_incremental(target_date: str = None):
    """incremental 模式：L1 → L4，仅处理新日期（自动检测缺失日期，排除当天）"""
    print("\n" + "=" * 80)
    print("🚀 执行模式: INCREMENTAL (增量处理)")
    print("=" * 80)

    start_time = datetime.now()

    if target_date:
        print(f"   ⚠️  注意: incremental模式将忽略--date参数，自动检测所有缺失日期")

    # L1: Fact层（自动跳过已有日期和当天数据），返回新增日期
    _, new_dates = process_fact_layer(target_date=None)

    if not new_dates:
        print("   ✅ 无新增日期需要处理，流水线结束")
        return

    print(f"\n   📋 新增日期: {sorted(new_dates)}")

    # L2: Snapshot层（仅处理新增日期）
    process_snapshot_layer(target_dates=new_dates)

    # L3: Lifecycle层（需要全量快照数据计算7天滚动窗口）
    process_lifecycle_layer(target_date=None)

    # L4: Report层
    process_report_layer(target_date=None)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  增量流水线完成，总耗时: {elapsed:.2f} 秒")


def run_refresh(target_date: str = None):
    """refresh 模式：L2 → L4，调整聚合逻辑后重跑（保留L1数据）"""
    print("\n" + "=" * 80)
    print("🚀 执行模式: REFRESH (刷新聚合)")
    print("=" * 80)

    start_time = datetime.now()

    # L2: Snapshot层（重算聚合）
    process_snapshot_layer(target_date=target_date)

    # L3: Lifecycle层
    process_lifecycle_layer(target_date=target_date)

    # L4: Report层
    process_report_layer(target_date=target_date)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  刷新流水线完成，总耗时: {elapsed:.2f} 秒")


def run_recalculate(target_date: str = None):
    """recalculate 模式：L3 → L4，调整评级规则后重跑（保留L1-L2数据）"""
    print("\n" + "=" * 80)
    print("🚀 执行模式: RECALCULATE (重新评级)")
    print("=" * 80)

    start_time = datetime.now()

    # L3: Lifecycle层（重新评分和分类）
    process_lifecycle_layer(target_date=target_date)

    # L4: Report层
    process_report_layer(target_date=target_date)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  重算流水线完成，总耗时: {elapsed:.2f} 秒")


def run_report_only(target_date: str = None):
    """report_only 模式：L4，仅重新生成报告文件（保留L1-L3数据）"""
    print("\n" + "=" * 80)
    print("🚀 执行模式: REPORT_ONLY (仅报告)")
    print("=" * 80)

    start_time = datetime.now()

    # L4: Report层（重新生成报告）
    process_report_layer(target_date=target_date)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  报告生成完成，总耗时: {elapsed:.2f} 秒")




def main():
    parser = argparse.ArgumentParser(description='两轮车换电用户分析系统 - 数据流水线')
    parser.add_argument(
        '--mode',
        choices=['full', 'incremental', 'refresh', 'recalculate', 'report_only'],
        default='incremental',
        help='执行模式 (默认: incremental)'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='目标日期 YYYY-MM-DD (默认: 今天)'
    )

    args = parser.parse_args()

    mode_map = {
        'full': run_full_pipeline,
        'incremental': run_incremental,
        'refresh': run_refresh,
        'recalculate': run_recalculate,
        'report_only': run_report_only,
    }

    pipeline_func = mode_map[args.mode]
    pipeline_func(target_date=args.date)


if __name__ == "__main__":
    main()