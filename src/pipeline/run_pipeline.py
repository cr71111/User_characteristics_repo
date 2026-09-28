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
    # 日常增量处理（默认模式）
    python src/pipeline/run_pipeline.py
    python src/pipeline/run_pipeline.py --mode incremental

    # 全量重建
    python src/pipeline/run_pipeline.py --mode full

    # 刷新聚合（修改L2逻辑后）
    python src/pipeline/run_pipeline.py --mode refresh
    python src/pipeline/run_pipeline.py --mode refresh --date 2026-04-19

    # 重新评级（修改L3规则后）
    python src/pipeline/run_pipeline.py --mode recalculate
    python src/pipeline/run_pipeline.py --mode recalculate --date 2026-04-19

    # 仅生成报告
    python src/pipeline/run_pipeline.py --mode report_only
    python src/pipeline/run_pipeline.py --mode report_only --date 2026-04-19

    # 从指定层开始重建
    python src/pipeline/run_pipeline.py --layer L1
    python src/pipeline/run_pipeline.py --layer L2
    python src/pipeline/run_pipeline.py --layer L3
    python src/pipeline/run_pipeline.py --layer L4

    # 仅重建最近7天（避免全量OOM，修改fact逻辑后推荐）
    python src/pipeline/run_pipeline.py --layer L1 --days 7

    # 关闭数据完整性检查
    python src/pipeline/run_pipeline.py --mode incremental --skip-incomplete false
"""

import os
import sys
import argparse
import io
import traceback
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

LOGS_DIR = os.path.join(project_root, 'logs')


class TeeOutput:
    """同时输出到控制台和日志文件"""

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = io.open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            self.terminal.write(message.encode(self.terminal.encoding or 'utf-8', errors='replace').decode(self.terminal.encoding or 'utf-8', errors='replace'))
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def setup_pipeline_logging(mode: str) -> str:
    """设置日志：将所有 stdout 输出同时写入日志文件

    Returns:
        str: 日志文件路径
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOGS_DIR, f'pipeline_{mode}_{timestamp}.log')
    sys.stdout = TeeOutput(log_path)
    return log_path

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config.config import DATA_OUTPUT_ROOT

from src.pipeline.fact_layer import process_fact_layer
from src.pipeline.snapshot_layer import process_snapshot_layer
from src.pipeline.lifecycle_layer import process_lifecycle_layer
from src.pipeline.report_layer import process_report_layer


def run_full_pipeline(target_date: str = None, skip_incomplete: bool = True):
    """full 模式：清除历史数据 → L1 → L4，全量重建"""
    print("\n" + "=" * 80)
    print("🚀 执行模式: FULL (全量重建)")
    print("=" * 80)

    start_time = datetime.now()

    layers_to_clean = ['fact', 'snapshot', 'lifecycle', 'reports']
    for layer in layers_to_clean:
        layer_dir = os.path.join(DATA_OUTPUT_ROOT, layer)
        _clear_directory(layer_dir, layer)

    print()

    # L1: Fact层（全量处理所有日期）
    process_fact_layer(target_date=None, skip_incomplete=skip_incomplete)

    # L2: Snapshot层
    process_snapshot_layer(target_date=None)

    # L3: Lifecycle层
    process_lifecycle_layer(target_date=None)

    # L4: Report层
    process_report_layer(target_date=None)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  全量流水线完成，总耗时: {elapsed:.2f} 秒")


def run_incremental(target_date: str = None, skip_incomplete: bool = True):
    """incremental 模式：L1 → L4，仅处理新日期（自动检测缺失日期，排除当天）"""
    print("\n" + "=" * 80)
    print("🚀 执行模式: INCREMENTAL (增量处理)")
    print("=" * 80)

    start_time = datetime.now()

    if target_date:
        print(f"   ⚠️  注意: incremental模式将忽略--date参数，自动检测所有缺失日期")

    # L1: Fact层（自动跳过已有日期和当天数据），返回新增日期
    _, new_dates, _ = process_fact_layer(target_date=None, skip_incomplete=skip_incomplete)

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


def run_refresh(target_date: str = None, **kwargs):
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


def run_recalculate(target_date: str = None, **kwargs):
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


def run_report_only(target_date: str = None, **kwargs):
    """report_only 模式：L4，仅重新生成报告文件（保留L1-L3数据）"""
    print("\n" + "=" * 80)
    print("🚀 执行模式: REPORT_ONLY (仅报告)")
    print("=" * 80)

    start_time = datetime.now()

    # L4: Report层（重新生成报告）
    process_report_layer(target_date=target_date)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  报告生成完成，总耗时: {elapsed:.2f} 秒")


def _clear_directory(dir_path: str, dir_label: str) -> bool:
    """清除目录，处理 OneDrive 等文件锁问题"""
    import shutil
    import time

    if not os.path.exists(dir_path):
        return True

    try:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"   🗑️  已清除 {dir_label}/ 目录")
        return True
    except (PermissionError, OSError):
        pass

    deleted = 0
    failed = 0
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for f in files:
            fp = os.path.join(root, f)
            try:
                os.remove(fp)
                deleted += 1
            except (PermissionError, OSError):
                failed += 1
        for d in dirs:
            dp = os.path.join(root, d)
            try:
                os.rmdir(dp)
            except (PermissionError, OSError):
                pass

    if failed > 0:
        print(f"   ⚠️ {dir_label}/ 目录: 已删 {deleted} 个文件, {failed} 个被 OneDrive 锁定（将覆盖写入）")
    else:
        print(f"   🗑️  已清除 {dir_label}/ 目录 ({deleted} 个文件)")
    return failed == 0


def run_from_layer(layer: str, target_date: str = None, skip_incomplete: bool = True, recent_days: int = None):
    """从指定层开始重建：清除该层及下游数据，然后重算

    Args:
        layer: 起始层级 (L1/L2/L3/L4)
        target_date: 目标日期 YYYY-MM-DD
        skip_incomplete: 是否跳过数据不完整的日期
        recent_days: 仅处理最近N天的源数据（仅对 L1 生效），避免全量重算 OOM
    """

    layer_order = ['L1', 'L2', 'L3', 'L4']
    layer_dirs = {
        'L1': 'fact',
        'L2': 'snapshot',
        'L3': 'lifecycle',
        'L4': 'reports',
    }

    if layer not in layer_order:
        print(f"❌ 无效层级: {layer}，可选: {layer_order}")
        return

    start_idx = layer_order.index(layer)
    mode_label = f"FROM_{layer}"
    if recent_days and layer == 'L1':
        mode_label += f" (最近{recent_days}天)"

    print("\n" + "=" * 80)
    print(f"🚀 执行模式: {mode_label}")
    print("=" * 80)

    start_time = datetime.now()

    if recent_days and layer == 'L1':
        fact_dir = os.path.join(DATA_OUTPUT_ROOT, 'fact')
        _clear_directory(fact_dir, 'fact')

        for i in range(1, len(layer_order)):
            l = layer_order[i]
            dir_name = layer_dirs[l]
            dir_path = os.path.join(DATA_OUTPUT_ROOT, dir_name)
            _clear_directory(dir_path, dir_name)
    else:
        for i in range(start_idx, len(layer_order)):
            l = layer_order[i]
            dir_name = layer_dirs[l]
            dir_path = os.path.join(DATA_OUTPUT_ROOT, dir_name)
            _clear_directory(dir_path, dir_name)

    print()

    if start_idx <= 0:
        process_fact_layer(target_date=None, skip_incomplete=skip_incomplete, recent_days=recent_days)
    if start_idx <= 1:
        process_snapshot_layer(target_date=None)
    if start_idx <= 2:
        process_lifecycle_layer(target_date=None)
    if start_idx <= 3:
        process_report_layer(target_date=None)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  从 {layer} 层重建完成，总耗时: {elapsed:.2f} 秒")




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
    parser.add_argument(
        '--layer',
        type=str,
        default=None,
        choices=['L1', 'L2', 'L3', 'L4'],
        help='从指定层开始重建，清除该层及下游数据后重算 (L1=全量, L2=跳过Fact, L3=跳过Fact+Snapshot, L4=仅报告)'
    )
    parser.add_argument(
        '--skip-incomplete',
        type=lambda x: x.lower() not in ('false', 'no', '0', 'off'),
        default=True,
        nargs='?',
        const=True,
        help='跳过数据不完整的日期（默认开启）。使用 --skip-incomplete false 关闭'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=None,
        help='仅处理最近N天的源数据（仅对 --layer L1 生效），避免全量重算时 OOM。例如 --layer L1 --days 7'
    )

    args = parser.parse_args()

    # 确定日志标签
    log_mode = f"L{args.layer}" if args.layer else args.mode
    log_path = setup_pipeline_logging(log_mode)

    exit_code = 0
    try:
        print(f"📋 日志文件: {log_path}")
        print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        if args.layer:
            run_from_layer(args.layer, target_date=args.date, skip_incomplete=args.skip_incomplete, recent_days=args.days)
        else:
            mode_map = {
                'full': run_full_pipeline,
                'incremental': run_incremental,
                'refresh': run_refresh,
                'recalculate': run_recalculate,
                'report_only': run_report_only,
            }
            pipeline_func = mode_map[args.mode]
            pipeline_func(target_date=args.date, skip_incomplete=args.skip_incomplete)

        print(f"\n✅ 流水线执行成功")

    except Exception as e:
        exit_code = 1
        print(f"\n❌ 流水线执行失败: {e}")
        print(traceback.format_exc())

    finally:
        print(f"🕐 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 日志已保存: {log_path}")

        # 恢复 stdout 并关闭日志文件
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()