# -*- coding: utf-8 -*-
"""
===============================================================================
ML标注工具 v1.0 — 人工标注管理 + 标注数据积累
===============================================================================
职责：管理人工标注流程，积累标注数据用于监督学习。

功能：
  1. 生成待标注CSV — 从异常检测结果中提取需要人工审核的用户
  2. 加载已有标注 — 读取历史标注数据，避免重复标注
  3. 保存标注结果 — 追加到标注库，不覆盖历史
  4. 标注统计 — 查看标注进度、标注一致性等

标注类型（多分类）：
  - 正常           : 模型误判，用户行为正常
  - 改装/超速      : 速度或电流异常，疑似改装车
  - 地摊/储能      : 怠速放电占比高，行驶距离短
  - 电池老化       : 频繁换电、SOC消耗快、续航短
  - 暴力驾驶       : 频繁超100A、高温、高速
  - 其他异常       : 不属于以上类别但确实异常
  - 不确定         : 无法判断，需要更多信息

使用方式：
    # 生成待标注文件
    python src/ml_labeler.py generate --input ./data/output/ml_anomaly/anomaly_report_xxx.csv

    # 查看标注统计
    python src/ml_labeler.py stats

    # 导出标注数据（用于训练）
    python src/ml_labeler.py export --output ./data/output/ml_anomaly/labeled_dataset.csv
===============================================================================
"""

import os
import sys
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import DATA_OUTPUT_ROOT

LABEL_OUTPUT_DIR = os.path.join(DATA_OUTPUT_ROOT, 'ml_anomaly', 'labels')
LABEL_STORE_PATH = os.path.join(LABEL_OUTPUT_DIR, 'labeled_data.csv')
LABEL_META_PATH = os.path.join(LABEL_OUTPUT_DIR, 'label_meta.json')

LABEL_CATEGORIES = {
    '正常': '模型误判，用户行为正常',
    '改装/超速': '速度或电流异常，疑似改装车',
    '地摊/储能': '怠速放电占比高，行驶距离短',
    '电池老化': '频繁换电、SOC消耗快、续航短',
    '暴力驾驶': '频繁超100A、高温、高速',
    '其他异常': '不属于以上类别但确实异常',
    '不确定': '无法判断，需要更多信息',
}

LABEL_PRIORITY_ORDER = [
    '改装/超速', '暴力驾驶', '地摊/储能', '电池老化', '其他异常', '正常', '不确定',
]

ANOMALY_EXPLANATION_COLS = [
    '近7d最高速度_kmh', '近7d最大电流_A', '近7d日均超100A连续次数',
    '近7d百公里电耗_kWh', '近7d最高温度_℃', '近7d_SOC低于20%时长占比',
    '近7d_SOC低于10%时长占比', '近7d日均换电次数', '近7d最低SOC',
    '近7d单合约日均怠速放电_h', '近7d单合约日均放电时长_h',
    '近7d日均高速骑行点数', '近7d夜间骑行均速_kmh',
    '近7d深夜换电占比', '近7d单合约日均行驶里程_km',
    '近7d平均骑行速度_kmh', '近7d平均骑行电流_A',
]


class LabelManager:
    """标注数据管理器"""

    def __init__(self):
        os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)
        self._labeled_df: Optional[pd.DataFrame] = None

    def _load_existing_labels(self) -> pd.DataFrame:
        """加载已有标注数据"""
        if os.path.exists(LABEL_STORE_PATH):
            df = pd.read_csv(LABEL_STORE_PATH, encoding='utf-8-sig')
            if '用户id' in df.columns:
                df['用户id'] = df['用户id'].astype(str)
            return df
        return pd.DataFrame()

    def generate_labeling_sheet(
        self,
        anomaly_report_path: str,
        n_top: int = 50,
        only_new_findings: bool = True,
    ) -> str:
        """
        生成待标注CSV

        参数:
            anomaly_report_path: 异常检测报告路径
            n_top: 取Top-N最异常用户
            only_new_findings: 是否只取"规则判正常但模型判异常"的用户

        返回:
            生成的CSV文件路径
        """
        df_report = pd.read_csv(anomaly_report_path, encoding='utf-8-sig')
        if '用户id' in df_report.columns:
            df_report['用户id'] = df_report['用户id'].astype(str)

        existing = self._load_existing_labels()
        existing_ids = set(existing['用户id'].tolist()) if len(existing) > 0 else set()

        if only_new_findings and '风险标签' in df_report.columns:
            candidates = df_report[
                (df_report['ml_异常标签_iforest'] == -1) &
                (df_report['风险标签'] == '正常')
            ].copy()
        else:
            candidates = df_report[df_report['ml_异常标签_iforest'] == -1].copy()

        candidates = candidates[~candidates['用户id'].isin(existing_ids)]

        if 'ml_异常分数_iforest' in candidates.columns:
            candidates = candidates.sort_values('ml_异常分数_iforest')

        candidates = candidates.head(n_top)

        if len(candidates) == 0:
            print("没有新的待标注用户。所有异常用户已标注完毕。")
            return ""

        sheet_cols = ['用户id', 'ml_异常分数_iforest', 'ml_异常程度', 'ml_双模型异常']
        extra_cols = []
        for col in ['风险标签', '用户等级_动态', '用户形态', '用户生命周期状态_7d',
                     '近7d出勤天数', '近7d出勤率', '重评分数']:
            if col in candidates.columns:
                extra_cols.append(col)

        for col in ANOMALY_EXPLANATION_COLS:
            if col in candidates.columns:
                extra_cols.append(col)

        sheet_cols += extra_cols
        sheet_cols = [c for c in sheet_cols if c in candidates.columns]

        df_sheet = candidates[sheet_cols].copy()
        df_sheet['人工标注'] = ''
        df_sheet['标注备注'] = ''
        df_sheet['标注时间'] = ''

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(LABEL_OUTPUT_DIR, f'to_label_{timestamp}.csv')
        df_sheet.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"待标注文件已生成: {output_path}")
        print(f"待标注用户数: {len(df_sheet)}")
        print(f"已有标注用户数: {len(existing_ids)}")
        print(f"\n标注类别说明:")
        for cat, desc in LABEL_CATEGORIES.items():
            print(f"  [{cat}] - {desc}")

        return output_path

    def save_labels(self, labeled_csv_path: str) -> int:
        """
        保存标注结果到标注库

        参数:
            labeled_csv_path: 人工填写后的CSV路径

        返回:
            新增标注数量
        """
        df_new = pd.read_csv(labeled_csv_path, encoding='utf-8-sig')
        if '用户id' in df_new.columns:
            df_new['用户id'] = df_new['用户id'].astype(str)

        if '人工标注' not in df_new.columns:
            raise ValueError("CSV中缺少'人工标注'列，请确认已填写标注")

        df_labeled = df_new[df_new['人工标注'].notna() & (df_new['人工标注'] != '')].copy()

        invalid_labels = df_labeled[~df_labeled['人工标注'].isin(LABEL_CATEGORIES.keys())]
        if len(invalid_labels) > 0:
            bad_values = invalid_labels['人工标注'].unique().tolist()
            raise ValueError(f"存在无效标注值: {bad_values}，有效值为: {list(LABEL_CATEGORIES.keys())}")

        if '标注时间' not in df_labeled.columns or df_labeled['标注时间'].isna().all():
            df_labeled['标注时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        existing = self._load_existing_labels()

        if len(existing) > 0:
            existing_ids = set(existing['用户id'].tolist())
            df_labeled = df_labeled[~df_labeled['用户id'].isin(existing_ids)]

        if len(df_labeled) == 0:
            print("没有新的标注数据需要保存（所有用户已标注过）。")
            return 0

        keep_cols = ['用户id', '人工标注', '标注备注', '标注时间']
        for col in ANOMALY_EXPLANATION_COLS:
            if col in df_labeled.columns:
                keep_cols.append(col)
        for col in ['ml_异常分数_iforest', 'ml_异常程度', '风险标签', '用户等级_动态']:
            if col in df_labeled.columns:
                keep_cols.append(col)

        keep_cols = [c for c in keep_cols if c in df_labeled.columns]
        df_labeled = df_labeled[keep_cols]

        if len(existing) > 0:
            existing_cols = [c for c in keep_cols if c in existing.columns]
            df_merged = pd.concat([existing[existing_cols], df_labeled], ignore_index=True)
        else:
            df_merged = df_labeled

        df_merged.to_csv(LABEL_STORE_PATH, index=False, encoding='utf-8-sig')

        self._update_meta()

        print(f"标注已保存: {LABEL_STORE_PATH}")
        print(f"本次新增: {len(df_labeled)} 条")
        print(f"累计标注: {len(df_merged)} 条")

        return len(df_labeled)

    def get_stats(self) -> Dict:
        """获取标注统计"""
        existing = self._load_existing_labels()

        if len(existing) == 0:
            return {'total': 0, 'by_category': {}, 'message': '暂无标注数据'}

        stats = {
            'total': len(existing),
            'by_category': existing['人工标注'].value_counts().to_dict(),
            'labeled_users': existing['用户id'].tolist(),
        }

        if '标注时间' in existing.columns:
            existing['标注时间'] = pd.to_datetime(existing['标注时间'], errors='coerce')
            valid_times = existing['标注时间'].dropna()
            if len(valid_times) > 0:
                stats['first_label_time'] = valid_times.min().strftime('%Y-%m-%d')
                stats['last_label_time'] = valid_times.max().strftime('%Y-%m-%d')

        return stats

    def export_labeled_dataset(self, output_path: Optional[str] = None) -> str:
        """
        导出标注数据集（用于训练监督模型）

        返回:
            导出文件路径
        """
        existing = self._load_existing_labels()

        if len(existing) == 0:
            raise ValueError("暂无标注数据，请先完成标注")

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(LABEL_OUTPUT_DIR, f'labeled_dataset_{timestamp}.csv')

        existing.to_csv(output_path, index=False, encoding='utf-8-sig')

        stats = self.get_stats()
        print(f"标注数据集已导出: {output_path}")
        print(f"总标注数: {stats['total']}")
        print(f"类别分布:")
        for cat in LABEL_PRIORITY_ORDER:
            count = stats['by_category'].get(cat, 0)
            if count > 0:
                print(f"  [{cat}]: {count} 条")

        return output_path

    def _update_meta(self):
        """更新标注元数据"""
        stats = self.get_stats()
        meta = {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_labels': stats['total'],
            'categories': LABEL_CATEGORIES,
            'label_store': LABEL_STORE_PATH,
        }
        with open(LABEL_META_PATH, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def print_stats(self):
        """打印标注统计"""
        stats = self.get_stats()

        print(f"\n{'='*50}")
        print("标注数据统计")
        print(f"{'='*50}")

        if stats['total'] == 0:
            print("暂无标注数据。")
            print(f"\n生成待标注文件: python src/ml_labeler.py generate --input <异常报告路径>")
            return

        print(f"累计标注: {stats['total']} 条")
        if 'first_label_time' in stats:
            print(f"标注时间: {stats['first_label_time']} ~ {stats['last_label_time']}")

        print(f"\n类别分布:")
        for cat in LABEL_PRIORITY_ORDER:
            count = stats['by_category'].get(cat, 0)
            bar = '█' * min(count, 40)
            print(f"  [{cat}]: {count:>4} {bar}")

        anomaly_count = sum(
            v for k, v in stats['by_category'].items()
            if k not in ['正常', '不确定']
        )
        normal_count = stats['by_category'].get('正常', 0)
        uncertain_count = stats['by_category'].get('不确定', 0)
        print(f"\n确认异常: {anomaly_count} | 确认正常(误判): {normal_count} | 不确定: {uncertain_count}")
        if stats['total'] > 0:
            precision = anomaly_count / (anomaly_count + normal_count) * 100 if (anomaly_count + normal_count) > 0 else 0
            print(f"模型准确率(排除不确定): {precision:.1f}%")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML标注工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    gen_parser = subparsers.add_parser('generate', help='生成待标注CSV')
    gen_parser.add_argument('--input', type=str, required=True,
                            help='异常检测报告CSV路径')
    gen_parser.add_argument('--n-top', type=int, default=50,
                            help='取Top-N最异常用户 (默认: 50)')
    gen_parser.add_argument('--all-anomalies', action='store_true',
                            help='包含所有异常用户，不限于规则判正常的')

    save_parser = subparsers.add_parser('save', help='保存标注结果')
    save_parser.add_argument('--input', type=str, required=True,
                             help='人工填写后的CSV路径')

    subparsers.add_parser('stats', help='查看标注统计')

    export_parser = subparsers.add_parser('export', help='导出标注数据集')
    export_parser.add_argument('--output', type=str, default=None,
                               help='导出路径 (默认自动生成)')

    args = parser.parse_args()

    manager = LabelManager()

    if args.command == 'generate':
        manager.generate_labeling_sheet(
            anomaly_report_path=args.input,
            n_top=args.n_top,
            only_new_findings=not args.all_anomalies,
        )
    elif args.command == 'save':
        manager.save_labels(labeled_csv_path=args.input)
    elif args.command == 'stats':
        manager.print_stats()
    elif args.command == 'export':
        manager.export_labeled_dataset(output_path=args.output)
    else:
        parser.print_help()