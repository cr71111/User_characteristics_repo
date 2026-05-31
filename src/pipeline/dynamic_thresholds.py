# -*- coding: utf-8 -*-
"""
===============================================================================
两轮车换电分析 · 动态自适应阈值体系 v1.0
===============================================================================
位置：D:/PY代码/Battery_data/battery_repo/src/scheduled_tasks/daily_2/
        dynamic_thresholds.py

设计原则：
  1. 分位数动态计算（不复用历史阈值）
  2. 基准线持久化（BaselineStore）
  3. 分布漂移检测（DistributionShiftDetector）
  4. 渐进式阈值更新（EMA 平滑）
  5. 零人工干预：阈值从数据中来，到评级里去

使用方式（集成到用户生命周期管理.py）：
  from src.pipeline.dynamic_thresholds import DynamicBatteryAnalyzer
  analyzer = DynamicBatteryAnalyzer(baseline_path='./src/tools/thresholds_baseline.json')
  thresholds, baseline = analyzer.run(df_lifecycle)
  df = analyzer.score_and_classify(df_lifecycle, thresholds, baseline)
"""

import json, os, warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ==============================================================================
# 1. BaselineStore — 基准线持久化
# ==============================================================================

class BaselineStore:
    """
    管理阈值基准线的存储、读取与版本追踪。

    baseline 结构（JSON）：
    {
      "version": "v1.0",
      "updated_at": "2026-04-17T13:35:00",
      "data_hash": "sha256_...",        # 数据指纹，变动时提示更新
      "sample_size": 21006,
      "quantiles": {
        "avg_current":    {"P75":22.3,"P90":28.3,"P95":32.1,"P99":40.4},
        "max_current":     {"P75":57.9,"P90":64.7,"P95":68.9,"P99":81.9},
        "energy_per_100km":{"P75":10.6,"P90":13.8,"P95":17.0,"P99":32.9},
        "daily_changes":   {"P75": 2.1,"P90": 3.3,"P95": 3.9,"P99": 5.1},
        "soc_below_10":    {"P75": 0.0,"P90": 0.0,"P95": 0.0,"P99": 0.01},
        "soc_below_20":    {"P75": 0.02,"P90":0.07,"P95":0.12,"P99":0.26},
        "monthly_score":  {"P25":52.3,"P50":69.4,"P75":82.1,"P90":93.9},
        "daily_ride_hours":{"P75": 3.2,"P90": 4.6,"P95": 5.5,"P99": 7.8},
      },
      "user_level_thresholds": {
        "excellent": 82,    # P75
        "good":      69,     # P50
        "normal":    52,     # P25
      },
      "ema_baseline": { ... }   # EMA 平滑后的基准线（可选）
    }
    """

    def __init__(self, path: str = './src/tools/thresholds_baseline.json'):
        self.path = path
        self._data: Optional[Dict] = None

    # ── 读 ──────────────────────────────────────────────────────────────────

    def load(self) -> Optional[Dict]:
        """读取基准线文件。文件不存在或损坏时返回 None。"""
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            return self._data
        except (json.JSONDecodeError, IOError) as e:
            warnings.warn(f'基准线读取失败 [{self.path}]: {e}')
            return None

    def is_fresh(self, data_hash: str, max_age_days: int = 30) -> bool:
        """判断基准线是否新鲜（未过期）。"""
        b = self.load()
        if b is None:
            return False
        try:
            updated = datetime.fromisoformat(b['updated_at'])
            age = (datetime.now() - updated).days
            return age <= max_age_days and b.get('data_hash') == data_hash
        except Exception:
            return False

    # ── 写 ──────────────────────────────────────────────────────────────────

    def save(self, quantiles: Dict, sample_size: int,
             data_hash: str = '', extra: Optional[Dict] = None) -> None:
        """保存基准线到文件。"""
        record = {
            'version': 'v1.0',
            'updated_at': datetime.now().isoformat(),
            'data_hash': data_hash,
            'sample_size': sample_size,
            'quantiles': quantiles,
            'user_level_thresholds': {
                'excellent': round(float(quantiles['monthly_score']['P75']), 1),
                'good':      round(float(quantiles['monthly_score']['P50']), 1),
                'normal':    round(float(quantiles['monthly_score']['P25']), 1),
            },
        }
        if extra:
            record.update(extra)
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self._data = record

    # ── EMA 平滑更新（可选）────────────────────────────────────────────────

    def update_ema(self, key_path: str, new_value: float,
                   alpha: float = 0.3) -> float:
        """
        对指定阈值做 EMA 平滑更新，避免单次数据波动导致基准线剧烈跳变。

        Example: update_ema('quantiles.max_current.P95', 69.5)
        Returns: 平滑后的新值
        """
        b = self.load()
        if b is None:
            return new_value

        keys = key_path.split('.')
        node = b
        for k in keys[:-1]:
            node = node.setdefault(k, {})

        old_val = node.get(keys[-1], new_value)
        smoothed = alpha * new_value + (1 - alpha) * old_val
        node[keys[-1]] = round(smoothed, 3)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(b, f, ensure_ascii=False, indent=2)
        return round(smoothed, 3)

    @property
    def baseline(self) -> Optional[Dict]:
        return self._data


# ==============================================================================
# 2. DistributionShiftDetector — 分布漂移检测
# ==============================================================================

class DistributionShiftDetector:
    """
    检测实时数据与基准线之间是否存在显著分布漂移。

    漂移判定逻辑：
      - 对关键指标（电流、电耗、SOC）计算实时 P90
      - 与基准线 P90 对比，相对偏差 > SHIFT_THRESHOLD_PCT → 触发告警
      - 偏差方向：正向（数据恶化）vs 负向（数据改善）

    输出：
      - shift_alert: bool
      - shift_report: dict of per-metric deviations
      - suggestion: str (调整建议)
    """

    # 偏差超过此百分比 → 认为分布发生漂移
    SHIFT_THRESHOLD_PCT = 0.15    # 15% 相对偏差

    def __init__(self, baseline: Dict):
        self.baseline = baseline
        self.bq = baseline.get('quantiles', {})

    def detect(self, live_quantiles: Dict) -> Dict:
        result = {
            'has_shift': False,
            'shift_direction': 'stable',
            'alerts': [],
            'suggestions': [],
            'details': {},
        }

        # 关键指标映射：live_key → (baseline_key, friendly_name, is_positive_bad)
        indicators = [
            ('avg_current',    'avg_current',    '平均骑行电流',  True),
            ('max_current',    'max_current',     '最大电流',      True),
            ('energy_per_100km','energy_per_100km','百公里电耗',   True),
            ('daily_changes',  'daily_changes',   '日均换电次数',  False),
        ]

        for live_key, base_key, label, is_positive_bad in indicators:
            live_q = live_quantiles.get(live_key, {})
            base_q = self.bq.get(base_key, {})

            for pct in ('P90', 'P95'):
                live_val = live_q.get(pct)
                base_val = base_q.get(pct)
                if live_val is None or base_val is None or base_val == 0:
                    continue

                deviation = (live_val - base_val) / base_val   # 相对偏差
                abs_dev   = abs(deviation)

                result['details'][f'{label}_{pct}'] = {
                    'baseline': round(base_val, 3),
                    'live':     round(live_val, 3),
                    'deviation': round(deviation * 100, 1),   # 百分比
                }

                if abs_dev > self.SHIFT_THRESHOLD_PCT:
                    result['has_shift'] = True
                    direction = '恶化' if (deviation > 0 and is_positive_bad) or (deviation < 0 and not is_positive_bad) else '改善'
                    alert = (f'[WARN] [{label}] 实时{pct}({live_val:.2f}) '
                             f'相对基准({base_val:.2f})变化 {deviation*100:+.1f}%，'
                             f'分布{direction}')
                    result['alerts'].append(alert)
                    result['suggestions'].append(
                        f'{label}分布{direction}，建议{"上调" if direction=="恶化" else "下调"}对应阈值 '
                        f'+{int(deviation*100)}%'
                    )

        # 汇总方向
        if not result['has_shift']:
            result['shift_direction'] = 'stable'
        else:
            worsening = any('恶化' in a for a in result['alerts'])
            improving  = any('改善' in a for a in result['alerts'])
            if worsening and improving:
                result['shift_direction'] = 'mixed'
            elif worsening:
                result['shift_direction'] = 'worsening'
            else:
                result['shift_direction'] = 'improving'

        return result


# ==============================================================================
# 3. AdaptiveThresholds — 动态阈值计算引擎
# ==============================================================================

class AdaptiveThresholds:
    """
    从原始 DataFrame 动态计算所有阈值。

    计算策略：
      - 只用有效正值数据计算分位数
      - 分位数列表可配置（P10/P25/P50/P75/P90/P95/P99）
      - 对极端极值（>99.5%分位）做 Winsorize 截断，避免异常值扭曲阈值
    """

    WINSORIZE_LEVEL = 0.005   # 截断最极端的 0.5% 数据

    def __init__(self, column_map: Optional[Dict] = None):
        """
        column_map: 实际列名 → 逻辑列名 的映射表。
                    若为 None，使用内置默认映射。
        """
        self.column_map = column_map or {
            '近7d平均骑行电流_A':       'avg_current',
            '近7d最大电流_A':           'max_current',
            '近7d百公里电耗_kWh':       'energy_per_100km',
            '近7d日均换电次数':         'daily_changes',
            '近7d_SOC低于10%时长占比': 'soc_below_10',
            '近7d_SOC低于20%时长占比': 'soc_below_20',
            '近7d平均包月友好分':       'monthly_score',
            '近7d单合约日均骑行时长_h':  'daily_ride_hours',
        }
        # 反向映射
        self.reverse_map = {v: k for k, v in self.column_map.items()}

    # ── 核心计算 ─────────────────────────────────────────────────────────────

    def compute(self, df: pd.DataFrame,
                percentiles: List[float] = [0.10, 0.25, 0.45, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.99]
                ) -> Dict:
        """
        从 DataFrame 计算所有分位数阈值。

        Returns:
            Dict[str, Dict[str, float]]
            e.g. {
              'avg_current': {'P10':..., 'P25':..., ..., 'P99':...},
              ...
            }
        """
        results: Dict[str, Dict[str, float]] = {}

        for orig_col, logical in self.column_map.items():
            if orig_col not in df.columns:
                warnings.warn(f'列 [{orig_col}] 不存在，跳过阈值计算')
                continue

            s = df[orig_col].dropna()
            
            # 修复：过滤掉被硬编码为20.0的沉默用户，还原真实的得分分布
            if logical == 'monthly_score':
                s_pos = s[s != 20.0]
            else:
                s_pos = s[s > 0] if logical != 'monthly_score' else s

            if len(s_pos) < 30:
                warnings.warn(f'[{logical}] 有效样本 < 30，跳过')
                continue

            # Winsorize 截断极端值
            lo = s_pos.quantile(self.WINSORIZE_LEVEL)
            hi = s_pos.quantile(1 - self.WINSORIZE_LEVEL)
            s_clean = s_pos.clip(lo, hi)

            pct_dict = {}
            for p in percentiles:
                label = f'P{int(p*100)}'
                pct_dict[label] = round(float(s_clean.quantile(p)), 3)

            # 额外统计
            pct_dict['_mean']   = round(float(s_clean.mean()), 3)
            pct_dict['_median'] = round(float(s_clean.median()), 3)
            pct_dict['_std']    = round(float(s_clean.std()), 3)
            pct_dict['_n']      = int(len(s_pos))

            results[logical] = pct_dict

        return results

    # ── 导出为可读报告 ──────────────────────────────────────────────────────

    @staticmethod
    def format_report(quantiles: Dict) -> str:
        lines = ['=' * 60, '  动态阈值计算报告', '=' * 60]
        for key, vals in sorted(quantiles.items()):
            vals = dict(vals)
            mean  = vals.pop('_mean',   None)
            med   = vals.pop('_median', None)
            std   = vals.pop('_std',    None)
            n     = vals.pop('_n',      None)
            header = f'\n[{key}]  n={n}'
            lines.append(header)
            row = '  '.join(f'{k}={v:.2f}' for k, v in sorted(vals.items()))
            lines.append(f'  {row}')
            if mean is not None:
                lines.append(f'  均值={mean:.2f}  中位数={med:.2f}  σ={std:.2f}')
        return '\n'.join(lines)


# ==============================================================================
# 4. DynamicBatteryAnalyzer — 综合入口
# ==============================================================================

class DynamicBatteryAnalyzer:
    """
    整合 BaselineStore / AdaptiveThresholds / DistributionShiftDetector
    的统一入口，替代用户生命周期管理.py 中的静态阈值。

    用法：
      analyzer = DynamicBatteryAnalyzer(baseline_path='./src/tools/thresholds_baseline.json')
      thresholds, baseline = analyzer.run(df_lifecycle)
      df_result = analyzer.score_and_classify(df_lifecycle, thresholds, baseline)
    """

    def __init__(self,
                 baseline_path: str = './src/tools/thresholds_baseline.json',
                 column_map: Optional[Dict] = None,
                 use_ema_update: bool = False,
                 ema_alpha: float = 0.3):
        self.baseline_path = baseline_path
        self.store = BaselineStore(baseline_path)
        self.engine = AdaptiveThresholds(column_map)
        self.use_ema = use_ema_update
        self.ema_alpha = ema_alpha

    # ── 主流程 ─────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, force_refresh: bool = False
            ) -> Tuple[Dict, Dict]:
        """
        执行动态阈值计算主流程。

        逻辑分支：
          1. force_refresh=True → 强制从数据重新计算，忽略基准线
          2. 基准线新鲜且数据指纹一致 → 直接复用（零计算开销）
          3. 基准线过期或数据指纹变化 → 重新计算 + 检测漂移 + 可选 EMA 更新

        Returns:
            (live_quantiles, active_baseline)
            live_quantiles:  本次计算的实时分位数
            active_baseline: 用于实际评级的阈值（可能是 EMA 平滑后的值）
        """
        # Step 1: 尝试加载现有基准线
        baseline = self.store.load()

        # Step 2: 计算数据指纹
        fingerprint = self._data_fingerprint(df)

        # Step 3: 决策分支
        if not force_refresh and baseline:
            age_days = (datetime.now() - datetime.fromisoformat(
                baseline['updated_at'])).days if 'updated_at' in baseline else 999
            hash_ok = baseline.get('data_hash') == fingerprint
            fresh   = age_days <= 30

            if fresh and hash_ok:
                print('[DynamicThresholds] [OK] Baseline fresh and hash match, reusing')
                live_q = baseline['quantiles']
                active = baseline
                return live_q, active

            print('[DynamicThresholds] [CHANGED] Data fingerprint changed or baseline expired, recomputing thresholds')
            if not hash_ok:
                print('   Data source changed, rebuilding baseline')
            if not fresh:
                print(f'   Baseline expired ({age_days}d > 30d)')

        # Step 4: 从数据计算实时分位数
        print('[DynamicThresholds] [CALC] Computing quantiles from data...')
        live_quantiles = self.engine.compute(df)
        live_baseline = {
            'version': 'live',
            'quantiles': live_quantiles,
            'sample_size': len(df),
            'data_hash': fingerprint,
        }

        # Step 5: 分布漂移检测
        if baseline:
            detector = DistributionShiftDetector(baseline)
            shift_report = detector.detect(live_quantiles)
            print()
            self._print_shift_report(shift_report)

            # Step 6: EMA 平滑更新基准线
            print(f'[DynamicThresholds] [DEBUG] EMA配置状态 - use_ema: {self.use_ema}')
            print(f'[DynamicThresholds] [DEBUG] 分布漂移检测结果 - has_shift: {shift_report.get("has_shift")}')
            
            if self.use_ema and shift_report.get('has_shift'):
                print('[DynamicThresholds] [EMA] ✅ 检测到分布漂移，执行EMA平滑更新...')
                for suggestion in shift_report.get('suggestions', []):
                    print(f'   -> {suggestion}')
                self._ema_update_baseline(live_quantiles)
                active = self.store.load()
                print('[DynamicThresholds] [EMA] ✅ 基准线文件已更新')
            elif self.use_ema and not shift_report.get('has_shift'):
                print('[DynamicThresholds] [EMA] ℹ️ 未检测到分布漂移，保持当前基准线不变')
                active = live_baseline
            else:
                print('[DynamicThresholds] [EMA] ⚠️ EMA功能未启用，使用实时计算的基准线')
                active = live_baseline
        else:
            print('[DynamicThresholds] [NEW] No history baseline, creating initial baseline')
            self.store.save(
                quantiles=live_quantiles,
                sample_size=len(df),
                data_hash=fingerprint,
            )
            active = self.store.load()

        return live_quantiles, (active or live_baseline)

    # ── 评分 & 分类 ────────────────────────────────────────────────────────
    # 注意：此方法仅用于 dynamic_thresholds.py 独立运行模式。
    # Pipeline 中统一使用 score_layer.score_and_classify() 进行评分分类。

    def score_and_classify(
        self,
        df: pd.DataFrame,
        live_quantiles: Dict,
        active_baseline: Dict,
    ) -> pd.DataFrame:
        """
        基于动态阈值对 DataFrame 进行评分和用户等级分类。

        【已委托】此方法已委托给 score_layer.score_and_classify()，
        确保 Pipeline 和独立运行模式使用完全一致的评分逻辑。
        """
        from src.pipeline.score_layer import score_and_classify as _score_and_classify
        return _score_and_classify(df, live_quantiles, active_baseline)

    # ── 辅助方法 ─────────────────────────────────────────────────────────

    @staticmethod
    def _data_fingerprint(df: pd.DataFrame) -> str:
        """简单数据指纹：基于行数+数值列均值和，快速判断数据是否变化。"""
        import hashlib
        # 只取数值列，避免日期/字符串列干扰
        numeric_cols = df.select_dtypes(include='number').columns
        mean_sum = df[numeric_cols].mean().sum() if len(numeric_cols) > 0 else 0
        key = f'{len(df)}_{len(df.columns)}_{mean_sum:.4f}'
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _ema_update_baseline(self, live_quantiles: Dict) -> None:
        """将实时分位数以 EMA 方式更新到基准线文件。"""
        for metric, vals in live_quantiles.items():
            for pct, val in vals.items():
                if pct.startswith('_'):
                    continue
                self.store.update_ema(
                    f'quantiles.{metric}.{pct}', val, self.ema_alpha)

    @staticmethod
    def _print_shift_report(report: Dict) -> None:
        print('[Distribution Shift Detection Report]')
        if not report.get('has_shift'):
            print('  [OK] No significant shift detected')
            return
        print(f'  [WARN] Distribution shift detected, direction: {report["shift_direction"]}')
        for a in report.get('alerts', []):
            print(f'  {a}')
        if report.get('suggestions'):
            print('  Suggestions:')
            for s in report['suggestions']:
                print(f'    -> {s}')


# ==============================================================================
# 5. 独立运行入口（用于调试和演示）
# ==============================================================================

if __name__ == '__main__':
    import argparse, pathlib

    parser = argparse.ArgumentParser(description='动态自适应阈值分析')
    parser.add_argument('--data',   default='E:/test/用户行为习惯/用户特征画像/全量用户7天滚动生命周期档案.csv')
    parser.add_argument('--output', default='E:/test/换电运营分析输出/')
    parser.add_argument('--baseline', default='./src/tools/thresholds_baseline.json')
    parser.add_argument('--force-refresh', action='store_true')
    parser.add_argument('--use-ema', action='store_true',
                        help='启用 EMA 平滑更新基准线')
    args = parser.parse_args()

    # 加载数据
    data_icon = '[DATA]' if os.name=='nt' else '\U0001f4c2'
    print(f'{data_icon} Loading: {args.data}')
    try:
        df = pd.read_csv(args.data, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(args.data, encoding='gbk')
    print(f'   Records: {len(df)}\n')

    # 运行分析
    analyzer = DynamicBatteryAnalyzer(
        baseline_path=args.baseline,
        use_ema_update=args.use_ema,
        ema_alpha=0.3,
    )
    live_q, active_b = analyzer.run(df, force_refresh=args.force_refresh)

    # 打印阈值报告
    report_icon = '[REPORT]' if os.name=='nt' else '\U0001f4cb'
    print(f'\n{report_icon}')
    print(AdaptiveThresholds.format_report(live_q))

    # 评分 & 分类
    print('\n[评分与分类]')
    result = analyzer.score_and_classify(df, live_q, active_b)

    # 对比统计
    total = len(result)
    sep = '=' * 55
    print(f'\n{sep}')
    print(f'  动态阈值分析 · 用户等级分布（n={total}）')
    print(f'{sep}')

    cols = ['用户等级_动态', '策略建议']
    if '用户等级_综合_7d' in result.columns:
        cols.append('等级变化')
    for col in cols:
        print(f'\n[{col}]')
        for k, v in result[col].value_counts().items():
            print(f'  {k:<12}: {v:>5} 人  ({v/total*100:.1f}%)')

    # 保存
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, '动态阈值分析_用户明细.csv')
    keep_cols = ['user_id','用户等级_综合_7d','重评分数','用户等级_动态',
                 '风险标签','策略建议']
    if '等级变化' in result.columns:
        keep_cols.append('等级变化')
    keep_cols = [c for c in keep_cols if c in result.columns]
    result[keep_cols].to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'\n[OK] Result saved: {out_path}')
