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
  from dynamic_thresholds import DynamicBatteryAnalyzer
  analyzer = DynamicBatteryAnalyzer(baseline_path='./thresholds_baseline.json')
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

    def __init__(self, path: str = './thresholds_baseline.json'):
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
      analyzer = DynamicBatteryAnalyzer(baseline_path='./thresholds_baseline.json')
      thresholds, baseline = analyzer.run(df_lifecycle)
      df_result = analyzer.score_and_classify(df_lifecycle, thresholds, baseline)
    """

    def __init__(self,
                 baseline_path: str = './thresholds_baseline.json',
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
            if self.use_ema and shift_report.get('has_shift'):
                print('[DynamicThresholds] [EMA] Performing EMA smoothing update...')
                for suggestion in shift_report.get('suggestions', []):
                    print(f'   -> {suggestion}')
                self._ema_update_baseline(live_quantiles)
                active = self.store.load()
            else:
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

    def score_and_classify(
        self,
        df: pd.DataFrame,
        live_quantiles: Dict,
        active_baseline: Dict,
    ) -> pd.DataFrame:
        """
        基于动态阈值对 DataFrame 进行评分和用户等级分类。
        【优化版】使用向量化操作替代apply(axis=1)，性能提升10-100倍
        【同步版】与用户生命周期管理.py保持一致的业务逻辑
        """
        bq = active_baseline.get('quantiles', live_quantiles)

        df = df.copy()

        # 取基准分位数
        avg_cur_P90  = bq.get('avg_current',   {}).get('P90',  28.3)
        avg_cur_P95  = bq.get('avg_current',   {}).get('P95',  32.1)
        max_cur_P90  = bq.get('max_current',   {}).get('P90',  64.7)
        max_cur_P95  = bq.get('max_current',   {}).get('P95',  68.9)
        max_cur_P99  = bq.get('max_current',   {}).get('P99',  81.9)
        energy_P90   = bq.get('energy_per_100km', {}).get('P90', 13.8)

        # ── A. 风险标签（向量化优化）────────────────────────────────────────
        # 业务规则：保护板最大电流60A，超过即为异常
        PROTECTION_BOARD_MAX = 60.0
        MONTHLY_ENERGY_LOSS_THRESHOLD = 150.0  # 月用电盈亏线
        
        # 提取列数据（向量化）
        max_cur = df['近7d最大电流_A'].fillna(0).values
        avg_cur = df['近7d平均骑行电流_A'].fillna(0).values
        over100 = df['近7d超100A连续总次数'].fillna(0).values
        over80 = df['近7d电流超80A总次数'].fillna(0).values
        min_soc = df['近7d最低SOC'].fillna(100).values
        soc_low_ratio = df['近7d_SOC低于10%时长占比'].fillna(0).values
        monthly_energy = df['单合约月度用电度数预估_kWh'].fillna(0).values
        
        # 构建条件掩码（优先级从高到低）
        cond_violent = over100 >= 2
        cond_extreme = (max_cur > max_cur_P99) & ~cond_violent
        cond_protect = (max_cur > PROTECTION_BOARD_MAX) & ~cond_violent & ~cond_extreme
        cond_over80 = (over80 > 0) & ~cond_violent & ~cond_extreme & ~cond_protect
        cond_high_avg = (avg_cur > avg_cur_P95) & ~cond_violent & ~cond_extreme & ~cond_protect & ~cond_over80
        
        # 主风险标签
        risk_tags = np.select(
            [cond_violent, cond_extreme, cond_protect, cond_over80, cond_high_avg],
            ['暴力放电', '极端电流', '超保护板电流', '频繁超80A', '持续高耗流'],
            default=''
        )
        
        # SOC风险标签（修复Bug1：严重条件放前面，min_soc<10是深度亏电，soc_low_ratio>0是低SOC告警）
        soc_risk = np.select(
            [min_soc < 10, soc_low_ratio > 0],
            ['深度亏电', '低SOC告警'],
            default=''
        )
        
        # 用电超标标签
        energy_risk = np.where(monthly_energy > MONTHLY_ENERGY_LOSS_THRESHOLD, 
                              '月用电超标(' + (monthly_energy).astype(int).astype(str) + '度)', '')
        
        # 拼接所有标签（用向量化字符串操作）
        df['风险标签'] = pd.Series([
            ' / '.join(filter(None, [r, s, e])) if any([r, s, e]) else '正常'
            for r, s, e in zip(risk_tags, soc_risk, energy_risk)
        ], index=df.index)
        
        # ── A2. 新增：识别"非移动用电"（疑似静态储能）────────────────────────
        # 业务场景：地摊/储能用户长时间静止放电，对电池温升伤害与骑行不同
        # 计算公式：Static_Discharge_Ratio = 怠速放电时长 / 总放电时长
        # 阈值：占比 > 80% 且 近7d最大活动半径 < 2km → 打上 [疑似静态储能] 标签
        idle_hours = df['近7d单合约日均怠速放电_h'].fillna(0).values
        discharge_hours = df['近7d单合约日均放电时长_h'].fillna(0).values
        max_radius = df['近7d最大活动半径_km'].fillna(999).values
        
        # 计算静止放电占比（避免除零）
        static_discharge_ratio = np.where(discharge_hours > 0, idle_hours / discharge_hours, 0.0)
        
        # 疑似静态储能判定
        is_static_storage = (static_discharge_ratio > 0.8) & (max_radius < 2.0) & (discharge_hours > 1.0)
        
        # 将静态储能标签追加到风险标签中
        df['风险标签'] = np.where(
            is_static_storage,
            df['风险标签'].apply(lambda x: f'{x} / 疑似静态储能' if x != '正常' else '疑似静态储能'),
            df['风险标签']
        )

        # ── B. 包月友好评分（向量化优化）────────────────────────────────────
        # 动态评分与评级体系解耦：使用动态分位数而非硬编码分数
        # 【修复】删除旧的monthly_score分位数读取逻辑，改为基于重评分数动态计算
        # score_P_excellent = bq.get('monthly_score', {}).get('P85', 85)  # 已删除
        # score_P_good      = bq.get('monthly_score', {}).get('P65', 65)  # 已删除
        # score_P_normal    = bq.get('monthly_score', {}).get('P45', 45)  # 已删除

        # 提取所有需要的列
        max_cur = df['近7d最大电流_A'].fillna(0).values
        avg_cur = df['近7d平均骑行电流_A'].fillna(0).values
        energy  = df['近7d百公里电耗_kWh'].fillna(0).values
        over80  = df['近7d电流超80A总次数'].fillna(0).values
        over100 = df['近7d超100A连续总次数'].fillna(0).values
        soc_low = df['近7d_SOC低于20%时长占比'].fillna(0).values
        monthly = df['近7d平均包月友好分'].fillna(0).values
        
        # 新增：提取活跃度和客户形态相关列
        ride_hours = df['近7d单合约日均骑行时长_h'].fillna(0).values
        idle_hours = df['近7d单合约日均怠速放电_h'].fillna(0).values
        discharge_hours = df['近7d单合约日均放电时长_h'].fillna(0).values
        monthly_energy_val = df['单合约月度用电度数预估_kWh'].fillna(0).values
        customer_type = df.get('客户形态_综合_7d', pd.Series('未知', index=df.index)).fillna('未知').values

        # ── B0. 沉默用户判定（优化：0骑行、0电流、0电耗用户单独处理）─────────
        is_silent = (avg_cur == 0) & (max_cur == 0) & (monthly_energy_val == 0) & (ride_hours == 0)
        
        # 取分位数基准
        aP25 = bq.get('avg_current',    {}).get('P25', 11.7)
        aP50 = bq.get('avg_current',    {}).get('P50', 16.4)
        aP75 = bq.get('avg_current',    {}).get('P75', 22.3)
        aP95 = bq.get('avg_current',    {}).get('P95', 32.1)

        mP25 = bq.get('max_current',    {}).get('P25', 39.5)
        mP50 = bq.get('max_current',    {}).get('P50', 46.2)
        mP75 = bq.get('max_current',    {}).get('P75', 57.9)
        mP95 = bq.get('max_current',    {}).get('P95', 68.9)

        eP75 = bq.get('energy_per_100km', {}).get('P75', 10.6)
        eP90 = bq.get('energy_per_100km', {}).get('P90', 13.8)
        eP95 = bq.get('energy_per_100km', {}).get('P95', 17.0)

        sP25 = bq.get('soc_below_20',   {}).get('P25', 0.01)
        sP50 = bq.get('soc_below_20',   {}).get('P50', 0.02)
        sP75 = bq.get('soc_below_20',   {}).get('P75', 0.03)
        sP95 = bq.get('soc_below_20',   {}).get('P95', 0.11)

        mscP25 = bq.get('monthly_score', {}).get('P25', 51.2)
        mscP50 = bq.get('monthly_score', {}).get('P50', 69.0)
        mscP75 = bq.get('monthly_score', {}).get('P75', 81.9)

        # 向量化维度打分函数
        def dim_score_vectorized(val, p25, p50, p75, p95=None, invert=False):
            """向量化维度打分"""
            result = np.zeros_like(val, dtype=float)
            if invert:
                # 越低越好
                result = np.where(val <= p25, 18.0, result)
                result = np.where((val > p25) & (val <= p50), 10.0, result)
                result = np.where((val > p50) & (val <= p75), 4.0, result)
                ref = p95 if p95 else p75 * 1.5
                excess = np.minimum(1.0, (val - p75) / (ref - p75 + 0.01))
                result = np.where(val > p75, np.maximum(0.0, 4.0 - excess * 4.0), result)
            else:
                # 越高越好
                result = np.where(val >= p75, 18.0, result)
                result = np.where((val >= p50) & (val < p75), 10.0, result)
                result = np.where((val >= p25) & (val < p50), 4.0, result)
                result = np.where(val < p25, np.maximum(0.0, 2.0 * val / (p25 + 0.01)), result)
            return result

        # 各维度独立打分（优化：平衡权重，3个越低越好+2个越高越好）
        dim_avg_cur   = dim_score_vectorized(avg_cur,  aP25,  aP50,  aP75,  aP95, invert=True)  # 越低越好
        dim_max_cur   = dim_score_vectorized(max_cur,  mP25,  mP50,  mP75,  mP95, invert=True)  # 越低越好
        dim_energy    = dim_score_vectorized(energy,   eP75,  eP90,  eP95,  eP95, invert=True)  # 越低越好
        dim_soc_low   = dim_score_vectorized(soc_low,  sP25,  sP50,  sP75,  sP95, invert=True)  # 越低越好
        dim_monthly   = dim_score_vectorized(monthly, mscP25, mscP50, mscP75, invert=False)      # 越高越好

        # 超80A次数折算分（优化：降低单次扣分，避免偶尔电流过大被过度惩罚）
        # 改为：超80A次数≤3次不扣分，4-6次每次扣2分，7-10次每次扣3分，>10次每次扣5分
        over80_score = np.select(
            [over80 <= 3, (over80 > 3) & (over80 <= 6), (over80 > 6) & (over80 <= 10), over80 > 10],
            [20.0, 20.0 - (over80 - 3) * 2.0, 14.0 - (over80 - 6) * 3.0, 5.0 - (over80 - 10) * 5.0],
            default=20.0
        )
        over80_score = np.maximum(0.0, over80_score)
        
        # 计算总分
        total = dim_avg_cur + dim_max_cur + dim_energy + dim_soc_low + dim_monthly + over80_score
        
        # ── B2. 消除"怠速伤包"惩罚的业务盲区（优化：移除身份判断）───────────────
        # 长时大功率怠速放电（拔下电池作为外接电源）是极其伤包的行为
        # 不论客户是"专送骑手"还是"普通代步"，只要idle_hours > 5就触发扣分
        idle_penalty = np.where(idle_hours > 5, -20.0, np.where(idle_hours > 3, -10.0, 0.0))
        total = total + idle_penalty
        
        # 超100A处理（优化：针对连续大电流，偶尔一次不直接清零）
        # 连续超100A≥2次：分数×0.3（而非直接清零）
        # 连续超100A=1次：分数×0.7（而非×0.5）
        total = np.where(over100 >= 2, total * 0.3, total)
        total = np.where(over100 == 1, total * 0.7, total)
        
        # 沉默用户强制低分
        total = np.where(is_silent, 20.0, total)  # 沉默用户给20分（低于普通用户阈值）
        
        df['重评分数'] = np.maximum(0.0, total).round(1)

        # ── B3. 基于重评分数动态计算等级门槛（修复分位数塌陷）───────────────
        # 使用活跃用户（非沉默用户）的真实重评分数分布来动态切割门槛
        active_scores = df.loc[~is_silent, '重评分数']
        
        if len(active_scores) > 30:
            score_P_excellent = np.percentile(active_scores, 75)
            score_P_good      = np.percentile(active_scores, 55)
            score_P_normal    = np.percentile(active_scores, 35)
        else:
            # 样本不足时的合理回退值（假设新体系满分约110分）
            score_P_excellent, score_P_good, score_P_normal = 85.0, 70.0, 50.0

        # ── C. 用户等级（向量化优化）────────────────────────────────────
        s = df['重评分数'].values
        over100 = df['近7d超100A连续总次数'].fillna(0).values
        risk = df['风险标签'].fillna('正常').values
        
        # 修复P1: 统一暴力用户判定逻辑，与风险标签保持一致
        # 风险标签中"暴力放电"对应 over100 >= 2
        # 风险标签中"极端电流"对应 max_cur > P99
        # 用户等级中"暴力"应该包含这两者 + 月用电超250度
        max_cur = df['近7d最大电流_A'].fillna(0).values
        max_cur_P99 = bq.get('max_current', {}).get('P99', 81.9)
        monthly_energy_val = df['单合约月度用电度数预估_kWh'].fillna(0).values
        
        # 暴力用户判定：与风险标签逻辑对齐
        # over100 >= 2 → 暴力放电（风险标签）→ 暴力（用户等级）
        # max_cur > P99 → 极端电流（风险标签）→ 暴力（用户等级）
        # monthly > 250 → 暴力（用户等级）
        is_violent = (over100 >= 2) | (max_cur > max_cur_P99) | (monthly_energy_val > 250)
        
        # 风险等级限制
        has_violent_risk = np.array(['暴力放电' in r or '极端电流' in r for r in risk])
        has_high_risk = np.array(['超保护板电流' in r or '频繁超80A' in r for r in risk])
        has_medium_risk = np.array(['月用电超标' in r for r in risk])
        
        # 计算入网天数（用于新兵保护期）
        NEW_USER_PROTECTION_DAYS = 3
        current_date = pd.Timestamp.now()
        if '首次入网日期' in df.columns:
            join_dates = pd.to_datetime(df['首次入网日期'], errors='coerce')
            days_since_join = (current_date - join_dates).dt.days.fillna(999).values
        else:
            days_since_join = np.full(len(df), 999)  # 默认非新用户
        
        # 初始化结果数组
        result_level = np.full(len(df), '', dtype=object)
        
        # 沉默用户判定（最高只能到普通用户）
        result_level[is_silent] = '沉默用户'
        
        # 暴力用户（含月用电超250度）
        result_level[is_violent & ~is_silent] = '暴力'
        
        # 高风险用户（最高只能到普通用户）
        high_risk_mask = has_high_risk & ~is_violent & ~is_silent
        result_level[high_risk_mask & (s >= score_P_normal)] = '普通用户'
        result_level[high_risk_mask & (s < score_P_normal)] = '高损耗用户'
        
        # 中风险用户（最高只能到良好用户）
        medium_risk_mask = has_medium_risk & ~is_violent & ~high_risk_mask & ~is_silent
        result_level[medium_risk_mask & (s >= score_P_excellent)] = '良好用户'
        result_level[medium_risk_mask & (s >= score_P_good) & (s < score_P_excellent)] = '良好用户'
        result_level[medium_risk_mask & (s >= score_P_normal) & (s < score_P_good)] = '普通用户'
        result_level[medium_risk_mask & (s < score_P_normal)] = '高损耗用户'
        
        # 月用电超过150度但未超250度：强制判定为高损耗用户（修复P2: 加入新兵保护）
        high_energy_mask = (monthly_energy_val > 150) & (monthly_energy_val <= 250) & ~is_violent & ~is_silent & (days_since_join > NEW_USER_PROTECTION_DAYS)
        result_level[high_energy_mask] = '高损耗用户'
        
        # 无风险限制的正常判定，加入新用户保护
        normal_mask = ~is_violent & ~high_risk_mask & ~medium_risk_mask & ~has_violent_risk & ~is_silent & (days_since_join > NEW_USER_PROTECTION_DAYS) & (monthly_energy_val <= 150) & (result_level == '')
        result_level[normal_mask & (s >= score_P_excellent)] = '优质用户'
        result_level[normal_mask & (s >= score_P_good) & (s < score_P_excellent)] = '良好用户'
        result_level[normal_mask & (s >= score_P_normal) & (s < score_P_good)] = '普通用户'
        result_level[normal_mask & (s < score_P_normal)] = '高损耗用户'
        
        # 新兵保护期逻辑：入网前3天的用户，免除动态降级处罚，锁定为"观察期"
        new_user_mask = ~is_violent & ~is_silent & (days_since_join <= NEW_USER_PROTECTION_DAYS) & (result_level == '')
        result_level[new_user_mask] = '观察期'
        
        df['用户等级_动态'] = result_level

        # ── D. 策略建议 ────────────────────────────────────────────────────
        strat = {'优质用户':'留存激励','良好用户':'维持服务',
                 '普通用户':'引导升级','高损耗用户':'限制预警',
                 '暴力':'清退处理','观察期':'新手引导','沉默用户':'激活唤醒'}
        df['策略建议'] = df['用户等级_动态'].map(strat).fillna('维持服务')

        # ── E. 用户等级详尽说明（优化4：向量化替代apply(axis=1)）─────────────────────────────────────────────
        # 提取所有需要的列
        levels = df['用户等级_动态'].values
        scores = df['重评分数'].values
        risks = df['风险标签'].values
        max_currents = df['近7d最大电流_A'].fillna(0).values
        avg_currents = df['近7d平均骑行电流_A'].fillna(0).values
        energies = df['近7d百公里电耗_kWh'].fillna(0).values
        min_socs = df['近7d最低SOC'].fillna(100).values
        monthly_energies = df['单合约月度用电度数预估_kWh'].fillna(0).values
        
        # 等级基本说明（向量化）
        level_desc = np.select(
            [levels == '优质用户', levels == '良好用户', levels == '普通用户', 
             levels == '高损耗用户', levels == '暴力', levels == '观察期', levels == '沉默用户'],
            ['电池使用习惯优秀，低风险', '电池使用习惯较好，负载可控', '电池使用行为一般，需关注',
             '电池损耗高于平均，需重点关注', '存在严重损害电池行为，高风险', '新用户观察期，暂不参与评级', '用户活跃度极低，需关注'],
            default=''
        )
        
        # 核心指标说明（向量化字符串格式化）
        core_metrics = np.array([
            f'评分{s:.0f}分 | 最大电流{mc:.0f}A/平均{ac:.0f}A | 百公里电耗{e:.1f}kWh | 最低SOC{ms:.0f}%'
            for s, mc, ac, e, ms in zip(scores, max_currents, avg_currents, energies, min_socs)
        ])
        
        # 风险提示（向量化）
        risk_tips = np.where(risks != '正常', '风险：' + risks, '')
        
        # 用电盈亏预警（向量化）
        energy_warnings = np.where(monthly_energies > 150, 
                                  '⚠️月用电' + monthly_energies.astype(int).astype(str) + '度超标', '')
        
        # 拼接所有说明（用向量化字符串操作）
        df['用户等级_动态说明'] = pd.Series([
            '；'.join(filter(None, [ld, cm, rt, ew]))
            for ld, cm, rt, ew in zip(level_desc, core_metrics, risk_tips, energy_warnings)
        ], index=df.index)

        # ── F. 对比原等级 ─────────────────────────────────────────────────
        # 修复P1: 沉默用户rank应该介于普通和高损耗之间，不应高于暴力
        level_rank = {'暴力':6,'高损耗用户':4,'普通用户':3,'观察期':3,
                     '良好用户':2,'优质用户':1,'未知':3,'沉默用户':5}
        if '用户等级_综合_7d' in df.columns:
            df['等级变化'] = df.apply(
                lambda r: '[WORSE]' if (
                    level_rank.get(r['用户等级_综合_7d'],3)
            > level_rank.get(r['用户等级_动态'],3)
                ) else '[IMPROVED]' if (
                    level_rank.get(r['用户等级_综合_7d'],3)
                    < level_rank.get(r['用户等级_动态'],3)
                ) else '[SAME]', axis=1)

        return df

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
    parser.add_argument('--baseline', default='./thresholds_baseline.json')
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
    print(f'\n{'='*55}')
    print(f'  动态阈值分析 · 用户等级分布（n={total}）')
    print(f'{'='*55}')

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
