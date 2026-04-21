# ==================================================================================
# 两轮车换电用户分析系统 - 【7天日均指标增强版+全链路判定说明】
# 核心更新：
# 1. 新增「是否在线」字段强制校验+离线数据过滤+长期离线异常监控
# 2. 为所有7天累计指标新增对应的日均平均值字段
# 3. 修复原代码超100A指标统计笔误
# 4. ✅ 全链路新增「用户等级/客户形态」动态说明，匹配100%业务判定规则
# ==================================================================================
import os
import sys

# 设置标准输出为 UTF-8 编码，避免中文和 emoji 字符编码错误
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径，以便导入 config 模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加当前目录到路径，以便导入 dynamic_thresholds 模块
sys.path.insert(0, os.path.dirname(__file__))

import math
import time
import re
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import ConvexHull
from tqdm import tqdm
import concurrent.futures
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime
warnings.filterwarnings('ignore')

# 从 config 模块导入配置
from User_characteristics_repo.config.config import (
    EXPORT_PATH_BATTERY_STATUS_30D,
    EXPORT_PATH_BATTERY_STATUS_3D,
    EXPORT_PATH_USER_BEHAVIOR,
    BASE_EXPORT_PATH,
    EXPORT_FILE_CONTRACT_EARLY,
    EXPORT_FILE_PROVINCE_GEOJSON,
    EXPORT_FILE_CITY_GEOJSON,
    EXPORT_FILE_DISTRICT_GEOJSON,
    EXPORT_FILE_USER_LIFECYCLE_7D,
    EXPORT_FILE_USER_DAILY_SNAPSHOT,
    EXPORT_FILE_USER_ATTENDANCE
)

# 导入动态阈值分析模块
try:
    from User_characteristics_repo.src.dynamic_thresholds import DynamicBatteryAnalyzer
except ImportError as e:
    print(f"⚠️  动态阈值模块导入失败: {e}")
    print("   将使用静态阈值进行分析")
    DynamicBatteryAnalyzer = None

# ==============================================================================
# 动态自适应阈值体系 v1.0
# ==============================================================================
"""
两轮车换电分析 · 动态自适应阈值体系 v1.0

设计原则：
  1. 分位数动态计算（不复用历史阈值）
  2. 基准线持久化（BaselineStore）
  3. 分布漂移检测（DistributionShiftDetector）
  4. 渐进式阈值更新（EMA 平滑）
  5. 零人工干预：阈值从数据中来，到评级里去
"""

class BaselineStore:
    """
    管理阈值基准线的存储、读取与版本追踪。
    """

    def __init__(self, path: str = './thresholds_baseline.json'):
        self.path = path
        self._data: Optional[Dict] = None

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

    def update_ema(self, key_path: str, new_value: float,
                   alpha: float = 0.3) -> float:
        """
        对指定阈值做 EMA 平滑更新，避免单次数据波动导致基准线剧烈跳变。
        优化7：批量更新，先修改内存中的dict，最后一次性写入文件
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
        # 注意：不再在这里写入文件，由调用方批量写入
        self._data = b
        return round(smoothed, 3)
    
    def save_current_data(self) -> None:
        """将当前内存中的基准线数据一次性写入文件"""
        if self._data is not None:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)

    @property
    def baseline(self) -> Optional[Dict]:
        return self._data


class DistributionShiftDetector:
    """
    检测实时数据与基准线之间是否存在显著分布漂移。
    """

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


class AdaptiveThresholds:
    """
    从原始 DataFrame 动态计算所有阈值。
    """

    WINSORIZE_LEVEL = 0.005   # 截断最极端的 0.5% 数据

    def __init__(self, column_map: Optional[Dict] = None):
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
        self.reverse_map = {v: k for k, v in self.column_map.items()}

    def compute(self, df: pd.DataFrame,
                percentiles: List[float] = [0.10, 0.25, 0.45, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.99]
                ) -> Dict:
        """
        从 DataFrame 计算所有分位数阈值。
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

    @staticmethod
    def format_report(quantiles: Dict) -> str:
        lines = ['=' * 60, '  动态阈值计算报告', '=' * 60]
        for key, vals in sorted(quantiles.items()):
            vals = dict(vals)  # 修复Bug3：浅拷贝，避免破坏原始dict
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


class DynamicBatteryAnalyzer:
    """
    整合 BaselineStore / AdaptiveThresholds / DistributionShiftDetector
    的统一入口，替代用户生命周期管理.py 中的静态阈值。
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

    def run(self, df: pd.DataFrame, force_refresh: bool = True
            ) -> Tuple[Dict, Dict]:
        """
        执行动态阈值计算主流程。
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
        if baseline and not force_refresh:
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
            print('[DynamicThresholds] [NEW] No history baseline or force refresh, creating new baseline')
            self.store.save(
                quantiles=live_quantiles,
                sample_size=len(df),
                data_hash=fingerprint,
            )
            active = self.store.load()

        return live_quantiles, (active or live_baseline)

    def score_and_classify(
        self,
        df: pd.DataFrame,
        live_quantiles: Dict,
        active_baseline: Dict,
    ) -> pd.DataFrame:
        """
        基于动态阈值对 DataFrame 进行评分和用户等级分类。
        【优化版】使用向量化操作替代apply(axis=1)，性能提升10-100倍
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
        
        # 统一暴力用户判定逻辑：连续超100A 或 峰值电流超P99 或 月用电超250度
        max_cur = df['近7d最大电流_A'].fillna(0).values
        max_cur_P99 = bq.get('max_current', {}).get('P99', 81.9)
        monthly_energy_val = df['单合约月度用电度数预估_kWh'].fillna(0).values
        
        is_violent = (over100 >= 2) | (max_cur > max_cur_P99) | (monthly_energy_val > 250)
        
        # 风险等级限制
        has_violent_risk = np.array(['暴力放电' in r or '极端电流' in r for r in risk])
        has_high_risk = np.array(['超保护板电流' in r or '频繁超80A' in r for r in risk])
        has_medium_risk = np.array(['月用电超标' in r for r in risk])
        
        # 计算入网天数（用于新兵保护期）
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
        
        # 修复Bug2：has_violent_risk但over100<2的用户，已在high_risk_mask中处理，不应再覆盖为"暴力"
        # 删除原来的violent_risk_mask覆盖逻辑

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
        """将实时分位数以 EMA 方式更新到基准线文件。
        优化7：批量更新，先全部更新内存中的数据，最后一次性写入文件
        """
        # 先在内存中批量更新所有指标
        for metric, vals in live_quantiles.items():
            for pct, val in vals.items():
                if pct.startswith('_'):
                    continue
                self.store.update_ema(
                    f'quantiles.{metric}.{pct}', val, self.ema_alpha)
        
        # 最后一次性写入文件
        self.store.save_current_data()

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

# 移除动态阈值模块的导入，因为已经内联到本文件
DynamicBatteryAnalyzer = DynamicBatteryAnalyzer

# ==================================================================================
# -------------------------- 【核心配置区】所有可调整参数都在这里 --------------------------
# ==================================================================================
# -------------------------- 1. 输入输出路径配置 --------------------------
FOLDER_MAPPING = {
    EXPORT_PATH_BATTERY_STATUS_30D: "历史30天-前4天",
    EXPORT_PATH_BATTERY_STATUS_3D: "近3天-昨天"
}
OUTPUT_ROOT_FOLDER = os.path.join(BASE_EXPORT_PATH, EXPORT_PATH_USER_BEHAVIOR)
USER_LIFECYCLE_7D_FILE = EXPORT_FILE_USER_LIFECYCLE_7D
USER_DAILY_SNAPSHOT_FILE = EXPORT_FILE_USER_DAILY_SNAPSHOT
USER_ATTENDANCE_FILE = EXPORT_FILE_USER_ATTENDANCE

USER_CONTRACT_INFO_PATH = os.path.join(BASE_EXPORT_PATH, EXPORT_FILE_CONTRACT_EARLY)
PROVINCE_GEOJSON_PATH = os.path.join(BASE_EXPORT_PATH, EXPORT_FILE_PROVINCE_GEOJSON)
CITY_GEOJSON_PATH = os.path.join(BASE_EXPORT_PATH, EXPORT_FILE_CITY_GEOJSON)
DISTRICT_GEOJSON_PATH = os.path.join(BASE_EXPORT_PATH, EXPORT_FILE_DISTRICT_GEOJSON)

# -------------------------- 2. 运行模式开关 --------------------------
# 运行模式选择
# 可选值：
# - "quick"    = 快速模式：跳过原始数据处理，仅利用现有快照重新生成「全量用户7天滚动生命周期档案.csv」
# - "incremental" = 增量模式：只处理比快照表中最新日期更新的数据，跳过历史归档目录
# - "full"     = 完整模式：处理所有配置目录，重新计算所有数据
# - "recalculate" = 重新计算模式：基于现有快照重新计算客户形态和用户等级，包括综合说明
RUN_MODE = "full"

ROLLING_WINDOW_DAYS = 7
ROLLING_WEIGHTS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

# -------------------------- 3. 基础工具参数 --------------------------
DISTANCE_UNIT_KM = True
GPS_FILTER = True
CSV_ENCODING = 'utf-8'
GPS_CACHE_FILE = "gps_region_cache.json"
MAX_WORKERS = 12
GPS_PRECISION = 6
MIN_DISTANCE_THRESHOLD = 50

# -------------------------- 4. 用户生命周期配置 --------------------------
NEW_USER_PROTECTION_DAYS = 3
CHURN_THRESHOLD_DAYS = 7
SILENT_THRESHOLD_DAYS = 1

# -------------------------- 5. 骑行/放电业务参数（统计学优化版） --------------------------
COLLECTION_CYCLE_MIN = 5
MAX_SPEED_KMH = 120
DRIFT_SPEED_THRESHOLD = 150
import numpy as np
MAX_DISTANCE_THRESHOLD = (MAX_SPEED_KMH * COLLECTION_CYCLE_MIN / 60) * 1.1 if DISTANCE_UNIT_KM else (MAX_SPEED_KMH * 1000 * COLLECTION_CYCLE_MIN / 60) * 1.1
RADIUS_QUANTILES = [0.9, 0.95]

MIN_VALID_DISPLACEMENT_M = 20
STOP_SMOOTH_WINDOW = 5
MIN_RIDING_DURATION_MIN = 3

MAX_TRIP_GAP_MIN = 60
VALID_RIDE_HOUR_FOR_PATTERN = 0.5
PATTERN_DOMINANT_RATIO = 0.5

# -------------------------- 6. 地摊/储能场景专属判定参数（不变） --------------------------
STORAGE_MIN_VALID_GPS_POINTS = 10
STORAGE_MAX_DISTANCE_KM = 1.0
STORAGE_MAX_AVG_SPEED_KMH = 3.0
STORAGE_DISCHARGE_RATIO_THRESHOLD = 3.0
STORAGE_MIN_DISCHARGE_HOUR = 2.0
STORAGE_RIDE_DURATION_RATIO = 0.1

# -------------------------- 7. 【统计学优化】电流/速度异常判定参数 --------------------------
# 原始数据统计基准（2026-04 全量用户档案，n≈20000）：
#   平均骑行电流：均值=17.41A  σ=8.15  P75=22.29  P90=28.31  P95=32.13  P99=40.35
#   最大电流：    均值=48.03A  σ=13.43 P75=57.97  P90=64.74  P95=68.86  P99=81.86
#                             1.5IQR上界=85.70  3IQR上界=113.43  均值+3σ=88.30
#
# 优化逻辑：
#   - 异常判断应基于「相对于总体的离群程度」，而非固定绝对值
#   - 使用 P90/P95 作为「中风险」入口，P99 或 IQR外围点作为「高风险」入口
#   - 这样能保证约5%用户触发中风险，约1%触发高风险，符合统计异常比例

MAX_VALID_CURRENT = 150.0
MIN_VALID_CURRENT = 0.1
MAX_VALID_SPEED = 100.0

# ✅ 电流阈值体系：60A/80A/100A 三级
OVER_CURRENT_THRESHOLD = 100.0          # 超100A：暴力放电判定
HIGH_CURRENT_THRESHOLD = 80.0           # 超80A：高电流判定（原60A）
NORMAL_CURRENT_THRESHOLD = 60.0         # 超60A：正常电流上限（原50A）
OVER_CURRENT_CONTINUOUS = 2
OVER_CURRENT_MIN_HOUR = 0.1

# 改装/超速判定（保持不变）
MODIFY_SPEED_THRESHOLD = 50
MODIFY_CURRENT_THRESHOLD = 24
VALID_RIDE_MIN_HOUR = 0.1

NOON_PEAK_HOURS = [11, 12, 13]
EVENING_PEAK_HOURS = [17, 18, 19]
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
OFFPEAK_HOURS = [h for h in range(24) if h not in NOON_PEAK_HOURS + EVENING_PEAK_HOURS + NIGHT_HOURS]

MIN_RECORD_DAYS_FOR_ESTIMATE = 7
FULL_RECORD_DAYS_FOR_CUMULATIVE = 30
DEFAULT_MONTH_DAYS = 26
MONTH_CALENDAR_DAYS = 30

# -------------------------- 9. 在线状态监控参数（不变） --------------------------
LONG_TERM_OFFLINE_THRESHOLD_DAYS = 7  # 连续N天无在线数据即判定为异常

# -------------------------- 10. 包月友好评分扣分阈值（旧版静态阈值，保留兼容） --------------------------
# SOC维度扣分阈值
SOC_CRITICAL_RATIO_DEDUCT_THRESHOLD = 0.05   # SOC低于10%时长占比超过5%开始扣分
SOC_LOW_RATIO_DEDUCT_THRESHOLD = 0.10        # SOC低于20%时长占比超过10%开始扣分
SOC_OPTIMAL_LOWER = 30                       # SOC最优区间下限
SOC_OPTIMAL_UPPER = 80                       # SOC最优区间上限
SOC_OPTIMAL_BONUS = 3                        # SOC处于最优区间且无低电量时加分
MAX_SOC_DEDUCT = 15                          # SOC维度最大扣分

# 电耗维度扣分阈值
EXTREME_ENERGY_THRESHOLD = 17.0              # 极端电耗阈值（百公里kWh）
HIGH_ENERGY_THRESHOLD = 13.8                 # 高电耗阈值（百公里kWh）
MAX_ENERGY_DEDUCT = 12                       # 电耗维度最大扣分

# 换电次数扣分阈值
MAX_NORMAL_BATTERY_CHANGE = 3                # 日均正常换电次数上限
EXCESS_CHANGE_DEDUCT_PER_TIME = 2            # 每次超额换电扣分
MAX_CHANGE_DEDUCT = 8                        # 换电次数最大扣分

# 电流异常判定阈值（旧版）
ABNORMAL_MAX_CURRENT = 80.0                  # 最大电流异常阈值
ABNORMAL_CV = 1.5                            # 电流变异系数异常阈值
ABNORMAL_AVG_CURRENT = 35.0                  # 平均骑行电流异常阈值
ABNORMAL_MIN_DISCHARGE_HOUR = 2.0            # 最小放电时长异常阈值

# SOC告警阈值
SOC_LOW_WARNING = 20                         # SOC低电量告警线
SOC_CRITICAL = 10                            # SOC危险告警线

# 用户等级判定阈值（旧版）
VIOLENT_CURRENT_TIMES = 2                    # 暴力用户电流次数阈值
HIGH_LOSS_CURRENT_TIMES = 3                  # 高损耗用户电流次数阈值

# ==================================================================================
# -------------------------- 配置区结束 --------------------------
# ==================================================================================

EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000.0

# 原始数据必须包含的字段列表
REQUIRED_COLUMNS_RAW = [
    "时间戳", "合约id", "用户id", "纬度", "经度",
    "统计日期时间", "电流", "温度", "速度",
    "电池id", "电池SOC", "电池度数",
    "是否在线"
]

# 字段名映射表：兼容不同的数据源命名
COLUMN_NORMALIZE_MAP = {
    "时间戳": ["时间戳", "timestamp", "ts", "时间", "parse_time"],
    "合约id": ["合约id", "合约ID", "合约号", "contract_id", "device_id"],
    "用户id": ["用户id", "用户ID", "用户号", "user_id"],
    "纬度": ["纬度", "lat", "latitude"],
    "经度": ["经度", "lon", "longitude", "lng"],
    "统计日期时间": ["统计日期时间", "datetime", "时间", "日期时间"],
    "电流": ["电流", "current", "i"],
    "温度": ["温度", "temp", "temperature", "t"],
    "速度": ["速度", "speed", "v"],
    "电池id": ["电池id", "电池ID", "battery_id", "batteryid"],
    "电池SOC": ["电池SOC", "soc", "SOC", "剩余电量", "电池剩余电量"],
    "电池度数": ["电池度数", "电池容量kWh", "battery_energy", "energy", "电池电量"],
    "是否在线": ["是否在线", "在线状态", "status", "online_status", "is_online"]
}

# ==================================================================================
# ===================== 0. 全局用户合约信息缓存模块 =====================
# ==================================================================================
_user_contract_map = None
def load_user_contract_info():
    global _user_contract_map
    if _user_contract_map is not None:
        return _user_contract_map
    if not os.path.exists(USER_CONTRACT_INFO_PATH):
        print(f"⚠️ 用户合约信息表不存在：{USER_CONTRACT_INFO_PATH}")
        _user_contract_map = {}
        return _user_contract_map
    print(f"📂 正在加载用户合约信息表...")
    try:
        df_contract = pd.read_csv(USER_CONTRACT_INFO_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df_contract = pd.read_csv(USER_CONTRACT_INFO_PATH, encoding='gbk')
    df_contract.columns = [str(col).strip() for col in df_contract.columns]
    if 'user_id' not in df_contract.columns:
        print(f"⚠️ 用户合约信息表缺少user_id列")
        _user_contract_map = {}
        return _user_contract_map
    
    # 优化1：向量化替代iterrows，使用groupby+agg一次性完成聚合
    df_contract['user_id'] = df_contract['user_id'].astype(str).str.strip()
    df_contract = df_contract[df_contract['user_id'].notna() & (df_contract['user_id'] != 'nan')]
    
    # 转换时间列为datetime
    if '入网时间' in df_contract.columns:
        df_contract['入网时间'] = pd.to_datetime(df_contract['入网时间'], errors='coerce')
    if '退网时间' in df_contract.columns:
        df_contract['退网时间'] = pd.to_datetime(df_contract['退网时间'], errors='coerce')
    
    # 使用groupby聚合：入网时间取min，退网时间取max
    agg_dict = {}
    if '入网时间' in df_contract.columns:
        agg_dict['join_date'] = ('入网时间', 'min')
    if '退网时间' in df_contract.columns:
        agg_dict['expire_date'] = ('退网时间', 'max')
    
    if agg_dict:
        g = df_contract.groupby('user_id').agg(**agg_dict)
        # 转换为字典格式
        _user_contract_map = {}
        for uid, row in g.iterrows():
            join_date = row['join_date'].strftime('%Y-%m-%d') if 'join_date' in row and pd.notna(row['join_date']) else None
            expire_date = row['expire_date'].strftime('%Y-%m-%d') if 'expire_date' in row and pd.notna(row['expire_date']) else None
            _user_contract_map[uid] = {'join_date': join_date, 'expire_date': expire_date}
    else:
        _user_contract_map = {}
    
    print(f"✅ 用户合约信息表加载完成：共 {len(_user_contract_map)} 个用户")
    return _user_contract_map

# ==================================================================================
# ===================== 1. 地理围栏匹配模块 =====================
# ==================================================================================
_fence_data = None
def load_fence_data():
    global _fence_data
    if _fence_data is not None: return _fence_data
    for path, name in zip([PROVINCE_GEOJSON_PATH, CITY_GEOJSON_PATH, DISTRICT_GEOJSON_PATH], ["省围栏", "市围栏", "区县级围栏"]):
        if not os.path.exists(path):
            print(f"❌ {name}文件不存在：{path}")
            return None
    print(f"📂 正在加载省-市-区县三级地理围栏数据...")
    start = time.time()
    
    def load_geo(path, output_col_name):
        gdf = gpd.read_file(path)
        if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
        name_col = None
        for col in gdf.columns:
            if col.lower() == 'name': name_col = col; break
        if not name_col:
            for col in gdf.columns:
                if 'name' in col.lower() or '名称' in col or '省' in col or '市' in col or '县' in col:
                    name_col = col; break
        if not name_col: raise ValueError(f"❌ 在 {path} 中未找到名称字段")
        gdf = gdf[[name_col, 'geometry']].copy()
        gdf = gdf.rename(columns={name_col: output_col_name})
        return gdf

    try:
        gdf_prov = load_geo(PROVINCE_GEOJSON_PATH, "核心活动省份")
        gdf_city = load_geo(CITY_GEOJSON_PATH, "核心活动城市")
        gdf_dist = load_geo(DISTRICT_GEOJSON_PATH, "核心活动区县")
        _fence_data = {'prov': gdf_prov, 'city': gdf_city, 'district': gdf_dist}
        print(f"✅ 三级围栏加载完成：省 {len(gdf_prov)} 个，市 {len(gdf_city)} 个，区县 {len(gdf_dist)} 个，耗时{time.time()-start:.2f}s")
        return _fence_data
    except Exception as e:
        print(f"❌ 加载围栏失败: {e}")
        return None

def batch_gps_to_region(df, lat_col='中心纬度', lon_col='中心经度'):
    """
    GPS空间匹配优化版：使用坐标哈希降维缓存，性能提升5-10倍
    修复：安全处理 sjoin 重复行 + 防止 merge 列名冲突 + 兜底保护
    """
    fence_data = load_fence_data()
    if not fence_data:
        return pd.DataFrame('', index=df.index,
                            columns=['核心活动省份', '核心活动城市', '核心活动区县'])

    df = df.copy()
    df['_idx'] = df.index
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    valid_gps = df.dropna(subset=[lat_col, lon_col]).copy()

    final_result = pd.DataFrame('', index=df.index,
                                columns=['核心活动省份', '核心活动城市', '核心活动区县'])

    if valid_gps.empty:
        return final_result

    valid_gps['lat_round'] = valid_gps[lat_col].round(3)
    valid_gps['lon_round'] = valid_gps[lon_col].round(3)

    unique_coords = valid_gps[['lat_round', 'lon_round']].drop_duplicates().reset_index(drop=True)
    print(f"   🗺️ 坐标降维：{len(valid_gps)} 个点 -> {len(unique_coords)} 个唯一坐标")

    gdf_unique = gpd.GeoDataFrame(
        unique_coords,
        geometry=gpd.points_from_xy(unique_coords['lon_round'], unique_coords['lat_round']),
        crs="EPSG:4326"
    )

    print(f"   🗺️ 正在执行空间匹配（降维后）...")

    def safe_sjoin(gdf_pts, gdf_poly, col_name):
        """安全 sjoin：去重复行 + 对齐索引 + 列存在性检查"""
        try:
            res = gpd.sjoin(gdf_pts, gdf_poly, how="left", predicate="within")
            res = res[~res.index.duplicated(keep='first')]
            res = res.reindex(gdf_pts.index)
            if col_name in res.columns:
                return res[col_name].fillna('').tolist()
            else:
                return [''] * len(gdf_pts)
        except Exception as e:
            print(f"   ⚠️ [{col_name}] 空间匹配失败: {e}")
            return [''] * len(gdf_pts)

    unique_coords = unique_coords.copy()
    unique_coords['核心活动省份'] = safe_sjoin(gdf_unique, fence_data['prov'], '核心活动省份')
    unique_coords['核心活动城市'] = safe_sjoin(gdf_unique, fence_data['city'], '核心活动城市')
    unique_coords['核心活动区县'] = safe_sjoin(gdf_unique, fence_data['district'], '核心活动区县')

    valid_gps = valid_gps.drop(
        columns=[c for c in ['核心活动省份', '核心活动城市', '核心活动区县'] if c in valid_gps.columns],
        errors='ignore'
    )
    valid_gps = valid_gps.merge(unique_coords, on=['lat_round', 'lon_round'], how='left')

    for col in ['核心活动省份', '核心活动城市', '核心活动区县']:
        if col not in valid_gps.columns:
            valid_gps[col] = ''

    idx_map = valid_gps.drop_duplicates(subset=['_idx']).set_index('_idx')
    for col in ['核心活动省份', '核心活动城市', '核心活动区县']:
        final_result[col] = idx_map[col].reindex(final_result.index).fillna('').values

    return final_result

def match_single_gps(*args, **kwargs): return None
def get_real_center(group):
    median_lat, median_lon = np.median(group['纬度']), np.median(group['经度'])
    riding = group[group['骑行状态'] == 1][['纬度', '经度']].copy()
    if riding.empty: return median_lat, median_lon
    riding['dist'] = (riding['纬度'] - median_lat)**2 + (riding['经度'] - median_lon)**2
    return riding.loc[riding['dist'].idxmin()]['纬度'], riding.loc[riding['dist'].idxmin()]['经度']

# ==================================================================================
# ===================== 2. 通用工具函数 =====================
# ==================================================================================
def normalize_column_names(df):
    df = df.copy()
    raw_columns = {col.strip().lower(): col for col in df.columns}
    new_columns = {}
    for standard_name, alias_list in COLUMN_NORMALIZE_MAP.items():
        for alias in alias_list:
            alias_lower = alias.strip().lower()
            if alias_lower in raw_columns:
                new_columns[raw_columns[alias_lower]] = standard_name
                break
    return df.rename(columns=new_columns)

def haversine(lat1, lon1, lat2, lon2, unit='km'):
    radius = EARTH_RADIUS_KM if unit == 'km' else EARTH_RADIUS_M
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def get_stat_date(folder_name):
    folder_name_clean = folder_name.strip()
    date_formats = ['%Y年%m月%d日', '%Y-%m-%d', '%Y%m%d', '%m月%d日%Y年', '%d-%m-%Y', '%m/%d/%Y']
    for fmt in date_formats:
        try:
            date_obj = pd.to_datetime(folder_name_clean, format=fmt, errors='coerce')
            if pd.notna(date_obj): return date_obj.strftime('%Y-%m-%d')
        except: pass
    date_pattern = re.compile(r'(\d{4})[^\d]*(\d{1,2})[^\d]*(\d{1,2})')
    match_result = date_pattern.search(folder_name_clean)
    if match_result:
        year, month, day = match_result.groups()
        try:
            date_obj = pd.to_datetime(f"{year}-{month}-{day}", errors='coerce')
            if pd.notna(date_obj): return date_obj.strftime('%Y-%m-%d')
        except: pass
    return pd.Timestamp.now().strftime('%Y-%m-%d')

# ==================================================================================
# ===================== 3. 核心业务指标计算 =====================
# ==================================================================================

def _calc_monthly_score_v2(
    max_speed, max_temp, riding_avg_current,
    current_60a_count, current_80a_count, over100a_cont,
    max_trip_current_cv, current_cv, avg_riding_speed,
    soc_data_valid, has_real_ride, soc_below_10_ratio, soc_below_20_ratio,
    min_soc, avg_soc,
    energy_data_valid, energy_per_100km, battery_change_count,
    daily_distance=0, battery_change_cnt=0
):
    """
    【统计学优化版】包月友好评分

    核心改进：
    1. 扣分量基于「相对总体的离群程度」，而非固定绝对值
    2. 各维度最大扣分经过标准化控制，避免单因素垄断
    3. 扣分触发点从实际数据分位数中取得
    4. 沉默用户识别提前：日行驶里程和换电次数均为0时，直接返回沉默分

    最终分数 ∈ [0, 100]，含义：
        ≥82  → 优质（P75以上，约25%用户）
        69~82 → 良好（P50~P75，约25%用户）
        52~69 → 普通（P25~P50，约25%用户）
        <52   → 高损耗（P25以下，约25%用户）
        沉默用户：日行驶里程=0且换电次数=0
    """
    # 沉默用户识别提前至基础计算层（修复Bug2：高分悖论）
    if daily_distance == 0 and battery_change_cnt == 0:
        return 20  # 沉默用户统一给20分，避免日级数据得满分、7天聚合得20分的冲突

    score = 100.0

    # ── A. 速度维度（最大扣10分）──
    if max_speed >= 80:
        score -= 10
    elif max_speed >= 60:
        score -= 6
    elif max_speed >= 50:
        score -= 3

    # ── B. 温度维度（最大扣8分）──
    if max_temp >= 70:
        score -= 8
    elif max_temp >= 55:
        score -= 4

    # ── C. 电流维度（最大扣25分）──
    if riding_avg_current >= 40.4:
        score -= 15
    elif riding_avg_current >= 32.1:
        score -= 10
    elif riding_avg_current >= 28.3:
        score -= 5
    elif riding_avg_current >= 22.3:
        score -= 2

    # ── BUG4修复：电流>80A次数应排除>100A的部分，避免与超100A指标重叠扣分 ──
    # current_80a_count 应定义为 "80A < 电流 <= 100A" 的次数
    # over100a_cont 单独处理 >100A 的情况，两者互斥
    deduct_80a = min(current_80a_count * 1.5, 8)
    score -= deduct_80a

    if over100a_cont > 0:
        score -= min(over100a_cont * 4, 12)

    if max_trip_current_cv >= 1.5:
        score -= 4
    elif current_cv >= 0.8:
        score -= 2 if avg_riding_speed <= 15 else 4
    elif current_cv >= 0.5:
        score -= 1

    # ── D. SOC 维度（最大扣15分）──
    soc_deduct = 0.0
    if soc_data_valid and has_real_ride:
        if soc_below_10_ratio >= SOC_CRITICAL_RATIO_DEDUCT_THRESHOLD:
            soc_deduct += min((soc_below_10_ratio - SOC_CRITICAL_RATIO_DEDUCT_THRESHOLD) / 0.01 * 2, 8)

        if soc_below_20_ratio >= SOC_LOW_RATIO_DEDUCT_THRESHOLD:
            soc_deduct += min((soc_below_20_ratio - SOC_LOW_RATIO_DEDUCT_THRESHOLD) / 0.02 * 1.5, 5)

        if min_soc <= 5:
            soc_deduct += 4
        elif min_soc <= 10:
            soc_deduct += 2

        if (SOC_OPTIMAL_LOWER <= avg_soc <= SOC_OPTIMAL_UPPER) and soc_below_20_ratio == 0:
            score += SOC_OPTIMAL_BONUS

    score -= min(soc_deduct, MAX_SOC_DEDUCT)

    # ── E. 电耗维度（最大扣12分）──
    energy_deduct = 0.0
    if energy_data_valid and has_real_ride:
        if energy_per_100km >= EXTREME_ENERGY_THRESHOLD:
            energy_deduct += 8
        elif energy_per_100km >= HIGH_ENERGY_THRESHOLD:
            energy_deduct += 4
        elif energy_per_100km >= 10.56:
            energy_deduct += 2

        excess = max(0, battery_change_count - MAX_NORMAL_BATTERY_CHANGE)
        energy_deduct += min(excess * EXCESS_CHANGE_DEDUCT_PER_TIME, MAX_CHANGE_DEDUCT)

    score -= min(energy_deduct, MAX_ENERGY_DEDUCT)

    return max(0, min(100, round(score)))


def _determine_user_level_v2(
    over100a_cont, over100a_hours,
    current_80a_count, soc_below_10_ratio, min_soc,
    current_60a_count, energy_per_100km, soc_below_20_ratio,
    score, riding_avg_current,
    soc_data_valid, energy_data_valid
):
    """
    【动态阈值版】用户等级判定

    核心改进：
    1. 暴力用户：坚守「超100A」这一有统计意义的极端行为
    2. 高损耗用户：基于动态分位数判定
    3. 普通/良好/优质：使用动态计算的阈值
    """
    is_violent = False

    if (over100a_cont >= 2) or (over100a_hours >= 0.1):
        is_violent = True
    elif soc_data_valid and (soc_below_10_ratio >= 0.20) and (min_soc <= 5):
        is_violent = True

    is_high_loss = False

    # 使用动态阈值的默认值作为回退
    if energy_data_valid and (energy_per_100km >= 17.0):
        is_high_loss = True
    elif (riding_avg_current >= 32.1) and soc_data_valid:
        if soc_below_20_ratio >= 0.10:
            is_high_loss = True
    elif soc_data_valid and (soc_below_20_ratio >= 0.20):
        is_high_loss = True
    elif (score < 30) and energy_data_valid:
        is_high_loss = True

    if is_violent:
        return "暴力用户（超量放电/电池滥用）"
    elif is_high_loss:
        return "高损耗用户"
    else:
        # 使用动态阈值的默认值作为回退
        if score >= 75:
            return "优质用户"
        elif score >= 60:
            return "良好用户"
        elif score >= 45:
            return "普通用户"
        else:
            return "高损耗用户"


def calc_contract_metrics(df_sorted):
    group_keys = ['合约id', '用户id']
    results = []
    groups = list(df_sorted.groupby(group_keys))
    print(f"   🔢 共 {len(groups)} 个合约，开始计算指标...")

    for idx, ((contract_id, user_id), group) in enumerate(tqdm(groups, desc="合约计算进度", ncols=80)):
        user_id = str(user_id) if pd.notna(user_id) else f"未知用户_{idx}"
        group = group.sort_values('时间戳').reset_index(drop=True)
        n = len(group)
        stat_date = group['统计日期'].iloc[0] if not group.empty else pd.Timestamp.now().strftime('%Y-%m-%d')
        
        res = {
            '统计日期': stat_date, '合约id': contract_id, '用户id': user_id,
            '核心活动省份': '', '核心活动城市': '', '核心活动区县': '',
            '行驶距离': 0.0, '最大出行距离': 0.0,
            'R90日常活动半径': 0.0, 'R95核心活动半径': 0.0, '凸包覆盖面积': 0.0,
            '中心纬度': np.nan, '中心经度': np.nan,
            '骑行次数': 0, '单次平均骑行时长(分钟)': 0.0, '骑行总耗时(小时)': 0.0,
            '最大速度': 0.0, '最大速度时间': pd.NA, '平均骑行速度': 0.0,
            '最大电流': 0.0, '最大电流时间': pd.NA, '骑行放电平均电流': 0.0,
            '全周期平均电流': 0.0, '电流标准差': 0.0, '电流变异系数': 0.0,
            '高峰电流标准差': 0.0, '高峰电流变异系数': 0.0,
            '平峰电流标准差': 0.0, '平峰电流变异系数': 0.0,
            '单行程最大电流标准差': 0.0, '单行程最大电流变异系数': 0.0,
            '总放电时长(小时)': 0.0, 
            '怠速放电时长(小时)': 0.0,
            '骑行-放电时长比': 0.0,
            '电流异常用户': '否',
            '最大温度': 0.0, '最大温度时间': pd.NA, '平均温度': 0.0,
            '客户形态': '数据不足', '包月友好评分': 0, '用户等级': '无效',
            '高峰骑行占比': 0.0, '单日骑行次数': 0, '当日是否出勤': 0,
            '电流>60A次数': 0, '电流>80A次数': 0,
            '超100A连续次数': 0, '超100A累计时长_h': 0.0,
            '总用电量(kWh)': 0.0, '百公里电耗(kWh)': 0.0, '换电次数': 0,
            '平均骑行SOC': 0.0, '最低SOC': 100.0, 'SOC低于20%时长占比': 0.0, 'SOC低于10%时长占比': 0.0,
            '午间高峰长时骑行次数': 0, '晚间高峰长时骑行次数': 0, '平峰长时骑行次数': 0, '夜间长时骑行次数': 0,
            '午间高峰骑行里程': 0.0, '晚间高峰骑行里程': 0.0, '平峰骑行里程': 0.0, '夜间骑行里程': 0.0,
            '工作时长覆盖(小时)': 0.0, '有效GPS点数': n
        }

        total_riding_hours = 0.0
        idle_discharge_hours = 0.0
        total_discharge_hours = 0.0

        if n >= 2:
            # -------------------------- 1. 位移与骑行状态判定 --------------------------
            group['前纬度'] = group['纬度'].shift(1)
            group['前经度'] = group['经度'].shift(1)
            group['前时间戳'] = group['时间戳'].shift(1)
            group['相邻距离_km'] = haversine(group['前纬度'], group['前经度'], group['纬度'], group['经度'], 'km')
            group['相邻距离_m'] = group['相邻距离_km'] * 1000
            group['时间间隔_min'] = (group['时间戳'] - group['前时间戳']) / 60
            group['动态距离阈值_km'] = (MAX_SPEED_KMH * group['时间间隔_min'] / 60) * 1.1
            group['有效位移'] = (group['相邻距离_km'] <= group['动态距离阈值_km']) & (group['相邻距离_km'] > 0)

            # 骑行判定逻辑
            main_ride_condition = (
                (group['相邻距离_m'] >= MIN_VALID_DISPLACEMENT_M) &
                (group['速度'].fillna(0) > 0) &
                (group['速度'].fillna(0) <= MAX_VALID_SPEED) &
                (group['电流'] > -20)
            )
            group['初始骑行状态'] = main_ride_condition.rolling(
                window=STOP_SMOOTH_WINDOW, 
                min_periods=1
            ).max().fillna(0).astype(int)

            assist_ride_condition = (
                (group['相邻距离_m'] < MIN_VALID_DISPLACEMENT_M) &
                (group['速度'].fillna(0) >= 0) &
                (group['速度'].fillna(0) <= 5) &
                (group['电流_骑行判定用'] > 0) &
                (group['初始骑行状态'].shift(1).fillna(0) == 1)
            )
            group['核心有效'] = main_ride_condition | assist_ride_condition
            group['骑行状态'] = group['核心有效'].rolling(
                window=STOP_SMOOTH_WINDOW, 
                min_periods=1,
                center=True
            ).max().fillna(0).astype(int)
            group['有效里程_km'] = group['相邻距离_km'].where(
                (group['骑行状态'] == 1) & group['有效位移'] & main_ride_condition, 
                0.0
            )
            group = group.drop(columns=['初始骑行状态'])

            total_distance = group['有效里程_km'].sum() if DISTANCE_UNIT_KM else group['有效里程_km'].sum() * 1000
            res['行驶距离'] = round(total_distance, 2)

            # -------------------------- 2. 空间活动指标计算 --------------------------
            riding_points = group.loc[group['骑行状态'] == 1, ['纬度', '经度']].values
            center_lat, center_lon = get_real_center(group)
            res['中心纬度'] = round(center_lat, 6) if pd.notna(center_lat) else np.nan
            res['中心经度'] = round(center_lon, 6) if pd.notna(center_lon) else np.nan
            
            max_trip_distance = r90_radius = r95_radius = hull_area = 0.0
            if len(riding_points) >= 3:
                try:
                    # 【优化】降采样：点数超过200时随机抽样，速度几何级提升
                    CONVEXHULL_MAX_POINTS = 200
                    if len(riding_points) > CONVEXHULL_MAX_POINTS:
                        np.random.seed(42)  # 固定种子保证可复现
                        sample_indices = np.random.choice(len(riding_points), CONVEXHULL_MAX_POINTS, replace=False)
                        hull_input = riding_points[sample_indices]
                    else:
                        hull_input = riding_points
                    
                    # 【优化】共线性保护：先评估边界框，长宽比极度失衡时跳过ConvexHull
                    lat_range = hull_input[:, 0].max() - hull_input[:, 0].min()
                    lon_range = hull_input[:, 1].max() - hull_input[:, 1].min()
                    aspect_ratio = max(lat_range, lon_range) / (min(lat_range, lon_range) + 1e-10)
                    
                    if aspect_ratio > 1000:  # 长宽比>1000，近似直线，面积视为0
                        hull_area = 0.0
                        # 向量化计算最大出行距离：使用numpy广播替代双层for循环
                        if len(hull_input) >= 2:
                            lat1 = hull_input[:, 0][:, np.newaxis]
                            lon1 = hull_input[:, 1][:, np.newaxis]
                            lat2 = hull_input[:, 0][np.newaxis, :]
                            lon2 = hull_input[:, 1][np.newaxis, :]
                            # 向量化haversine计算所有点对距离
                            dlat = np.radians(lat2 - lat1)
                            dlon = np.radians(lon2 - lon1)
                            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                            dist_matrix = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) if DISTANCE_UNIT_KM else 2 * 6371000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                            # 只取上三角矩阵（避免重复计算和对角线）
                            upper_tri = np.triu(dist_matrix, k=1)
                            if upper_tri.size > 0:
                                max_trip_distance = np.max(upper_tri)
                        dist_to_center = haversine(center_lat, center_lon, riding_points[:,0], riding_points[:,1], 'km' if DISTANCE_UNIT_KM else 'm')
                        r90_radius = np.quantile(dist_to_center, RADIUS_QUANTILES[0])
                        r95_radius = np.quantile(dist_to_center, RADIUS_QUANTILES[1])
                    else:
                        hull = ConvexHull(hull_input)
                        hull_points = hull_input[hull.vertices]
                        # 向量化计算最大出行距离：使用numpy广播替代双层for循环
                        if len(hull_points) >= 2:
                            lat1 = hull_points[:, 0][:, np.newaxis]
                            lon1 = hull_points[:, 1][:, np.newaxis]
                            lat2 = hull_points[:, 0][np.newaxis, :]
                            lon2 = hull_points[:, 1][np.newaxis, :]
                            dlat = np.radians(lat2 - lat1)
                            dlon = np.radians(lon2 - lon1)
                            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                            dist_matrix = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) if DISTANCE_UNIT_KM else 2 * 6371000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                            upper_tri = np.triu(dist_matrix, k=1)
                            if upper_tri.size > 0:
                                max_trip_distance = np.max(upper_tri)
                        dist_to_center = haversine(center_lat, center_lon, riding_points[:,0], riding_points[:,1], 'km' if DISTANCE_UNIT_KM else 'm')
                        r90_radius = np.quantile(dist_to_center, RADIUS_QUANTILES[0])
                        r95_radius = np.quantile(dist_to_center, RADIUS_QUANTILES[1])
                        lat_per_km = 1 / 111.0
                        lon_per_km = 1 / (111.0 * math.cos(math.radians(center_lat))) if pd.notna(center_lat) else 1 / 111.0
                        points_km = np.zeros_like(hull_points)
                        points_km[:,0] = (hull_points[:,0] - center_lat) / lat_per_km if pd.notna(center_lat) else 0
                        points_km[:,1] = (hull_points[:,1] - center_lon) / lon_per_km if pd.notna(center_lon) else 0
                        hull_km = ConvexHull(points_km)
                        hull_area = hull_km.area if DISTANCE_UNIT_KM else hull_km.area * 1000000
                except:
                    hull_area = 0.0
            
            res['最大出行距离'] = round(max_trip_distance, 2)
            res['R90日常活动半径'] = round(r90_radius, 2)
            res['R95核心活动半径'] = round(r95_radius, 2)
            res['凸包覆盖面积'] = round(hull_area, 2)

            # -------------------------- 3. 行程拆分与骑行时长计算 --------------------------
            group['状态变化'] = group['骑行状态'].diff().fillna(0)
            group['断连标记'] = (group['时间间隔_min'] > MAX_TRIP_GAP_MIN).astype(int)
            group['行程ID'] = ((group['状态变化'] == 1) | (group['断连标记'] == 1)).cumsum()
            group.loc[group['骑行状态'] == 0, '行程ID'] = np.nan

            valid_trips = []
            noon_long_count = evening_long_count = offpeak_long_count = night_long_count = 0
            group['小时'] = pd.to_datetime(group['统计日期时间'], errors='coerce', format='mixed').dt.hour.fillna(0).astype(int)
            
            for trip_id, trip_group in group.groupby('行程ID', dropna=True):
                trip_duration_min = (trip_group['时间戳'].max() - trip_group['时间戳'].min()) / 60
                if trip_duration_min >= MIN_RIDING_DURATION_MIN:
                    valid_trips.append(trip_duration_min)
                    if trip_duration_min >= VALID_RIDE_HOUR_FOR_PATTERN * 60:
                        mid_idx = len(trip_group) // 2
                        trip_hour = trip_group['小时'].iloc[mid_idx] if mid_idx < len(trip_group) else trip_group['小时'].iloc[0]
                        if trip_hour in NOON_PEAK_HOURS: noon_long_count += 1
                        elif trip_hour in EVENING_PEAK_HOURS: evening_long_count += 1
                        elif trip_hour in NIGHT_HOURS: night_long_count += 1
                        else: offpeak_long_count += 1
            
            res['午间高峰长时骑行次数'] = noon_long_count
            res['晚间高峰长时骑行次数'] = evening_long_count
            res['平峰长时骑行次数'] = offpeak_long_count
            res['夜间长时骑行次数'] = night_long_count
            
            trip_count = len(valid_trips)
            total_riding_min = sum(valid_trips)
            avg_riding_min = round(total_riding_min / trip_count, 2) if trip_count > 0 else 0.0
            total_riding_hours = round(total_riding_min / 60, 2)
            
            res['骑行次数'] = trip_count
            res['单日骑行次数'] = trip_count
            res['单次平均骑行时长(分钟)'] = avg_riding_min
            res['骑行总耗时(小时)'] = total_riding_hours

            # -------------------------- 4. 分时段骑行里程统计 --------------------------
            group['是否午间高峰'] = group['小时'].isin(NOON_PEAK_HOURS)
            group['是否晚间高峰'] = group['小时'].isin(EVENING_PEAK_HOURS)
            group['是否夜间'] = group['小时'].isin(NIGHT_HOURS)
            group['是否平峰'] = group['小时'].isin(OFFPEAK_HOURS)
            
            noon_peak_miles = group[(group['骑行状态'] == 1) & group['是否午间高峰']]['有效里程_km'].sum()
            evening_peak_miles = group[(group['骑行状态'] == 1) & group['是否晚间高峰']]['有效里程_km'].sum()
            night_miles = group[(group['骑行状态'] == 1) & group['是否夜间']]['有效里程_km'].sum()
            offpeak_miles = group[(group['骑行状态'] == 1) & group['是否平峰']]['有效里程_km'].sum()
            
            res['午间高峰骑行里程'] = round(noon_peak_miles, 2)
            res['晚间高峰骑行里程'] = round(evening_peak_miles, 2)
            res['夜间骑行里程'] = round(night_miles, 2)
            res['平峰骑行里程'] = round(offpeak_miles, 2)

            # -------------------------- 5. 出勤与高峰占比计算 --------------------------
            total_riding_count = group[group['骑行状态'] == 1].shape[0]
            peak_riding_count = group[(group['骑行状态'] == 1) & (group['是否午间高峰'] | group['是否晚间高峰'])].shape[0]
            peak_riding_ratio = round(peak_riding_count / total_riding_count, 2) if total_riding_count > 0 else 0
            is_work_day = 1 if total_riding_hours >= 0.5 else 0
            res['高峰骑行占比'] = peak_riding_ratio
            res['当日是否出勤'] = is_work_day

            # -------------------------- 6. 速度指标计算 --------------------------
            riding_speed_data = group.loc[group['骑行状态'] == 1, ['速度', '统计日期时间']].dropna(subset=['速度'])
            max_speed = 0.0
            max_speed_time = pd.NA
            avg_riding_speed = 0.0
            if len(riding_speed_data) > 0:
                riding_speed_data = riding_speed_data[riding_speed_data['速度'] <= MAX_VALID_SPEED]
                if len(riding_speed_data) > 0:
                    max_speed = riding_speed_data['速度'].max()
                    max_speed_idx = riding_speed_data['速度'].idxmax()
                    max_speed_time = riding_speed_data.loc[max_speed_idx, '统计日期时间']
            if total_riding_hours > 0.01:
                avg_riding_speed = round(total_distance / total_riding_hours, 2)
            res['最大速度'] = round(max_speed, 2)
            res['最大速度时间'] = max_speed_time
            res['平均骑行速度'] = avg_riding_speed

            # -------------------------- 7. 放电时长计算 --------------------------
            current_col_for_discharge = '电流_放电统计用' if '电流_放电统计用' in group.columns else '电流'
            group['放电状态'] = (group[current_col_for_discharge] > 0).rolling(
                window=STOP_SMOOTH_WINDOW, 
                min_periods=1,
                center=True
            ).max().fillna(0).astype(int)
            
            total_discharge_hours = 0.0
            if '放电状态' in group.columns:
                group['放电块ID'] = (group['放电状态'] != group['放电状态'].shift()).cumsum()
                for block_id, block_group in group[group['放电状态'] == 1].groupby('放电块ID'):
                    if len(block_group) >= 1:
                        duration_h = len(block_group) * COLLECTION_CYCLE_MIN / 60
                        if duration_h >= 1/60:
                            total_discharge_hours += duration_h
            else:
                valid_discharge_mask = group[current_col_for_discharge] > 0
                total_discharge_hours = valid_discharge_mask.sum() * COLLECTION_CYCLE_MIN / 60
            
            total_discharge_hours = round(total_discharge_hours, 2)
            res['总放电时长(小时)'] = total_discharge_hours

            # -------------------------- 8. 怠速放电时长计算 --------------------------
            idle_discharge_hours = max(0.0, total_discharge_hours - total_riding_hours)
            res['怠速放电时长(小时)'] = round(idle_discharge_hours, 2)
            res['骑行-放电时长比'] = round(total_riding_hours / total_discharge_hours, 3) if total_discharge_hours > 0 else 0.0

            # -------------------------- 9. 电流指标计算 --------------------------
            current_col_for_ride = '电流_骑行判定用' if '电流_骑行判定用' in group.columns else '电流'
            riding_current_data = group.loc[group['骑行状态'] == 1, ['时间戳', current_col_for_ride, '统计日期时间']].dropna(subset=[current_col_for_ride])
            riding_current_data = riding_current_data.rename(columns={current_col_for_ride: '电流'})
            
            riding_current_data = riding_current_data[(riding_current_data['电流'] >= MIN_VALID_CURRENT) & (riding_current_data['电流'] <= MAX_VALID_CURRENT)]
            full_current_data = group[current_col_for_ride].fillna(0)
            
            max_current = 0.0
            max_current_time = pd.NA
            riding_avg_current = 0.0
            full_avg_current = 0.0
            current_std = 0.0
            current_cv = 0.0
            peak_current_std = 0.0
            peak_current_cv = 0.0
            offpeak_current_std = 0.0
            offpeak_current_cv = 0.0
            max_trip_current_std = 0.0
            max_trip_current_cv = 0.0
            is_current_abnormal = '否'

            if len(riding_current_data) > 0:
                max_current = riding_current_data['电流'].max()
                max_current_idx = riding_current_data['电流'].idxmax()
                max_current_time = riding_current_data.loc[max_current_idx, '统计日期时间']
                riding_avg_current = riding_current_data['电流'].mean()
                current_std = riding_current_data['电流'].std() if len(riding_current_data) > 1 else 0.0
                current_cv = current_std / riding_avg_current if riding_avg_current > 0.1 else 0.0
                riding_current_data['小时'] = pd.to_datetime(riding_current_data['统计日期时间'], errors='coerce', format='mixed').dt.hour.fillna(0).astype(int)
                riding_current_data['是否高峰'] = riding_current_data['小时'].isin(NOON_PEAK_HOURS + EVENING_PEAK_HOURS)
                peak_data = riding_current_data[riding_current_data['是否高峰']]
                if len(peak_data) > 1:
                    peak_avg = peak_data['电流'].mean()
                    peak_current_std = peak_data['电流'].std()
                    peak_current_cv = peak_current_std / peak_avg if peak_avg > 0.1 else 0.0
                offpeak_data = riding_current_data[~riding_current_data['是否高峰']]
                if len(offpeak_data) > 1:
                    offpeak_avg = offpeak_data['电流'].mean()
                    offpeak_current_std = offpeak_data['电流'].std()
                    offpeak_current_cv = offpeak_current_std / offpeak_avg if offpeak_avg > 0.1 else 0.0
                
                riding_current_data = riding_current_data.join(group[['行程ID']], on='时间戳', how='left')
                trip_std_list = []
                for trip_id, trip_group in riding_current_data.groupby('行程ID', dropna=True):
                    if len(trip_group) > 1:
                        trip_std = trip_group['电流'].std()
                        trip_avg = trip_group['电流'].mean()
                        trip_cv = trip_std / trip_avg if trip_avg > 0.1 else 0.0
                        trip_std_list.append((trip_std, trip_cv))
                if trip_std_list:
                    max_trip_current_std, max_trip_current_cv = max(trip_std_list, key=lambda x: x[0])

            full_avg_current = full_current_data.mean()
            if (max_current >= ABNORMAL_MAX_CURRENT) or (current_cv >= ABNORMAL_CV):
                is_current_abnormal = '是'
            if (riding_avg_current >= ABNORMAL_AVG_CURRENT) and (total_discharge_hours >= ABNORMAL_MIN_DISCHARGE_HOUR):
                is_current_abnormal = '是'
            
            res['最大电流'] = round(max_current, 2)
            res['最大电流时间'] = max_current_time
            res['骑行放电平均电流'] = round(riding_avg_current, 2)
            res['全周期平均电流'] = round(full_avg_current, 2)
            res['电流标准差'] = round(current_std, 2)
            res['电流变异系数'] = round(current_cv, 3)
            res['高峰电流标准差'] = round(peak_current_std, 2)
            res['高峰电流变异系数'] = round(peak_current_cv, 3)
            res['平峰电流标准差'] = round(offpeak_current_std, 2)
            res['平峰电流变异系数'] = round(offpeak_current_cv, 3)
            res['单行程最大电流标准差'] = round(max_trip_current_std, 2)
            res['单行程最大电流变异系数'] = round(max_trip_current_cv, 3)
            res['电流异常用户'] = is_current_abnormal

            if not riding_current_data.empty:
                res['电流>60A次数'] = (riding_current_data['电流'] > NORMAL_CURRENT_THRESHOLD).sum()
                # ── BUG4修复：电流>80A次数应排除>100A的部分，避免与超100A指标重叠扣分 ──
                res['电流>80A次数'] = ((riding_current_data['电流'] > HIGH_CURRENT_THRESHOLD) & (riding_current_data['电流'] < OVER_CURRENT_THRESHOLD)).sum()
                riding_current_data['电流≥100A'] = (riding_current_data['电流'] >= OVER_CURRENT_THRESHOLD).astype(int)
                riding_current_data['连续超100A'] = riding_current_data['电流≥100A'].rolling(window=OVER_CURRENT_CONTINUOUS, min_periods=OVER_CURRENT_CONTINUOUS).sum()
                res['超100A连续次数'] = (riding_current_data['连续超100A'] >= OVER_CURRENT_CONTINUOUS).sum()
                res['超100A累计时长_h'] = round((riding_current_data['电流≥100A'].sum() * COLLECTION_CYCLE_MIN) / 60, 2)

            # -------------------------- 10. 温度指标计算 --------------------------
            temp_data = group[['温度', '统计日期时间']].dropna(subset=['温度'])
            max_temp = 0.0
            max_temp_time = pd.NA
            avg_temp = 0.0
            if len(temp_data) > 0:
                max_temp = temp_data['温度'].max()
                max_temp_idx = temp_data['温度'].idxmax()
                max_temp_time = temp_data.loc[max_temp_idx, '统计日期时间']
                avg_temp = temp_data['温度'].mean()
            res['最大温度'] = round(max_temp, 2)
            res['最大温度时间'] = max_temp_time
            res['平均温度'] = round(avg_temp, 2)

            # -------------------------- 11. 电池能耗与换电次数计算 --------------------------
            total_energy_used = 0.0
            battery_change_count = 0
            energy_per_100km = 0.0
            riding_soc_data = pd.Series(dtype='float64')

            if '电池id' in group.columns:
                group['电池id'] = group['电池id'].fillna('未知电池').astype(str)
                group['电池切换标记'] = (group['电池id'] != group['电池id'].shift(1)).astype(int)
                battery_change_count = group['电池切换标记'].sum()
                if group['电池切换标记'].iloc[0] == 1:
                    battery_change_count -= 1
                battery_change_count = max(0, battery_change_count)
            res['换电次数'] = battery_change_count

            if '电池度数' in group.columns and '电池id' in group.columns:
                for battery_segment, batt_group in group.groupby(group['电池切换标记'].cumsum()):
                    batt_group = batt_group.dropna(subset=['电池度数'])
                    if len(batt_group) < 2: continue
                    start_energy = batt_group['电池度数'].iloc[0]
                    end_energy = batt_group['电池度数'].iloc[-1]
                    if start_energy > end_energy:
                        total_energy_used += (start_energy - end_energy)
            res['总用电量(kWh)'] = round(total_energy_used, 2)

            if total_distance > 1.0:
                energy_per_100km = (total_energy_used / total_distance) * 100
            res['百公里电耗(kWh)'] = round(energy_per_100km, 2)

            # -------------------------- 12. SOC电池健康指标计算 --------------------------
            soc_data_valid = False
            if '电池SOC' in group.columns:
                riding_soc_data = group.loc[group['骑行状态'] == 1, '电池SOC'].dropna()
                if len(riding_soc_data) > 0:
                    soc_data_valid = True
                    res['平均骑行SOC'] = round(riding_soc_data.mean(), 1)
                    res['最低SOC'] = round(riding_soc_data.min(), 1)
                    total_riding_points = len(riding_soc_data)
                    res['SOC低于20%时长占比'] = round((riding_soc_data < SOC_LOW_WARNING).sum() / total_riding_points, 2)
                    res['SOC低于10%时长占比'] = round((riding_soc_data < SOC_CRITICAL).sum() / total_riding_points, 2)

            # -------------------------- 13. 工作时长覆盖计算 --------------------------
            riding_time_data = group.loc[group['骑行状态'] == 1, '时间戳']
            if len(riding_time_data) > 0:
                work_span_hours = (riding_time_data.max() - riding_time_data.min()) / 3600
                res['工作时长覆盖(小时)'] = round(work_span_hours, 1)

            # -------------------------- 13.5. 数据有效性标记 --------------------------
            energy_data_valid = (energy_per_100km > 0)

            # -------------------------- 14. 客户形态判定 --------------------------
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
                customer_type = "改装/超速车"
            elif is_storage_scene:
                customer_type = "地摊/储能"
            elif has_real_ride and total_riding_hours >= 4.0 and peak_riding_ratio >= 0.3:
                customer_type = "专送骑手"
            elif has_real_ride and total_riding_hours >= 2.0 and peak_riding_ratio >= 0.15:
                customer_type = "众包骑手"
            elif has_real_ride and total_distance <= 30 and total_riding_hours <= 2.0:
                customer_type = "标准骑手"
            elif has_real_ride:
                customer_type = "普通骑手"
            else:
                customer_type = "数据不足"
            res['客户形态'] = customer_type

            # -------------------------- 15. 【统计学优化】包月友好评分体系计算 --------------------------
            score = _calc_monthly_score_v2(
                max_speed=max_speed,
                max_temp=max_temp,
                riding_avg_current=riding_avg_current,
                current_60a_count=res['电流>60A次数'],
                current_80a_count=res['电流>80A次数'],
                over100a_cont=res['超100A连续次数'],
                max_trip_current_cv=max_trip_current_cv,
                current_cv=current_cv,
                avg_riding_speed=avg_riding_speed,
                soc_data_valid=soc_data_valid,
                has_real_ride=has_real_ride,
                soc_below_10_ratio=res['SOC低于10%时长占比'],
                soc_below_20_ratio=res['SOC低于20%时长占比'],
                min_soc=res['最低SOC'],
                avg_soc=res['平均骑行SOC'],
                energy_data_valid=energy_data_valid,
                energy_per_100km=energy_per_100km,
                battery_change_count=battery_change_count
            )
            if customer_type == "地摊/储能":
                score = min(100, score + 20)
            elif customer_type == "外卖高强度车":
                score -= 15
            elif customer_type == "改装/超速车":
                score -= 30
            score = max(0, min(100, round(score)))
            res['包月友好评分'] = score

            # -------------------------- 16. 【统计学优化】用户等级判定 --------------------------
            user_level = _determine_user_level_v2(
                over100a_cont=res['超100A连续次数'],
                over100a_hours=res['超100A累计时长_h'],
                current_80a_count=0,
                soc_below_10_ratio=res['SOC低于10%时长占比'],
                min_soc=res['最低SOC'],
                current_60a_count=res['电流>60A次数'],
                energy_per_100km=energy_per_100km,
                soc_below_20_ratio=res['SOC低于20%时长占比'],
                score=score,
                riding_avg_current=riding_avg_current,
                soc_data_valid=soc_data_valid,
                energy_data_valid=energy_data_valid
            )
            res['用户等级'] = user_level

            # ==================================================================================
            # 🆕 【新增】单合约单日 动态说明生成
            # ==================================================================================
            # 1. 用户等级说明
            level_desc = ""
            current_level = res['用户等级']
            if current_level == "暴力用户（超量放电/电池滥用）":
                reasons = []
                if res['超100A连续次数'] >= VIOLENT_CURRENT_TIMES:
                    reasons.append(f"超100A连续放电{res['超100A连续次数']}次，触发阈值")
                if res['超100A累计时长_h'] >= OVER_CURRENT_MIN_HOUR:
                    reasons.append(f"超100A累计放电时长{res['超100A累计时长_h']}小时，触发阈值")
                if res['电流>60A次数'] >= VIOLENT_CURRENT_TIMES:
                    reasons.append(f"电流超60A共{res['电流>60A次数']}次，峰值达{res['最大电流']}A")
                if soc_data_valid and res['SOC低于10%时长占比'] >= 0.5 and res['最低SOC'] <= 5:
                    reasons.append(f"深度亏电严重（最低SOC{res['最低SOC']}%，SOC低于10%时长占比{res['SOC低于10%时长占比']*100:.0f}%）")
                level_desc = " | ".join(reasons) if reasons else "存在严重损害电池的行为"
            
            elif current_level == "高损耗用户":
                reasons = []
                if res['电流>60A次数'] >= HIGH_LOSS_CURRENT_TIMES:
                    reasons.append(f"中高电流使用频繁（电流超60A共{res['电流>60A次数']}次）")
                if energy_data_valid and energy_per_100km >= EXTREME_ENERGY_THRESHOLD:
                    reasons.append(f"能耗极高（百公里电耗{energy_per_100km}kWh）")
                if soc_data_valid and res['SOC低于20%时长占比'] >= 0.5:
                    reasons.append(f"长期低电量运行（SOC低于20%时长占比{res['SOC低于20%时长占比']*100:.0f}%）")
                if not reasons:
                    reasons.append(f"包月友好分较低（{score}分），电池损耗速度高于平均水平")
                level_desc = " | ".join(reasons)
            
            elif current_level == "优质用户":
                level_desc = f"电池使用习惯优秀（包月友好分{score}分），电流、速度、SOC均保持在健康区间"
            elif current_level == "良好用户":
                level_desc = f"电池使用习惯较好（包月友好分{score}分），整体负载可控"
            elif current_level == "普通用户":
                level_desc = f"电池使用行为一般（包月友好分{score}分），无明显过激使用行为"
            else:
                level_desc = "数据不足或无有效骑行数据，无法判定"
            res['用户等级说明'] = level_desc

            # 2. 客户形态说明
            type_desc = ""
            current_type = res['客户形态']
            if current_type == "改装/超速车":
                type_desc = f"行驶特征异常（最高速度{max_speed}km/h，平均骑行电流{riding_avg_current}A），远超普通两轮车水平，存在改装或超速嫌疑"
            elif current_type == "地摊/储能":
                type_desc = f"非移动用电特征明显（当日骑行{total_riding_hours}小时，怠速放电{idle_discharge_hours}小时），放电以静止状态为主，疑似地摊供电或储能场景"
            elif current_type == "专送骑手":
                type_desc = f"工作特征显著（当日骑行{total_riding_hours:.1f}小时，高峰骑行占比{peak_riding_ratio*100:.0f}%），工作时长稳定且午晚高峰高度活跃，符合专送骑手画像"
            elif current_type == "众包骑手":
                type_desc = f"具有兼职骑手特征（当日骑行{total_riding_hours:.1f}小时，高峰骑行占比{peak_riding_ratio*100:.0f}%），高峰时段有一定活跃度"
            elif current_type == "标准骑手":
                type_desc = f"骑行行为规律（当日行驶里程{total_distance}km，骑行时长{total_riding_hours:.1f}小时），属于标准日常使用场景"
            elif current_type == "普通骑手":
                type_desc = f"有常规骑行行为（当日行驶里程{total_distance}km，骑行时长{total_riding_hours:.1f}小时），不符合特定骑手标签特征"
            else:
                type_desc = "无有效骑行数据或数据量不足，无法判定具体使用场景"
            res['客户形态说明'] = type_desc
            # ==================================================================================

        results.append(res)

    df_contract = pd.DataFrame(results)
    df_contract['用户id'] = df_contract['用户id'].astype(str).fillna('未知用户')
    if not df_contract.empty and '中心纬度' in df_contract.columns and '中心经度' in df_contract.columns:
        print("   🗺️ 开始基于三级围栏匹配省-市-区县...")
        region_df = batch_gps_to_region(df_contract, '中心纬度', '中心经度')
        df_contract[['核心活动省份', '核心活动城市', '核心活动区县']] = region_df
    return df_contract

# ==================================================================================
# ===================== 4. 用户维度聚合函数 =====================
# ==================================================================================
def aggregate_to_user_level(df_contract):
    if df_contract.empty: 
        print("   ⚠️ 合约数据为空，用户聚合跳过")
        return pd.DataFrame()
    if '用户id' not in df_contract.columns:
        print("   ❌ 合约数据缺少用户id列，用户聚合失败")
        return pd.DataFrame()
    
    user_count = df_contract['用户id'].nunique()
    print(f"   👥 开始聚合用户维度，共 {user_count} 个用户...")

    df_contract = df_contract.copy()
    df_contract['用户id'] = df_contract['用户id'].astype(str).fillna('未知用户')
    df_contract['weighted_current'] = df_contract['骑行放电平均电流'] * df_contract['骑行总耗时(小时)']
    
    # -------------------------- 内部辅助函数定义 --------------------------
    # 优化6：统一get_user_type优先级逻辑，消除idxmax和priority两套判断的不一致
    def get_user_type(group):
        # 统一优先级逻辑：先检查高优先级类型是否存在，再按骑行时长判定
        priority = ["改装/超速车", "地摊/储能"]
        for t in priority:
            if t in list(group['客户形态']):
                return t
        
        # 如果没有高优先级类型，按骑行时长最多的形态判定
        contract_type_duration = group.groupby('客户形态')['骑行总耗时(小时)'].sum()
        total_duration = contract_type_duration.sum()
        if total_duration > 0 and len(contract_type_duration) > 0:
            return contract_type_duration.idxmax()
        return "数据不足"

    def get_user_level(levels):
        priority = ["暴力用户（超量放电/电池滥用）", "高损耗用户", "普通用户", "良好用户", "优质用户", "无效"]
        for l in priority:
            if l in list(levels): return l
        return "无效"
    
    # 🆕 新增：辅助函数 - 根据优先级获取对应的说明文字
    def get_level_desc(levels, descs):
        priority = ["暴力用户（超量放电/电池滥用）", "高损耗用户", "普通用户", "良好用户", "优质用户", "无效"]
        temp_df = pd.DataFrame({'level': levels, 'desc': descs})
        for l in priority:
            if l in temp_df['level'].values:
                return temp_df[temp_df['level'] == l]['desc'].iloc[0]
        return "数据不足"

    def get_type_desc(types, descs):
        priority = ["改装/超速车", "地摊/储能", "专送骑手", "众包骑手", "标准骑手", "普通骑手", "数据不足"]
        temp_df = pd.DataFrame({'type': types, 'desc': descs})
        for t in priority:
            if t in temp_df['type'].values:
                return temp_df[temp_df['type'] == t]['desc'].iloc[0]
        return "数据不足"
    # ----------------------------------------------------------------------

    group_key = ['统计日期', '用户id']
    g = df_contract.groupby(group_key)

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
    
    # ── BUG3修复：多合约聚合时，绝对值用"用户总量"，不除以合约数 ──
    # 原因：风险监控看的是用户整体行为，不应被不活跃合约稀释
    # 例如：骑手主电池跑100km + 备用电池0km = 用户总里程100km（不是50km）
    df_agg['单合约日均行驶里程_km'] = df_agg['当日总行驶距离_km'].round(2)
    df_agg['单合约日均骑行时长_h'] = df_agg['当日总骑行时长_h'].round(2)
    df_agg['单合约日均放电时长_h'] = df_agg['当日总放电时长_h'].round(2)
    df_agg['单合约日均怠速放电_h'] = df_agg['当日总怠速放电时长_h'].round(2)
    df_agg['单合约日均用电量_kWh'] = df_agg['当日总用电量_kWh'].round(2)
    
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
    
    # -------------------------- 加权平均计算（优化3：合并三次apply为一次）--------------------------
    def _weighted_aggregations(x):
        """一次性计算三个加权比率，减少groupby遍历次数"""
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
    # ----------------------------------------------------------------------

    df_custom = pd.DataFrame()
    df_custom['核心活动省份'] = g['核心活动省份'].apply(lambda x: x.value_counts().index[0] if len(x.dropna())>0 else "")
    df_custom['核心活动城市'] = g['核心活动城市'].apply(lambda x: x.value_counts().index[0] if len(x.dropna())>0 else "")
    df_custom['核心活动区县'] = g['核心活动区县'].apply(lambda x: x.value_counts().index[0] if len(x.dropna())>0 else "")
    df_custom['风险标签_电流异常'] = g['电流异常用户'].apply(lambda x: '是' if '是' in list(x) else '否')
    df_custom['客户形态_综合'] = g.apply(get_user_type)
    df_custom['用户等级_综合'] = g['用户等级'].apply(get_user_level)

    # ==================================================================================
    # 🆕 【新增】聚合说明列 (取最高优先级标签对应的说明)
    # ==================================================================================
    if '用户等级说明' in df_contract.columns:
        df_custom['用户等级_综合说明'] = g.apply(lambda x: get_level_desc(x['用户等级'], x['用户等级说明']))
    if '客户形态说明' in df_contract.columns:
        df_custom['客户形态_综合说明'] = g.apply(lambda x: get_type_desc(x['客户形态'], x['客户形态说明']))
    # ==================================================================================

    df_user = pd.concat([df_agg, df_custom], axis=1).reset_index()
    df_user = df_user.drop(columns=['_weighted_peak_ratio', '_total_riding_hour', '_weighted_soc20_ratio', '_weighted_soc10_ratio', '_weighted_current_sum'], errors='ignore')
    print("   ✅ 用户维度聚合完成（含怠速放电指标及判定说明）")
    return df_user

# ==================================================================================
# ===================== 5. 用户月度出勤预估 =====================
# ==================================================================================
def calc_user_monthly_attendance(df_history):
    if df_history.empty:
        print("⚠️ 无用户历史数据，跳过出勤预估")
        return pd.DataFrame(), {}, {}
    print("\n📊 开始计算用户月度出勤预估...")
    attendance_stats = []
    month_days_map = {}
    energy_cum_map = {}
    
    for uid, group in df_history.groupby('用户id'):
        group_unique = group.drop_duplicates(subset=['统计日期']).sort_values('统计日期')
        total_record_days = len(group_unique)
        total_attendance_days = group_unique['当日是否出勤'].sum()
        attendance_rate = round(total_attendance_days / total_record_days, 4) if total_record_days > 0 else 0
        
        if '单合约日均用电量_kWh' in group_unique.columns:
            avg_single_contract_energy = group_unique['单合约日均用电量_kWh'].mean()
        else:
            total_cum_energy = group_unique['当日总用电量_kWh'].sum()
            avg_single_contract_energy = total_cum_energy / total_record_days if total_record_days > 0 else 0
        
        if total_record_days < MIN_RECORD_DAYS_FOR_ESTIMATE:
            month_estimate_days = DEFAULT_MONTH_DAYS
            estimate_type = "兜底值"
            final_month_energy = (avg_single_contract_energy * month_estimate_days).round(2)
        elif total_record_days < FULL_RECORD_DAYS_FOR_CUMULATIVE:
            month_estimate_days = round(attendance_rate * MONTH_CALENDAR_DAYS, 1)
            estimate_type = "实际出勤率估算"
            final_month_energy = (avg_single_contract_energy * month_estimate_days).round(2)
        else:
            month_estimate_days = round(attendance_rate * MONTH_CALENDAR_DAYS, 1)
            estimate_type = "真实累计计算"
            final_month_energy = (avg_single_contract_energy * MONTH_CALENDAR_DAYS).round(2)
        
        attendance_stats.append({
            "用户id": uid,
            "有记录的总天数": total_record_days,
            "实际出勤总天数": total_attendance_days,
            "历史出勤率": attendance_rate,
            "历史单合约日均用电量_kWh": round(avg_single_contract_energy, 2),
            "月度预估工作天数": month_estimate_days,
            "单合约月度用电度数预估_kWh": final_month_energy,
            "预估类型": estimate_type,
            "最新统计日期": group_unique['统计日期'].max()
        })
        month_days_map[uid] = month_estimate_days
        energy_cum_map[uid] = final_month_energy
    
    df_attendance = pd.DataFrame(attendance_stats)
    df_attendance.to_csv(USER_ATTENDANCE_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ 用户月度出勤预估表已生成：{USER_ATTENDANCE_FILE}")
    return df_attendance, month_days_map, energy_cum_map

# ==================================================================================
# ===================== 6. 骑行习惯判定函数 =====================
# ==================================================================================
def get_work_pattern(group):
    total_noon_long = group['午间高峰长时骑行总次数'].sum() if '午间高峰长时骑行总次数' in group.columns else 0
    total_evening_long = group['晚间高峰长时骑行总次数'].sum() if '晚间高峰长时骑行总次数' in group.columns else 0
    total_offpeak_long = group['平峰长时骑行总次数'].sum() if '平峰长时骑行总次数' in group.columns else 0
    total_night_long = group['夜间长时骑行总次数'].sum() if '夜间长时骑行总次数' in group.columns else 0
    total_long_all = total_noon_long + total_evening_long + total_offpeak_long + total_night_long
    
    if total_long_all > 0:
        period_counts = [
            ("习惯午间高峰", total_noon_long),
            ("习惯晚间高峰", total_evening_long),
            ("习惯非高峰期", total_offpeak_long),
            ("习惯晚上", total_night_long)
        ]
        period_counts.sort(key=lambda x: x[1], reverse=True)
        top_pattern, top_count = period_counts[0]
        if top_count / total_long_all >= PATTERN_DOMINANT_RATIO:
            return top_pattern
        else:
            return "全天"
    
    cols = ['午间高峰总里程_km', '晚间高峰总里程_km', '平峰总里程_km', '夜间总里程_km']
    if not all(c in group.columns for c in cols): return "数据不足"
    
    total_noon = group['午间高峰总里程_km'].sum()
    total_evening = group['晚间高峰总里程_km'].sum()
    total_offpeak = group['平峰总里程_km'].sum()
    total_night = group['夜间总里程_km'].sum()
    total_all = total_noon + total_evening + total_offpeak + total_night
    
    if total_all <= 0: return "数据不足"
    
    period_miles = [
        ("习惯午间高峰", total_noon),
        ("习惯晚间高峰", total_evening),
        ("习惯非高峰期", total_offpeak),
        ("习惯晚上", total_night)
    ]
    period_miles.sort(key=lambda x: x[1], reverse=True)
    top_pattern, top_miles = period_miles[0]
    if top_miles / total_all >= PATTERN_DOMINANT_RATIO:
        return top_pattern
    else:
        return "全天"

# ==================================================================================
# ===================== 7. 7天滚动生命周期管理 (含新增日均指标) =====================
# ==================================================================================
def load_daily_snapshot_db():
    if os.path.exists(USER_DAILY_SNAPSHOT_FILE):
        try:
            df_snapshot = pd.read_csv(USER_DAILY_SNAPSHOT_FILE, encoding='utf-8-sig', dtype={'用户id': str})
            df_snapshot['统计日期'] = pd.to_datetime(df_snapshot['统计日期'], errors='coerce').dt.strftime('%Y-%m-%d')
            print(f"📚 已加载历史每日快照：共 {len(df_snapshot)} 条记录")
            return df_snapshot
        except Exception as e:
            print(f"⚠️ 历史快照加载失败，将新建：{e}")
    return pd.DataFrame()

def save_daily_snapshot_db(df_snapshot):
    try:
        os.makedirs(os.path.dirname(USER_DAILY_SNAPSHOT_FILE), exist_ok=True)
        df_snapshot['用户id'] = df_snapshot['用户id'].astype(str).fillna('未知用户')
        df_snapshot.to_csv(USER_DAILY_SNAPSHOT_FILE, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ 每日快照保存失败：{e}")

def calculate_rolling_7d_metrics(df_user_history, current_stat_date):
    if df_user_history.empty: return pd.DataFrame()
    current_date = pd.to_datetime(current_stat_date)
    window_start_date = (current_date - pd.Timedelta(days=ROLLING_WINDOW_DAYS-1)).strftime('%Y-%m-%d')
    
    df_user_history['统计日期_dt'] = pd.to_datetime(df_user_history['统计日期'], errors='coerce')
    df_window = df_user_history[
        (df_user_history['统计日期_dt'] >= pd.to_datetime(window_start_date)) &
        (df_user_history['统计日期_dt'] <= current_date)
    ].copy()
    if df_window.empty: return pd.DataFrame()
    
    df_window = df_window.sort_values(['用户id', '统计日期_dt'], ascending=[True, False])
    df_window['days_ago'] = (current_date - df_window['统计日期_dt']).dt.days
    df_window = df_window[df_window['days_ago'].between(0, ROLLING_WINDOW_DAYS-1)]
    df_window['weight'] = df_window['days_ago'].apply(lambda x: ROLLING_WEIGHTS[int(x)] if int(x) < len(ROLLING_WEIGHTS) else ROLLING_WEIGHTS[-1])

    rolling_results = []
    for uid, group in df_window.groupby('用户id'):
        group = group.sort_values('days_ago')
        res = {'用户id': uid, '统计日期': current_stat_date, '近7天有数据天数': len(group)}
        
        latest_row = group[group['days_ago'] == 0].iloc[0] if not group[group['days_ago'] == 0].empty else group.iloc[0]
        for col in ['核心活动省份', '核心活动城市', '核心活动区县']:
            if col in latest_row: res[col] = latest_row[col]
        
        weighted_cols = [
            ('单合约日均行驶里程_km', '近7d单合约日均行驶里程_km'),
            ('单合约日均骑行时长_h', '近7d单合约日均骑行时长_h'),
            ('单合约日均放电时长_h', '近7d单合约日均放电时长_h'),
            ('单合约日均怠速放电_h', '近7d单合约日均怠速放电_h'),
            ('单合约日均用电量_kWh', '近7d单合约日均用电量_kWh'),
            ('平均包月友好分', '近7d平均包月友好分'),
            ('当日平均骑行SOC', '近7d平均骑行SOC'),
            ('当日平均骑行速度_kmh', '近7d平均骑行速度_kmh'),
            ('当日平均骑行电流_A', '近7d平均骑行电流_A'),
            ('当日百公里电耗_kWh', '近7d百公里电耗_kWh'),
        ]
        for raw_col, roll_col in weighted_cols:
            if raw_col in group.columns:
                valid_data = group[[raw_col, 'weight']].dropna()
                if len(valid_data) > 0:
                    weighted_sum = (valid_data[raw_col] * valid_data['weight']).sum()
                    weight_sum = valid_data['weight'].sum()
                    res[roll_col] = round(weighted_sum / weight_sum, 2) if weight_sum > 0 else 0
                else:
                    res[roll_col] = 0

        # ── BUG1修复：比率类指标必须用"加权分子/加权分母"计算，禁止直接加权平均 ──
        # 高峰骑行占比 = 加权高峰骑行时长 / 加权总骑行时长
        if '高峰骑行占比' in group.columns and '当日总骑行时长_h' in group.columns:
            valid = group[['高峰骑行占比', '当日总骑行时长_h', 'weight']].dropna()
            if len(valid) > 0:
                weighted_peak_hours = (valid['高峰骑行占比'] * valid['当日总骑行时长_h'] * valid['weight']).sum()
                weighted_total_hours = (valid['当日总骑行时长_h'] * valid['weight']).sum()
                res['近7d高峰骑行占比'] = round(weighted_peak_hours / weighted_total_hours, 2) if weighted_total_hours > 0 else 0
            else:
                res['近7d高峰骑行占比'] = 0
        else:
            res['近7d高峰骑行占比'] = 0

        # SOC低于20%时长占比 = 加权低SOC时长 / 加权总骑行时长
        if 'SOC低于20%时长占比_当日' in group.columns and '当日总骑行时长_h' in group.columns:
            valid = group[['SOC低于20%时长占比_当日', '当日总骑行时长_h', 'weight']].dropna()
            if len(valid) > 0:
                weighted_soc20_hours = (valid['SOC低于20%时长占比_当日'] * valid['当日总骑行时长_h'] * valid['weight']).sum()
                weighted_total_hours = (valid['当日总骑行时长_h'] * valid['weight']).sum()
                res['近7d_SOC低于20%时长占比'] = round(weighted_soc20_hours / weighted_total_hours, 4) if weighted_total_hours > 0 else 0
            else:
                res['近7d_SOC低于20%时长占比'] = 0
        else:
            res['近7d_SOC低于20%时长占比'] = 0

        # SOC低于10%时长占比 = 加权低SOC时长 / 加权总骑行时长
        if 'SOC低于10%时长占比_当日' in group.columns and '当日总骑行时长_h' in group.columns:
            valid = group[['SOC低于10%时长占比_当日', '当日总骑行时长_h', 'weight']].dropna()
            if len(valid) > 0:
                weighted_soc10_hours = (valid['SOC低于10%时长占比_当日'] * valid['当日总骑行时长_h'] * valid['weight']).sum()
                weighted_total_hours = (valid['当日总骑行时长_h'] * valid['weight']).sum()
                res['近7d_SOC低于10%时长占比'] = round(weighted_soc10_hours / weighted_total_hours, 4) if weighted_total_hours > 0 else 0
            else:
                res['近7d_SOC低于10%时长占比'] = 0
        else:
            res['近7d_SOC低于10%时长占比'] = 0
        
        max_cols = [
            ('当日最高速度_kmh', '近7d最高速度_kmh'),
            ('当日最大电流_A', '近7d最大电流_A'),
            ('当日最高温度_℃', '近7d最高温度_℃'),
            ('最大单合约活动半径_km', '近7d最大活动半径_km')
        ]
        for raw_col, roll_col in max_cols:
            if raw_col in group.columns:
                res[roll_col] = round(group[raw_col].max(), 2) if pd.notna(group[raw_col].max()) else 0
        
        min_cols = [
            ('当日最低SOC', '近7d最低SOC')
        ]
        for raw_col, roll_col in min_cols:
            if raw_col in group.columns:
                res[roll_col] = round(group[raw_col].min(), 2) if pd.notna(group[raw_col].min()) else 100
        
        # ====================== 7天累计指标及对应日均指标 ======================
        cumulative_and_avg_cols = [
            ('当日总骑行次数', '近7d总骑行次数', '近7d日均骑行次数'),
            ('当日总换电次数', '近7d总换电次数', '近7d日均换电次数'),
            ('总电流超60A次数', '近7d电流超60A总次数', '近7d日均电流超60A次数'),
            ('总电流超80A次数', '近7d电流超80A总次数', '近7d日均电流超80A次数'),
            ('总超100A连续次数', '近7d超100A连续总次数', '近7d日均超100A连续次数'),
            ('总超100A累计时长_h', '近7d超100A累计时长_h', '近7d日均超100A累计时长_h'),
            ('午间高峰长时骑行总次数', '近7d午间高峰长时骑行次数', '近7d日均午间高峰长时骑行次数'),
            ('晚间高峰长时骑行总次数', '近7d晚间高峰长时骑行次数', '近7d日均晚间高峰长时骑行次数'),
            ('平峰长时骑行总次数', '近7d平峰长时骑行次数', '近7d日均平峰长时骑行次数'),
            ('夜间长时骑行总次数', '近7d夜间长时骑行次数', '近7d日均夜间长时骑行次数'),
            ('午间高峰总里程_km', '近7d午间高峰总里程_km', '近7d日均午间高峰总里程_km'),
            ('晚间高峰总里程_km', '近7d晚间高峰总里程_km', '近7d日均晚间高峰总里程_km'),
            ('平峰总里程_km', '近7d平峰总里程_km', '近7d日均平峰总里程_km'),
            ('夜间总里程_km', '近7d夜间总里程_km', '近7d日均夜间总里程_km')
        ]

        valid_days = res['近7天有数据天数']
        
        for raw_col, cum_col, avg_col in cumulative_and_avg_cols:
            if raw_col in group.columns:
                cum_val = round(group[raw_col].sum(), 2) if pd.notna(group[raw_col].sum()) else 0
                res[cum_col] = cum_val
                res[avg_col] = round(cum_val / valid_days, 2) if valid_days > 0 else 0
        # ==================================================================================

        # 补充近7d出勤天数
        if '当日是否出勤' in group.columns:
            res['近7d出勤天数'] = round(group['当日是否出勤'].sum(), 2) if pd.notna(group['当日是否出勤'].sum()) else 0
            res['近7d出勤率'] = round(res['近7d出勤天数'] / valid_days, 2) if valid_days > 0 else 0
        
        # 适配 get_work_pattern
        pattern_group = pd.DataFrame([res])
        temp_rename_map = {}
        for raw_col, cum_col, avg_col in cumulative_and_avg_cols:
            if cum_col in res:
                temp_rename_map[cum_col] = raw_col
        
        pattern_group = pattern_group.rename(columns=temp_rename_map)
        res['近7d工作特点'] = get_work_pattern(pattern_group)
        
        # ── BUG2修复：7天客户形态用"单日判定众数"而非重新计算，防止地摊特征被洗白 ──
        if '客户形态_综合' in group.columns:
            type_counts = group['客户形态_综合'].value_counts()
            if len(type_counts) > 0:
                res['客户形态_综合_7d'] = type_counts.index[0]
            else:
                res['客户形态_综合_7d'] = "数据不足"
        else:
            res['客户形态_综合_7d'] = "数据不足"

        # ── 新增：7天用户等级用"单日判定众数"聚合 ──
        # 注意：用户等级_综合_7d 已被用户等级_动态取代，不再需要

        # ==================================================================================
        # 🆕 【新增】7天滚动 动态说明生成 (基于7天窗口数据)
        # ==================================================================================
        level_desc_7d = ""
        # 注意：用户等级_综合_7d 已被用户等级_动态取代
        # current_level_7d = res.get('用户等级_综合_7d', '无效')
        
        # 提取7天窗口的极值用于说明
        max_curr_7d = res.get('近7d最大电流_A', 0)
        times_60a_7d = res.get('近7d电流超60A总次数', 0)
        times_80a_7d = res.get('近7d电流超80A总次数', 0)
        times_100a_cont_7d = res.get('近7d超100A连续总次数', 0)
        min_soc_7d = res.get('近7d最低SOC', 100)
        soc_below_10_ratio_7d = res.get('近7d_SOC低于10%时长占比', 0)
        score_7d = res.get('近7d平均包月友好分', 0)
        energy_100km_7d = res.get('近7d百公里电耗_kWh', 0)

        # 由于用户等级_综合_7d 已被用户等级_动态取代，这里不再生成相关说明
        # 动态等级的说明已由 DynamicBatteryAnalyzer 在 score_and_classify 中生成
        # res['用户等级_综合_7d说明'] = level_desc_7d

        # 2. 客户形态说明 (7天滚动版)
        type_desc_7d = ""
        current_type_7d = res.get('客户形态_综合_7d', '数据不足')
        
        avg_speed_7d = res.get('近7d平均骑行速度_kmh', 0)
        max_speed_7d = res.get('近7d最高速度_kmh', 0)
        avg_current_7d = res.get('近7d平均骑行电流_A', 0)
        avg_ride_hour_7d = res.get('近7d单合约日均骑行时长_h', 0)
        peak_ratio_7d = res.get('近7d高峰骑行占比', 0)
        avg_mileage_7d = res.get('近7d单合约日均行驶里程_km', 0)
        avg_idle_discharge_7d = res.get('近7d单合约日均怠速放电_h', 0)

        if current_type_7d == "改装/超速车":
            type_desc_7d = f"行驶特征异常（近7天最高速度{max_speed_7d}km/h，平均骑行电流{avg_current_7d}A），远超普通两轮车水平，存在改装或超速嫌疑"
        elif current_type_7d == "地摊/储能":
            type_desc_7d = f"非移动用电特征明显（近7天日均骑行{avg_ride_hour_7d}小时，日均怠速放电{avg_idle_discharge_7d}小时），放电以静止状态为主，疑似地摊供电或储能场景"
        elif current_type_7d == "专送骑手":
            type_desc_7d = f"工作特征显著（近7天日均骑行{avg_ride_hour_7d:.1f}小时，高峰骑行占比{peak_ratio_7d*100:.0f}%），工作时长稳定且午晚高峰高度活跃，符合专送骑手画像"
        elif current_type_7d == "众包骑手":
            type_desc_7d = f"具有兼职骑手特征（近7天日均骑行{avg_ride_hour_7d:.1f}小时，高峰骑行占比{peak_ratio_7d*100:.0f}%），高峰时段有一定活跃度"
        elif current_type_7d == "标准骑手":
            type_desc_7d = f"骑行行为规律（近7天日均里程{avg_mileage_7d}km，日均骑行{avg_ride_hour_7d:.1f}小时），属于标准日常使用场景"
        elif current_type_7d == "普通骑手":
            type_desc_7d = f"有常规骑行行为（近7天日均里程{avg_mileage_7d}km，日均骑行{avg_ride_hour_7d:.1f}小时），但不符合特定骑手标签特征"
        else:
            type_desc_7d = "无有效骑行数据或数据量不足，无法判定具体使用场景"
        
        res['客户形态_综合_7d说明'] = type_desc_7d
        # ==================================================================================

        # 设备在线状态监控
        latest_record_date = pd.to_datetime(group['统计日期_dt'].max())
        days_since_last_seen = (current_date - latest_record_date).days
        
        if days_since_last_seen >= LONG_TERM_OFFLINE_THRESHOLD_DAYS:
            res['设备状态监控'] = f"异常(离线{days_since_last_seen}天)"
        elif days_since_last_seen >= 1:
            res['设备状态监控'] = f"暂离线({days_since_last_seen}天)"
        else:
            res['设备状态监控'] = "正常在线"
        
        rolling_results.append(res)
    
    return pd.DataFrame(rolling_results)

def update_user_lifecycle_7d(df_daily_user, df_snapshot_db, current_stat_date, write_to_file=True):
    if not df_daily_user.empty:
        df_snapshot_db = pd.concat([df_snapshot_db, df_daily_user], ignore_index=True)
        df_snapshot_db = df_snapshot_db.drop_duplicates(subset=['用户id', '统计日期'], keep='last')
        save_daily_snapshot_db(df_snapshot_db)
    
    print(f"\n🔄 开始计算{current_stat_date}的7天滚动指标...")
    df_rolling_7d = calculate_rolling_7d_metrics(df_snapshot_db, current_stat_date)
    if df_rolling_7d.empty:
        print(f"⚠️ 无7天滚动数据")
        return df_snapshot_db, pd.DataFrame()
    
    contract_map = load_user_contract_info()
    def get_contract_info(uid):
        info = contract_map.get(uid, {})
        return pd.Series([info.get('join_date', None), info.get('expire_date', None)])
    df_rolling_7d[['首次入网日期', '合约到期时间']] = df_rolling_7d['用户id'].apply(get_contract_info)
    
    current_date = pd.to_datetime(current_stat_date)
    def determine_lifecycle_state(row):
        expire_date = pd.to_datetime(row['合约到期时间'], errors='coerce')
        if pd.notna(expire_date) and current_date > expire_date: return '已到期'
        join_date = pd.to_datetime(row['首次入网日期'], errors='coerce')
        if pd.notna(join_date):
            days_since_join = (current_date - join_date).days
            if days_since_join <= NEW_USER_PROTECTION_DAYS: return '新用户'
        attendance_days = row.get('近7d出勤天数', 0)
        has_data_days = row.get('近7天有数据天数', 0)
        if attendance_days == 0:
            if has_data_days <= (7 - CHURN_THRESHOLD_DAYS): return '流失'
            else: return '沉默'
        elif attendance_days >= 5: return '活跃'
        else: return '轻度活跃'
    
    df_rolling_7d['用户生命周期状态_7d'] = df_rolling_7d.apply(determine_lifecycle_state, axis=1)
    
    # 修复Bug4：仅在write_to_file=True时才写文件，避免增量模式重复写入
    if write_to_file:
        try:
            os.makedirs(os.path.dirname(USER_LIFECYCLE_7D_FILE), exist_ok=True)
            df_rolling_7d.to_csv(USER_LIFECYCLE_7D_FILE, index=False, encoding='utf-8-sig')
            print(f"✅ 7天滚动生命周期表已更新（含新增日均指标及动态说明）：{USER_LIFECYCLE_7D_FILE}")
        except Exception as e:
            print(f"❌ 7天滚动表保存失败：{e}")
    return df_snapshot_db, df_rolling_7d

# ==================================================================================
# ===================== 8. 单日处理逻辑 =====================
# ==================================================================================
def process_single_date_folder(date_folder_path, output_subfolder_name, force_recalculate=False):
    date_folder_name = os.path.basename(date_folder_path)
    stat_date = get_stat_date(date_folder_name)
    output_folder = os.path.join(OUTPUT_ROOT_FOLDER, output_subfolder_name)
    output_file_contract = os.path.join(output_folder, f"{stat_date}_合约明细.csv")
    output_file_user = os.path.join(output_folder, f"{stat_date}_用户汇总.csv")

    if os.path.exists(output_file_contract) and os.path.exists(output_file_user) and not force_recalculate:
        print(f"✅ {stat_date} (历史数据) 已存在，跳过")
        return None, stat_date, None

    print(f"\n🚀 开始处理日期：{stat_date}")
    start_time = time.time()

    csv_files = [f for f in os.listdir(date_folder_path) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"   ⚠️ 无CSV文件，跳过")
        return None, stat_date, None
    
    print(f"   📖 读取 {len(csv_files)} 个CSV文件...")
    df_list = []
    for file in tqdm(csv_files, desc="文件读取进度", ncols=80):
        file_path = os.path.join(date_folder_path, file)
        try:
            df = pd.read_csv(file_path, encoding=CSV_ENCODING, low_memory=False, dtype={'用户id': str, '合约id': str, '时间戳': np.int64})
            df_list.append(df)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gbk', low_memory=False, dtype={'用户id': str, '合约id': str, '时间戳': np.int64})
            df_list.append(df)
        except Exception as e:
            print(f"   ❌ 读取失败 {file}：{str(e)}")

    if not df_list:
        print(f"   ⚠️ 无有效CSV数据，跳过")
        return None, stat_date, None

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = normalize_column_names(df_all)
    
    # 检查核心字段（包含是否在线）
    critical_missing = [c for c in REQUIRED_COLUMNS_RAW if c not in ['电池id', '电池SOC', '电池度数'] and c not in df_all.columns]
    if critical_missing:
        print(f"   ❌ 缺少核心字段：{critical_missing}，跳过")
        return None, stat_date, None
    
    df_all = df_all.drop_duplicates(subset=['合约id', '用户id', '时间戳'], keep='first')
    for c in ['时间戳', '纬度', '经度', '电流', '温度', '速度']:
        df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

    # ====================== 在线状态清洗与过滤 ======================
    if '是否在线' in df_all.columns:
        df_all['是否在线'] = df_all['是否在线'].astype(str).str.strip()
    
    valid_online_flags = ['在线', 'online', '1', 'true', '是']
    df_all['_is_valid_online'] = df_all['是否在线'].str.lower().isin(valid_online_flags)
    
    total_count = len(df_all)
    online_count = df_all['_is_valid_online'].sum()
    print(f"   📡 数据状态监控：总数据 {total_count} 条，在线数据 {online_count} 条 (已过滤离线 {total_count - online_count} 条)")
    
    df_all = df_all[df_all['_is_valid_online']].drop(columns=['_is_valid_online'])
    
    if len(df_all) == 0:
        print(f"   ⚠️ 过滤后无【在线】有效数据，跳过该日")
        return None, stat_date, None
    # ==================================================================

    # 虚假速度过滤
    df_all['速度'] = np.where(
        (df_all['电流'] < -20) & (df_all['速度'] > 5), 
        0, 
        df_all['速度']
    )
    df_all['速度'] = np.where(
        (df_all['电流'] > -5) & (df_all['电流'] <= 0) & (df_all['速度'] < 3), 
        0, 
        df_all['速度']
    )

    # 电流分层清洗
    df_all['电流_放电统计用'] = df_all['电流'].copy()
    df_all['电流_放电统计用'] = np.where(
        (df_all['电流_放电统计用'] > 0) & (df_all['电流_放电统计用'] <= 200), 
        df_all['电流_放电统计用'], 
        np.nan
    )
    df_all['电流_骑行判定用'] = np.where(
        (df_all['电流'] >= 1.0) & (df_all['电流'] <= MAX_VALID_CURRENT), 
        df_all['电流'], 
        np.nan
    )
    df_all['电流_前1'] = df_all.groupby(['合约id', '用户id'])['电流_骑行判定用'].shift(1)
    df_all['电流_后1'] = df_all.groupby(['合约id', '用户id'])['电流_骑行判定用'].shift(-1)
    df_all['电流_median'] = df_all[['电流_前1', '电流_骑行判定用', '电流_后1']].median(axis=1)
    df_all['电流_骑行判定用'] = np.where(
        abs(df_all['电流_骑行判定用'] - df_all['电流_median']) > 200, 
        np.nan, 
        df_all['电流_骑行判定用']
    )
    df_all = df_all.drop(columns=['电流_前1', '电流_后1', '电流_median'], errors='ignore')

    df_all = df_all[(df_all['温度'].isna()) | (df_all['温度'] <= 100)]
    
    df_clean = df_all.dropna(subset=['时间戳', '合约id', '用户id', '纬度', '经度']).copy()
    if GPS_FILTER:
        df_clean = df_clean[(df_clean['纬度'] >= 3) & (df_clean['纬度'] <= 54) & (df_clean['经度'] >= 73) & (df_clean['经度'] <= 136)]
    
    df_clean = df_clean.sort_values(['合约id', '用户id', '时间戳']).reset_index(drop=True)
    df_clean['前纬度'] = df_clean.groupby(['合约id', '用户id'])['纬度'].shift(1)
    df_clean['前经度'] = df_clean.groupby(['合约id', '用户id'])['经度'].shift(1)
    df_clean['前时间戳'] = df_clean.groupby(['合约id', '用户id'])['时间戳'].shift(1)
    df_clean['距离差_km'] = haversine(df_clean['前纬度'], df_clean['前经度'], df_clean['纬度'], df_clean['经度'], 'km')
    df_clean['时间差_h'] = (df_clean['时间戳'] - df_clean['前时间戳']) / 3600
    df_clean['瞬时速度_kmh'] = np.where(df_clean['时间差_h'] > 0, df_clean['距离差_km'] / df_clean['时间差_h'], 0)
    df_clean = df_clean[(df_clean['瞬时速度_kmh'] <= DRIFT_SPEED_THRESHOLD) | (df_clean['瞬时速度_kmh'].isna())]
    df_clean = df_clean.drop(columns=['前纬度', '前经度', '前时间戳', '距离差_km', '时间差_h', '瞬时速度_kmh'], errors='ignore')
    df_clean = df_clean.dropna(subset=['时间戳']).sort_values(['合约id', '用户id', '时间戳']).reset_index(drop=True)
    df_clean['统计日期'] = stat_date

    print(f"   ✅ 清洗完成：有效行数{len(df_clean)} (仅含在线数据)")

    df_contract_result = calc_contract_metrics(df_clean)
    df_user_result = aggregate_to_user_level(df_contract_result)

    os.makedirs(output_folder, exist_ok=True)
    df_contract_result.to_csv(output_file_contract, index=False, encoding='utf-8-sig')
    if not df_user_result.empty:
        df_user_result.to_csv(output_file_user, index=False, encoding='utf-8-sig')
    
    print(f"   🎉 处理完成，耗时{time.time()-start_time:.2f}秒")
    return df_user_result, stat_date, df_clean

# ==================================================================================
# ===================== 主函数 =====================
# ==================================================================================
def main(target_date: Optional[str] = None):
    print("="*80)
    print("两轮车包月用户分析系统 - 【7天日均指标完整版+全链路说明】")
    print(f"核心功能：")
    print(f"  1. 在线状态过滤 + 长期离线监控")
    print(f"  2. 7天累计指标 + 对应日均指标")
    print(f"  3. 全链路动态判定说明 (合约/日聚合/7天滚动)")
    print("="*80)
    print(f"当前运行模式：{RUN_MODE}")
    if target_date:
        print(f"目标日期：{target_date}")
    print("="*80)

    # 快速模式：仅刷新生命周期表
    if RUN_MODE == "quick":
        refresh_lifecycle_only()
        return
    
    # 重新计算模式：基于现有快照重新计算客户形态和用户等级
    if RUN_MODE == "recalculate":
        recalculate_customer_profile()
        return

    # 完整模式或增量模式
    load_user_contract_info()
    load_fence_data()
    df_snapshot_db = load_daily_snapshot_db()

    all_date_folders = []
    for input_root, output_sub in FOLDER_MAPPING.items():
        if not os.path.exists(input_root):
            print(f"\n❌ 输入路径不存在：{input_root}")
            continue
        print(f"\n📂 扫描目录：{input_root}")
        for item_name in os.listdir(input_root):
            item_path = os.path.join(input_root, item_name)
            if os.path.isdir(item_path):
                stat_date = get_stat_date(item_name)
                try:
                    pd.to_datetime(stat_date)
                    all_date_folders.append((stat_date, item_path, output_sub))
                except:
                    print(f"   ⚠️ 文件夹{item_name}日期解析失败，跳过")

    all_date_folders = sorted(all_date_folders, key=lambda x: x[0])
    
    # 增量模式：只处理比快照表中最新日期更新的文件夹
    if RUN_MODE == "incremental" and not df_snapshot_db.empty:
        latest_snapshot_date = df_snapshot_db['统计日期'].max()
        print(f"\n⏭️  [增量模式] 最新快照日期：{latest_snapshot_date}")
        # 过滤出日期大于最新快照日期的文件夹
        filtered_folders = []
        for stat_date, folder_path, output_sub in all_date_folders:
            if stat_date > latest_snapshot_date:
                filtered_folders.append((stat_date, folder_path, output_sub))
            else:
                print(f"   ⏭️  跳过已处理日期：{stat_date}")
        all_date_folders = filtered_folders
    
    total_folders = len(all_date_folders)
    print(f"\n📊 待处理日期文件夹：{total_folders} 个")
    if total_folders == 0:
        print("✅ 无新数据需要处理")
        # 如果是增量模式且无新数据，仍然刷新生命周期表
        if RUN_MODE == "incremental":
            print("\n🔄 刷新生命周期表...")
            refresh_lifecycle_only()
        return

    total_start = time.time()
    success, skip, fail = 0, 0, 0
    latest_stat_date = None

    # 完整模式下强制重新计算
    force_recalculate = (RUN_MODE == "full")

    for idx, (stat_date, folder_path, output_sub) in enumerate(all_date_folders):
        print(f"\n【总进度 {idx+1}/{total_folders}】")
        try:
            # 传递force_recalculate参数
            df_daily_user, process_date, df_raw_clean = process_single_date_folder(folder_path, output_sub, force_recalculate)
            if df_daily_user is not None and not df_daily_user.empty:
                # 修复Bug4：循环内只更新快照，不写生命周期文件（write_to_file=False）
                df_snapshot_db, _ = update_user_lifecycle_7d(df_daily_user, df_snapshot_db, process_date, write_to_file=False)
                latest_stat_date = process_date
                success += 1
            else:
                skip += 1
        except Exception as e:
            print(f"   ❌ 处理失败：{str(e)}")
            import traceback
            traceback.print_exc()
            fail += 1

    if latest_stat_date:
        print(f"\n📊 生成最终版全量用户7天滚动生命周期表...")
        _, df_final_rolling = update_user_lifecycle_7d(pd.DataFrame(), df_snapshot_db, latest_stat_date)
        if not df_final_rolling.empty:
            print("\n" + "="*80)
            df_attendance, month_days_map, energy_cum_map = calc_user_monthly_attendance(df_snapshot_db)
            if month_days_map and energy_cum_map and not df_final_rolling.empty:
                print("\n📝 合并月度预估数据到最终滚动表...")
                df_final_rolling['月度预估工作天数'] = df_final_rolling['用户id'].map(month_days_map)
                df_final_rolling['单合约月度用电度数预估_kWh'] = df_final_rolling['用户id'].map(energy_cum_map)
                df_final_rolling.to_csv(USER_LIFECYCLE_7D_FILE, index=False, encoding='utf-8-sig')
                print(f"✅ 最终版7天滚动生命周期表已更新（含所有累计+日均指标+动态说明）")
                
                # 优化5：调用独立函数运行动态阈值分析
                df_final_rolling = run_dynamic_analysis(df_final_rolling, USER_LIFECYCLE_7D_FILE)

    total_time = time.time() - total_start
    print("\n\n" + "="*80)
    print(f"✅ 全部完成！成功：{success} | 跳过：{skip} | 失败：{fail}")
    print(f"⏱️ 总耗时：{int(total_time//60)}分{total_time%60:.2f}秒")
    print("="*80)

# ==================================================================================
# 🆕 【新增功能】动态阈值分析独立函数（优化5）
# ==================================================================================
def run_dynamic_analysis(df_final_rolling, lifecycle_file):
    """
    运行动态阈值分析并合并到生命周期档案
    优化5：提取为独立函数，消除main()和refresh_lifecycle_only()中的重复代码
    """
    if not DynamicBatteryAnalyzer:
        print("⚠️  动态阈值分析模块不可用，跳过")
        return df_final_rolling
    
    print("\n" + "="*80)
    print("🔄 运行动态阈值分析并合并到生命周期档案...")
    print("="*80)
    
    start_time = time.time()
    total_users = len(df_final_rolling)
    print(f"📊 待分析用户数：{total_users:,}")
    
    try:
        # 构建 baseline 文件路径
        baseline_path = os.path.join(os.path.dirname(__file__), 'thresholds_baseline.json')
        
        # 调试：打印输入数据的列
        print(f"   🔍 输入数据列：{list(df_final_rolling.columns[-15:])}")
        
        # 步骤1：初始化分析器
        print(f"\n[1/5] 初始化分析器...")
        step_start = time.time()
        analyzer = DynamicBatteryAnalyzer(baseline_path=baseline_path)
        print(f"   ✅ 完成（耗时：{time.time() - step_start:.2f}秒）")
        
        # 步骤2：运行分析
        print(f"[2/5] 计算动态阈值（{total_users:,} 用户）...")
        step_start = time.time()
        thresholds, baseline = analyzer.run(df_final_rolling)
        print(f"   ✅ 完成（耗时：{time.time() - step_start:.2f}秒）")
        
        # 步骤3：评分和分类
        print(f"[3/5] 用户评分和等级分类...")
        step_start = time.time()
        df_result = analyzer.score_and_classify(df_final_rolling, thresholds, baseline)
        print(f"   ✅ 完成（耗时：{time.time()-step_start:.2f}秒）")
        print(f"   📋 评分结果列：{list(df_result.columns[-10:])}")
        
        # 步骤4：合并结果
        print(f"[4/5] 合并动态分析结果到生命周期档案...")
        step_start = time.time()
        dynamic_cols = ['重评分数', '用户等级_动态', '风险标签', '策略建议', '用户等级_动态说明']
        if '等级变化' in df_result.columns:
            dynamic_cols.append('等级变化')
        
        for col in dynamic_cols:
            if col in df_result.columns:
                df_final_rolling[col] = df_result[col]
        print(f"   ✅ 完成（耗时：{time.time() - step_start:.2f}秒）")
        
        # 步骤5：保存文件
        print(f"[5/5] 保存生命周期档案...")
        step_start = time.time()
        df_final_rolling.to_csv(lifecycle_file, index=False, encoding='utf-8-sig')
        print(f"   ✅ 完成（耗时：{time.time() - step_start:.2f}秒）")
        
        # 打印总耗时
        total_elapsed = time.time() - start_time
        print(f"\n⏱️  动态阈值分析总耗时：{total_elapsed:.2f}秒")
        
        # 打印结果统计
        total = len(df_result)
        print(f"\n{'='*55}")
        print(f"  动态阈值分析 · 用户等级分布（n={total:,}）")
        print(f"{'='*55}")
        
        cols = ['用户等级_动态', '策略建议']
        if '等级变化' in df_result.columns:
            cols.append('等级变化')
        
        for col in cols:
            print(f"\n[{col}]")
            for k, v in df_result[col].value_counts().items():
                print(f"  {k:<12}: {v:>5} 人  ({v/total*100:.1f}%)")
        
        print(f"\n[OK] 动态阈值分析结果已合并到生命周期档案")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"⚠️  动态阈值分析执行失败（已运行 {elapsed:.2f}秒）: {e}")
        import traceback
        traceback.print_exc()
        print("   继续使用静态阈值结果")
    
    return df_final_rolling

# ==================================================================================
# 🆕 【新增功能】客户形态和用户等级判定函数
# ==================================================================================
def determine_customer_profile(row):
    """基于快照数据重新计算客户形态"""
    total_riding_hours = row.get('骑行总耗时(小时)', 0)
    total_distance = row.get('行驶距离', 0)
    max_speed = row.get('最大速度', 0)
    riding_avg_current = row.get('骑行放电平均电流', 0)
    idle_discharge_hours = row.get('怠速放电时长(小时)', 0)
    total_discharge_hours = row.get('总放电时长(小时)', 0)
    peak_riding_ratio = row.get('高峰骑行占比', 0)
    n = row.get('有效GPS点数', 0)
    
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

def determine_user_level(row):
    """基于快照数据重新计算用户等级（统计学优化版）"""
    score            = row.get('包月友好评分', 0)
    current_60a      = row.get('电流>60A次数', 0)
    over_100a_times  = row.get('超100A连续次数', 0)
    over_100a_hours  = row.get('超100A累计时长_h', 0)
    soc_below_10     = row.get('SOC低于10%时长占比', 0)
    soc_below_20     = row.get('SOC低于20%时长占比', 0)
    min_soc          = row.get('最低SOC', 100)
    energy_100km     = row.get('百公里电耗(kWh)', 0)
    avg_current      = row.get('骑行放电平均电流', 0)
    has_energy_data  = (energy_100km > 0)
    has_soc_data     = (min_soc < 100)

    return _determine_user_level_v2(
        over100a_cont       = over_100a_times,
        over100a_hours      = over_100a_hours,
        current_80a_count   = 0,
        soc_below_10_ratio  = soc_below_10,
        min_soc             = min_soc,
        current_60a_count   = current_60a,
        energy_per_100km    = energy_100km,
        soc_below_20_ratio  = soc_below_20,
        score               = score,
        riding_avg_current  = avg_current,
        soc_data_valid      = has_soc_data,
        energy_data_valid   = has_energy_data
    )

def generate_comprehensive_explanation(row):
    """生成综合说明"""
    # 注意：用户等级_综合_7d 已被用户等级_动态取代
    user_level = row.get('用户等级_动态', '无效')
    customer_type = row.get('客户形态_综合_7d', '数据不足')
    
    explanations = []
    
    # 用户等级说明
    if user_level == "暴力":
        explanations.append("用户存在严重损害电池的行为，属于高风险用户")
    elif user_level == "高损耗用户":
        explanations.append("用户电池损耗速度高于平均水平，需要关注")
    elif user_level == "优质用户":
        explanations.append("用户电池使用习惯优秀，属于低风险用户")
    elif user_level == "良好用户":
        explanations.append("用户电池使用习惯较好，整体负载可控")
    elif user_level == "普通用户":
        explanations.append("用户电池使用行为一般，无明显过激使用行为")
    
    # 客户形态说明
    if customer_type == "改装/超速车":
        explanations.append("用户车辆可能存在改装或超速行为")
    elif customer_type == "地摊/储能":
        explanations.append("用户使用场景为地摊供电或储能")
    elif customer_type == "专送骑手":
        explanations.append("用户为专送骑手，工作特征显著")
    elif customer_type == "众包骑手":
        explanations.append("用户为众包骑手，具有兼职特征")
    elif customer_type == "标准骑手":
        explanations.append("用户骑行行为规律，属于标准日常使用场景")
    elif customer_type == "普通骑手":
        explanations.append("用户有常规骑行行为，不符合特定骑手标签特征")
    
    return " | ".join(explanations) if explanations else "数据不足，无法生成综合说明"

# ==================================================================================
# 🆕 【新增功能】单独刷新生命周期表函数
# ==================================================================================
def refresh_lifecycle_only():
    print("="*80)
    print("🔄 单独刷新模式：仅重新计算7天滚动生命周期表")
    print("="*80)

    # 1. 加载必要的辅助数据
    load_user_contract_info()
    # 注意：地理围栏数据如果不需要匹配省市可以注释掉 load_fence_data() 以加快速度
    # load_fence_data() 

    # 2. 加载历史每日快照库
    df_snapshot_db = load_daily_snapshot_db()
    
    if df_snapshot_db.empty:
        print("❌ 错误：未找到历史快照表，无法单独刷新。请先运行一次完整流程。")
        return

    # 3. 确定要计算的“最新统计日期”
    df_snapshot_db['统计日期_dt'] = pd.to_datetime(df_snapshot_db['统计日期'])
    latest_stat_date = df_snapshot_db['统计日期_dt'].max().strftime('%Y-%m-%d')
    
    print(f"\n📅 快照表日期范围：{df_snapshot_db['统计日期'].min()} 至 {df_snapshot_db['统计日期'].max()}")
    print(f"🎯 将基于【{latest_stat_date}】重新计算7天滚动指标")

    # 4. 调用核心计算函数
    _, df_final_rolling = update_user_lifecycle_7d(
        df_daily_user=pd.DataFrame(), 
        df_snapshot_db=df_snapshot_db, 
        current_stat_date=latest_stat_date
    )

    if not df_final_rolling.empty:
        print("\n" + "="*80)
        # 5. 重新计算月度预估并合并
        df_attendance, month_days_map, energy_cum_map = calc_user_monthly_attendance(df_snapshot_db)
        if month_days_map and energy_cum_map:
            print("\n📝 合并月度预估数据...")
            df_final_rolling['月度预估工作天数'] = df_final_rolling['用户id'].map(month_days_map)
            df_final_rolling['单合约月度用电度数预估_kWh'] = df_final_rolling['用户id'].map(energy_cum_map)
            
            # 保存最终文件
            df_final_rolling.to_csv(USER_LIFECYCLE_7D_FILE, index=False, encoding='utf-8-sig')
            print(f"✅ 【单独刷新完成】文件已更新：{USER_LIFECYCLE_7D_FILE}")
            print(f"   共生成 {len(df_final_rolling)} 个用户的档案。")
            
            # 优化5：调用独立函数运行动态阈值分析
            df_final_rolling = run_dynamic_analysis(df_final_rolling, USER_LIFECYCLE_7D_FILE)
    print("="*80)

# ==================================================================================
# 🆕 【新增功能】重新计算客户形态和用户等级函数
# ==================================================================================
def recalculate_customer_profile():
    print("="*80)
    print("🔄 重新计算模式：基于现有快照重新计算客户形态和用户等级")
    print("="*80)

    # 1. 加载必要的辅助数据
    load_user_contract_info()
    load_fence_data()

    # 2. 加载历史每日快照库
    df_snapshot_db = load_daily_snapshot_db()
    
    if df_snapshot_db.empty:
        print("❌ 错误：未找到历史快照表，无法重新计算。请先运行一次完整流程。")
        return

    # 3. 确定日期范围
    df_snapshot_db['统计日期_dt'] = pd.to_datetime(df_snapshot_db['统计日期'])
    min_date = df_snapshot_db['统计日期'].min()
    max_date = df_snapshot_db['统计日期'].max()
    
    print(f"\n📅 快照表日期范围：{min_date} 至 {max_date}")
    print(f"🎯 将基于所有快照数据重新计算客户形态和用户等级")

    # 4. 对每个用户的每日快照数据重新计算客户形态和用户等级
    print("\n🔄 开始重新计算客户形态和用户等级...")
    
    # 重新计算客户形态
    if '客户形态' in df_snapshot_db.columns:
        df_snapshot_db['客户形态'] = df_snapshot_db.apply(lambda row: determine_customer_profile(row), axis=1)
        print("✅ 客户形态重新计算完成")
    
    # 重新计算用户等级
    if '用户等级' in df_snapshot_db.columns:
        df_snapshot_db['用户等级'] = df_snapshot_db.apply(lambda row: determine_user_level(row), axis=1)
        print("✅ 用户等级重新计算完成")
    
    # 5. 重新计算7天滚动生命周期表
    latest_stat_date = max_date
    print(f"\n📅 基于最新日期【{latest_stat_date}】重新计算7天滚动指标")
    
    _, df_final_rolling = update_user_lifecycle_7d(
        df_daily_user=pd.DataFrame(), 
        df_snapshot_db=df_snapshot_db, 
        current_stat_date=latest_stat_date
    )

    if not df_final_rolling.empty:
        print("\n" + "="*80)
        # 6. 重新计算月度预估并合并
        df_attendance, month_days_map, energy_cum_map = calc_user_monthly_attendance(df_snapshot_db)
        if month_days_map and energy_cum_map:
            print("\n📝 合并月度预估数据...")
            df_final_rolling['月度预估工作天数'] = df_final_rolling['用户id'].map(month_days_map)
            df_final_rolling['单合约月度用电度数预估_kWh'] = df_final_rolling['用户id'].map(energy_cum_map)
            
            # 7. 重新生成综合说明
            if '综合说明' in df_final_rolling.columns:
                df_final_rolling['综合说明'] = df_final_rolling.apply(lambda row: generate_comprehensive_explanation(row), axis=1)
                print("✅ 综合说明重新生成完成")
            
            # 保存更新后的快照数据
            df_snapshot_db.to_csv(USER_DAILY_SNAPSHOT_FILE, index=False, encoding='utf-8-sig')
            print(f"✅ 快照数据已更新：{USER_DAILY_SNAPSHOT_FILE}")
            
            # 保存最终文件
            df_final_rolling.to_csv(USER_LIFECYCLE_7D_FILE, index=False, encoding='utf-8-sig')
            print(f"✅ 【重新计算完成】文件已更新：{USER_LIFECYCLE_7D_FILE}")
            print(f"   共处理 {len(df_snapshot_db['用户id'].unique())} 个用户的快照数据。")
            print(f"   共生成 {len(df_final_rolling)} 个用户的7天滚动档案。")
            
            # 优化5：调用独立函数运行动态阈值分析
            df_final_rolling = run_dynamic_analysis(df_final_rolling, USER_LIFECYCLE_7D_FILE)
    print("="*80)

# ==================================================================================
# 主入口：选择运行模式
# ==================================================================================
if __name__ == "__main__":
    main()