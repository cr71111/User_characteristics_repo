# -*- coding: utf-8 -*-
"""
Score Layer - 业务逻辑推断层
基于Fact层物理指标，应用业务规则和推断逻辑生成用户等级和策略建议。

核心职责：
✓ 用户评分（包月友好分/重评分数）
✓ 用户等级分类（优质/良好/普通/高损耗/暴力/观察期/沉默）
✓ 风险标签生成（超保护板电流/疑似静态储能等）
✓ 策略建议生成

输入：Fact层7天滚动事实表 + 动态阈值
输出：用户评分表（包含用户等级、风险标签、策略建议等）
"""

import os
import sys
import time
import warnings
from typing import Dict, Optional

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入动态阈值分析模块
try:
    from src.pipeline.dynamic_thresholds import (
        DynamicBatteryAnalyzer,
        BaselineStore,
        DistributionShiftDetector,
        AdaptiveThresholds
    )
except ImportError as e:
    print(f"⚠️  动态阈值模块导入失败: {e}")
    DynamicBatteryAnalyzer = None
    BaselineStore = None
    DistributionShiftDetector = None
    AdaptiveThresholds = None

# ==============================================================================
# 全局配置参数
# ==============================================================================
PROTECTION_BOARD_MAX = 60.0  # 保护板最大电流
MONTHLY_ENERGY_LOSS_THRESHOLD = 150.0  # 月用电盈亏线（度）
MONTHLY_ENERGY_VIOLENT_THRESHOLD = 250.0  # 暴力用户用电阈值（度）
NEW_USER_PROTECTION_DAYS = 3  # 新兵保护期天数

# 温度阈值（与 fact_layer 对齐）
TEMP_HIGH_THRESHOLD = 55.0
TEMP_EXTREME_THRESHOLD = 70.0

# 速度阈值
SPEED_HIGH_THRESHOLD = 50.0
SPEED_EXTREME_THRESHOLD = 60.0
SPEED_VIOLENT_THRESHOLD = 80.0

# SOC 黄金区间
SOC_OPTIMAL_LOWER = 30
SOC_OPTIMAL_UPPER = 80
SOC_BONUS_SCORE = 3.0

# 换电阈值
NORMAL_DAILY_SWAPS = 3
HIGH_DAILY_SWAPS = 5

# 60A 电流告警次数门槛
OVER60A_WARNING_COUNT = 10

# 超100A 累计时长门槛（7天）
OVER100A_HOURS_THRESHOLD_7D = 0.5


def score_and_classify(
    df: pd.DataFrame,
    live_quantiles: Dict,
    active_baseline: Dict,
) -> pd.DataFrame:
    """
    基于动态阈值对 DataFrame 进行评分和用户等级分类
    【优化版】使用向量化操作替代apply(axis=1)，性能提升10-100倍
    """
    bq = active_baseline.get('quantiles', live_quantiles)
    df = df.copy()

    avg_cur_P90  = bq.get('avg_current',   {}).get('P90',  28.3)
    avg_cur_P95  = bq.get('avg_current',   {}).get('P95',  32.1)
    max_cur_P90  = bq.get('max_current',   {}).get('P90',  64.7)
    max_cur_P95  = bq.get('max_current',   {}).get('P95',  68.9)
    max_cur_P99  = bq.get('max_current',   {}).get('P99',  81.9)
    energy_P90   = bq.get('energy_per_100km', {}).get('P90', 13.8)

    # ── A. 风险标签（向量化优化）────────────────────────────────────────
    max_cur = df['近7d最大电流_A'].fillna(0).values
    avg_cur = df['近7d平均骑行电流_A'].fillna(0).values
    over100 = df['近7d超100A连续总次数'].fillna(0).values
    over100_hours = df['近7d超100A累计时长_h'].fillna(0).values
    over80 = df['近7d电流超80A总次数'].fillna(0).values
    over60 = df['近7d电流超60A总次数'].fillna(0).values
    min_soc = df['近7d最低SOC'].fillna(100).values
    soc_low_ratio = df['近7d_SOC低于20%时长占比'].fillna(0).values
    soc_below_10_ratio = df['近7d_SOC低于10%时长占比'].fillna(0).values
    monthly_energy = df['单合约月度用电度数预估_kWh'].fillna(0).values
    max_temp = df['近7d最高温度_℃'].fillna(0).values
    max_speed = df['近7d最高速度_kmh'].fillna(0).values
    
    cond_violent = (over100 >= 2) | (over100_hours >= OVER100A_HOURS_THRESHOLD_7D)
    cond_extreme = (max_cur > max_cur_P99) & ~cond_violent
    cond_protect = (max_cur > PROTECTION_BOARD_MAX) & ~cond_violent & ~cond_extreme
    cond_over80 = (over80 > 0) & ~cond_violent & ~cond_extreme & ~cond_protect
    cond_over60 = (over60 > OVER60A_WARNING_COUNT) & ~cond_violent & ~cond_extreme & ~cond_protect & ~cond_over80
    cond_high_avg = (avg_cur > avg_cur_P95) & ~cond_violent & ~cond_extreme & ~cond_protect & ~cond_over80 & ~cond_over60
    
    risk_tags = np.select(
        [cond_violent, cond_extreme, cond_protect, cond_over80, cond_over60, cond_high_avg],
        ['暴力放电', '极端电流', '超保护板电流', '频繁超80A', '中高电流频繁', '持续高耗流'],
        default=''
    )
    
    soc_risk = np.select(
        [soc_below_10_ratio > 0.05, min_soc < 10, soc_below_10_ratio > 0, soc_low_ratio > 0],
        ['持续深度亏电', '深度亏电', '深度亏电', '低SOC告警'],
        default=''
    )
    
    energy_risk = np.where(monthly_energy > MONTHLY_ENERGY_LOSS_THRESHOLD, 
                          '月用电超标(' + (monthly_energy).astype(int).astype(str) + '度)', '')
    
    temp_risk = np.select(
        [max_temp >= TEMP_EXTREME_THRESHOLD, max_temp >= TEMP_HIGH_THRESHOLD],
        ['电池高温告警(>70°C)', '电池温度偏高(>55°C)'],
        default=''
    )
    
    df['风险标签'] = pd.Series([
        ' / '.join(filter(None, [r, s, e, t])) if any([r, s, e, t]) else '正常'
        for r, s, e, t in zip(risk_tags, soc_risk, energy_risk, temp_risk)
    ], index=df.index)
    
    # ── A2. 新增：识别"非移动用电"（疑似静态储能）────────────────────────
    idle_hours = df['近7d单合约日均怠速放电_h'].fillna(0).values
    discharge_hours = df['近7d单合约日均放电时长_h'].fillna(0).values
    max_radius = df['近7d最大活动半径_km'].fillna(999).values
    
    static_discharge_ratio = np.where(discharge_hours > 0, idle_hours / discharge_hours, 0.0)
    is_static_storage = (static_discharge_ratio > 0.8) & (max_radius < 2.0) & (discharge_hours > 1.0)
    
    df['风险标签'] = np.where(
        is_static_storage,
        df['风险标签'].apply(lambda x: f'{x} / 疑似静态储能' if x != '正常' else '疑似静态储能'),
        df['风险标签']
    )

    # ── B. 包月友好评分（向量化优化）────────────────────────────────────
    # 【修复】删除旧的monthly_score分位数读取逻辑，改为基于重评分数动态计算
    # score_P_excellent = bq.get('monthly_score', {}).get('P85', 85)  # 已删除
    # score_P_good      = bq.get('monthly_score', {}).get('P65', 65)  # 已删除
    # score_P_normal    = bq.get('monthly_score', {}).get('P45', 45)  # 已删除

    max_cur = df['近7d最大电流_A'].fillna(0).values
    avg_cur = df['近7d平均骑行电流_A'].fillna(0).values
    energy  = df['近7d百公里电耗_kWh'].fillna(0).values
    over80  = df['近7d电流超80A总次数'].fillna(0).values
    over100 = df['近7d超100A连续总次数'].fillna(0).values
    soc_low = df['近7d_SOC低于20%时长占比'].fillna(0).values
    monthly = df['近7d平均包月友好分'].fillna(0).values
    max_temp_score = df['近7d最高温度_℃'].fillna(0).values
    max_speed_score = df['近7d最高速度_kmh'].fillna(0).values
    total_swaps = df['近7d总换电次数'].fillna(0).values
    avg_soc = df['近7d平均骑行SOC'].fillna(0).values
    
    ride_hours = df['近7d单合约日均骑行时长_h'].fillna(0).values
    idle_hours = df['近7d单合约日均怠速放电_h'].fillna(0).values
    discharge_hours = df['近7d单合约日均放电时长_h'].fillna(0).values
    monthly_energy_val = df['单合约月度用电度数预估_kWh'].fillna(0).values
    customer_type = df.get('客户形态_综合_7d', pd.Series('未知', index=df.index)).fillna('未知').values
    work_pattern = df.get('近7d工作特点', pd.Series('', index=df.index)).fillna('').values

    # ── B0. 沉默用户判定 ─────────────────────────────────────────────────
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
            result = np.where(val <= p25, 18.0, result)
            result = np.where((val > p25) & (val <= p50), 10.0, result)
            result = np.where((val > p50) & (val <= p75), 4.0, result)
            ref = p95 if p95 else p75 * 1.5
            excess = np.minimum(1.0, (val - p75) / (ref - p75 + 0.01))
            result = np.where(val > p75, np.maximum(0.0, 4.0 - excess * 4.0), result)
        else:
            result = np.where(val >= p75, 18.0, result)
            result = np.where((val >= p50) & (val < p75), 10.0, result)
            result = np.where((val >= p25) & (val < p50), 4.0, result)
            result = np.where(val < p25, np.maximum(0.0, 2.0 * val / (p25 + 0.01)), result)
        return result

    dim_avg_cur   = dim_score_vectorized(avg_cur,  aP25,  aP50,  aP75,  aP95, invert=True)
    dim_max_cur   = dim_score_vectorized(max_cur,  mP25,  mP50,  mP75,  mP95, invert=True)
    dim_energy    = dim_score_vectorized(energy,   eP75,  eP90,  eP95,  eP95, invert=True)
    dim_soc_low   = dim_score_vectorized(soc_low,  sP25,  sP50,  sP75,  sP95, invert=True)
    dim_monthly   = dim_score_vectorized(monthly, mscP25, mscP50, mscP75, invert=False)

    over80_score = np.select(
        [over80 <= 3, (over80 > 3) & (over80 <= 6), (over80 > 6) & (over80 <= 10), over80 > 10],
        [20.0, 20.0 - (over80 - 3) * 2.0, 14.0 - (over80 - 6) * 3.0, 5.0 - (over80 - 10) * 5.0],
        default=20.0
    )
    over80_score = np.maximum(0.0, over80_score)
    
    total = dim_avg_cur + dim_max_cur + dim_energy + dim_soc_low + dim_monthly + over80_score
    
    idle_penalty = np.where(idle_hours > 5, -20.0, np.where(idle_hours > 3, -10.0, 0.0))
    total = total + idle_penalty
    
    # 温度扣分（与日级 _calc_monthly_score_v2 保持一致）
    temp_penalty = np.select(
        [max_temp_score >= TEMP_EXTREME_THRESHOLD, max_temp_score >= TEMP_HIGH_THRESHOLD],
        [-8.0, -4.0],
        default=0.0
    )
    total = total + temp_penalty
    
    # 速度扣分（与日级逻辑对齐）
    speed_penalty = np.select(
        [max_speed_score >= SPEED_VIOLENT_THRESHOLD, max_speed_score >= SPEED_EXTREME_THRESHOLD, max_speed_score >= SPEED_HIGH_THRESHOLD],
        [-10.0, -6.0, -3.0],
        default=0.0
    )
    total = total + speed_penalty
    
    # 换电次数扣分（参考日级逻辑：超3次/天开始扣分）
    daily_avg_swaps = total_swaps / 7.0
    swap_penalty = np.where(
        daily_avg_swaps > HIGH_DAILY_SWAPS, -12.0,
        np.where(daily_avg_swaps > NORMAL_DAILY_SWAPS, -6.0,
        np.where(daily_avg_swaps > 2, -3.0, 0.0))
    )
    total = total + swap_penalty
    
    # SOC 黄金区间奖励（优质 SOC 管理用户正向激励）
    soc_bonus = np.where(
        (avg_soc >= SOC_OPTIMAL_LOWER) & 
        (avg_soc <= SOC_OPTIMAL_UPPER) & 
        (soc_low == 0),
        SOC_BONUS_SCORE,
        0.0
    )
    total = total + soc_bonus
    
    total = np.where(over100 >= 2, total * 0.3, total)
    total = np.where(over100 == 1, total * 0.7, total)
    
    total = np.where(is_silent, 20.0, total)
    
    df['重评分数'] = np.clip(total, 0, 100).round(1)

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

    # ── C. 用户等级分类（向量化优化）────────────────────────────────────
    s = df['重评分数'].values
    risk = df['风险标签'].fillna('正常').values
    
    is_violent = (over100 >= 2) | (over100_hours >= OVER100A_HOURS_THRESHOLD_7D) | (max_cur > max_cur_P99) | (monthly_energy_val > MONTHLY_ENERGY_VIOLENT_THRESHOLD)
    
    has_violent_risk = np.array(['暴力放电' in r or '极端电流' in r for r in risk])
    has_high_risk = np.array(['超保护板电流' in r or '频繁超80A' in r for r in risk])
    has_medium_risk = np.array(['月用电超标' in r for r in risk])
    
    days_since_join = df.get('入网天数', pd.Series([999] * len(df))).fillna(999).values
    
    result_level = np.full(len(df), '', dtype=object)
    
    result_level[is_silent] = '沉默用户'
    
    result_level[is_violent & ~is_silent] = '暴力'
    
    high_risk_mask = has_high_risk & ~is_violent & ~is_silent
    result_level[high_risk_mask & (s >= score_P_normal)] = '普通用户'
    result_level[high_risk_mask & (s < score_P_normal)] = '高损耗用户'
    
    medium_risk_mask = has_medium_risk & ~is_violent & ~high_risk_mask & ~is_silent
    result_level[medium_risk_mask & (s >= score_P_excellent)] = '良好用户'
    result_level[medium_risk_mask & (s >= score_P_good) & (s < score_P_excellent)] = '良好用户'
    result_level[medium_risk_mask & (s >= score_P_normal) & (s < score_P_good)] = '普通用户'
    result_level[medium_risk_mask & (s < score_P_normal)] = '高损耗用户'
    
    # 修复P1: 新兵保护期提前执行，避免覆盖高耗电新用户
    new_user_mask = ~is_violent & ~is_silent & (days_since_join <= NEW_USER_PROTECTION_DAYS) & (result_level == '')
    result_level[new_user_mask] = '观察期'
    
    high_energy_mask = (monthly_energy_val > MONTHLY_ENERGY_LOSS_THRESHOLD) & (monthly_energy_val <= MONTHLY_ENERGY_VIOLENT_THRESHOLD) & ~is_violent & ~is_silent & (days_since_join > NEW_USER_PROTECTION_DAYS)
    result_level[high_energy_mask] = '高损耗用户'
    
    normal_mask = ~is_violent & ~high_risk_mask & ~medium_risk_mask & ~has_violent_risk & ~is_silent & (days_since_join > NEW_USER_PROTECTION_DAYS) & (monthly_energy_val <= MONTHLY_ENERGY_LOSS_THRESHOLD) & (result_level == '')
    result_level[normal_mask & (s >= score_P_excellent)] = '优质用户'
    result_level[normal_mask & (s >= score_P_good) & (s < score_P_excellent)] = '良好用户'
    result_level[normal_mask & (s >= score_P_normal) & (s < score_P_good)] = '普通用户'
    result_level[normal_mask & (s < score_P_normal)] = '高损耗用户'

    df['用户等级_动态'] = result_level

    # ── D. 策略建议生成（向量化优化 + 差异化）────────────────────────────
    def _get_strategy(level, customer_type, work_pattern):
        """根据用户等级 + 客户形态 + 工作特点生成差异化策略"""
        base_strategy = {
            '优质用户':'留存激励','良好用户':'维持服务',
            '普通用户':'引导升级','高损耗用户':'限制预警',
            '暴力':'清退处理','观察期':'新手引导','沉默用户':'激活唤醒'
        }.get(level, '维持服务')
        
        if customer_type == '专送骑手' and level in ('普通用户', '良好用户'):
            return '专送骑手关怀'
        if customer_type == '地摊/储能':
            return '非正常用电核查'
        if customer_type == '改装/超速车':
            return '风险用户核查'
        if customer_type == '众包骑手' and level == '普通用户':
            return '众包骑手引导'
        if work_pattern in ('习惯晚上',) and level in ('高损耗用户',):
            return '夜间高损耗预警'
        
        return base_strategy
    
    df['策略建议'] = [
        _get_strategy(lv, ct, wp)
        for lv, ct, wp in zip(
            df['用户等级_动态'].values,
            df.get('客户形态_综合_7d', pd.Series('未知', index=df.index)).fillna('未知').values,
            df.get('近7d工作特点', pd.Series('', index=df.index)).fillna('').values
        )
    ]

    # ── E. 等级说明生成（向量化优化）────────────────────────────────────
    levels = df['用户等级_动态'].values
    scores = df['重评分数'].values
    risks = df['风险标签'].values
    max_currents = df['近7d最大电流_A'].fillna(0).values
    avg_currents = df['近7d平均骑行电流_A'].fillna(0).values
    energies = df['近7d百公里电耗_kWh'].fillna(0).values
    min_socs = df['近7d最低SOC'].fillna(100).values
    monthly_energies = df['单合约月度用电度数预估_kWh'].fillna(0).values
    
    level_desc = np.select(
        [levels == '优质用户', levels == '良好用户', levels == '普通用户', 
         levels == '高损耗用户', levels == '暴力', levels == '观察期', levels == '沉默用户'],
        ['电池使用习惯优秀，低风险', '电池使用习惯较好，负载可控', '电池使用行为一般，需关注',
         '电池损耗高于平均，需重点关注', '存在严重损害电池行为，高风险', '新用户观察期，暂不参与评级', '用户活跃度极低，需关注'],
        default=''
    )
    
    core_metrics = np.array([
        f'评分{s:.0f}分 | 最大电流{mc:.0f}A/平均{ac:.0f}A | 百公里电耗{e:.1f}kWh | 最低SOC{ms:.0f}%'
        for s, mc, ac, e, ms in zip(scores, max_currents, avg_currents, energies, min_socs)
    ])
    
    risk_tips = np.where(risks != '正常', '风险：' + risks, '')
    
    energy_warnings = np.where(monthly_energies > 150, 
                              '⚠️月用电' + monthly_energies.astype(int).astype(str) + '度超标', '')
    
    df['用户等级_动态说明'] = pd.Series([
        '；'.join(filter(None, [ld, cm, rt, ew]))
        for ld, cm, rt, ew in zip(level_desc, core_metrics, risk_tips, energy_warnings)
    ], index=df.index)

    return df


# ==============================================================================
# 主入口函数
# ==============================================================================
def generate_score_layer(fact_7d_rolling_path, baseline_path=None, output_path=None):
    """
    Score层生成函数
    
    输入：7天滚动事实表路径
    输出：用户评分表（包含用户等级、风险标签、策略建议等）
    """
    print("="*80)
    print("Score层生成 - 业务逻辑推断层")
    print("="*80)
    
    start_time = time.time()
    
    print("\n[1/4] 加载Fact层数据...")
    if not os.path.exists(fact_7d_rolling_path):
        print(f"❌ Fact层数据文件不存在：{fact_7d_rolling_path}")
        return None
    
    df_fact = pd.read_csv(fact_7d_rolling_path)
    print(f"   ✅ 加载完成：{len(df_fact):,} 个用户记录")
    
    print("\n[2/4] 初始化动态阈值分析器...")
    if baseline_path is None:
        baseline_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'thresholds_baseline.json')
    
    if DynamicBatteryAnalyzer:
        analyzer = DynamicBatteryAnalyzer(
            baseline_path=baseline_path,
            use_ema_update=True,  # 启用EMA自动更新基准文件
            ema_alpha=0.3
        )
    else:
        print("❌ 动态阈值分析器不可用")
        return None
    
    print("\n[3/4] 计算动态阈值并评分分类...")
    thresholds, baseline = analyzer.run(df_fact)
    df_score = score_and_classify(df_fact, thresholds, baseline)
    print(f"   ✅ 评分分类完成")
    
    print("\n[4/4] 持久化存储...")
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'score')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'score_user_profile.csv')
    
    df_score.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 用户评分表：{output_path}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Score层生成完成，总耗时：{elapsed:.2f}秒")
    
    return df_score

def run_dynamic_analysis_on_fact(df_fact, baseline_path=None):
    """
    对Fact层数据运行动态分析（供主入口调用）
    
    输入：Fact层7天滚动DataFrame
    输出：包含评分和分类结果的DataFrame
    """
    if not DynamicBatteryAnalyzer:
        print("⚠️  动态阈值分析模块不可用，跳过")
        return df_fact
    
    print("\n" + "="*80)
    print("🔄 运行动态阈值分析并合并到生命周期档案...")
    print("="*80)
    
    start_time = time.time()
    total_users = len(df_fact)
    print(f"📊 待分析用户数：{total_users:,}")
    
    if baseline_path is None:
        baseline_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'thresholds_baseline.json')
    
    try:
        print(f"\n[1/3] 初始化分析器...")
        analyzer = DynamicBatteryAnalyzer(
            baseline_path=baseline_path,
            use_ema_update=True,  # 启用EMA自动更新基准文件
            ema_alpha=0.3
        )
        
        print(f"[2/3] 计算动态阈值（{total_users:,} 用户）...")
        thresholds, baseline = analyzer.run(df_fact)
        
        print(f"[3/3] 用户评分和等级分类...")
        df_result = score_and_classify(df_fact, thresholds, baseline)
        
        dynamic_cols = ['重评分数', '用户等级_动态', '风险标签', '策略建议', '用户等级_动态说明']
        if '等级变化' in df_result.columns:
            dynamic_cols.append('等级变化')
        
        for col in dynamic_cols:
            if col in df_result.columns:
                df_fact[col] = df_result[col]
        
        total_elapsed = time.time() - start_time
        print(f"\n⏱️  动态阈值分析总耗时：{total_elapsed:.2f}秒")
        
        return df_fact
        
    except Exception as e:
        print(f"❌ 动态阈值分析失败：{e}")
        import traceback
        traceback.print_exc()
        return df_fact
