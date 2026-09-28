# -*- coding: utf-8 -*-
"""
用户运营分析报告 - 自动生成脚本
===============================
读取 reports/ 下的数据文件，自动统计并生成 Markdown 格式的用户运营分析报告。

数据源：{DATA_ROOT}\reports
输出：  {DATA_ROOT}\用户分析报告\用户运营分析报告.md

用法：
    python generate_report.py                    # 使用默认路径
    python generate_report.py --input ./reports  # 指定输入目录
    python generate_report.py --output ./报告.md  # 指定输出文件

依赖：pandas, numpy
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# HTML / PDF 生成依赖
try:
    import markdown as _md
    _HTML_AVAILABLE = True
except ImportError:
    _HTML_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# ── 路径配置 ────────────────────────────────────────────────
DATA_ROOT = r'E:\OneDrive\DataBase\DataBase\用户行为习惯\用户特征画像'
DEFAULT_INPUT_DIR = os.path.join(DATA_ROOT, 'reports')
DEFAULT_OUTPUT_FILE = os.path.join(DATA_ROOT, '用户分析报告', '用户运营分析报告.md')

# ── 业务阈值常量（与 score_layer.py 对齐）────────────
MONTHLY_ENERGY_LOSS_THRESHOLD = 150.0  # 月用电盈亏线（度）


def load_data(input_dir: str) -> dict:
    """加载所有数据源，返回字典"""
    data = {}

    # user_detail.csv
    path = os.path.join(input_dir, 'user_detail.csv')
    if os.path.exists(path):
        data['detail'] = pd.read_csv(path)
        print(f"  ✅ user_detail.csv: {len(data['detail']):,} 行")
    else:
        print(f"  ⚠️  user_detail.csv 不存在: {path}")

    # user_7d_full.csv
    path = os.path.join(input_dir, 'user_7d_full.csv')
    if os.path.exists(path):
        data['full'] = pd.read_csv(path)
        print(f"  ✅ user_7d_full.csv: {len(data['full']):,} 行")
    else:
        print(f"  ⚠️  user_7d_full.csv 不存在: {path}")

    # level_distribution.csv
    path = os.path.join(input_dir, 'level_distribution.csv')
    if os.path.exists(path):
        data['level_dist'] = pd.read_csv(path)
        print(f"  ✅ level_distribution.csv")

    # risk_alerts.csv
    path = os.path.join(input_dir, 'risk_alerts.csv')
    if os.path.exists(path):
        data['risk_alerts'] = pd.read_csv(path)
        print(f"  ✅ risk_alerts.csv: {len(data['risk_alerts']):,} 行")
    else:
        print(f"  ⚠️  risk_alerts.csv 不存在: {path}")

    # shift_report.json
    path = os.path.join(input_dir, 'shift_report.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data['shift'] = json.load(f)
        print(f"  ✅ shift_report.json")

    # user_monthly_attendance_detail.csv
    path = os.path.join(input_dir, 'user_monthly_attendance_detail.csv')
    if os.path.exists(path):
        data['attendance'] = pd.read_csv(path)
        print(f"  ✅ user_monthly_attendance_detail.csv: {len(data['attendance']):,} 行")
    else:
        print(f"  ⚠️  user_monthly_attendance_detail.csv 不存在: {path}")

    # 在网用户-加代理.csv（合约代理信息：一级/二级/三级代理）
    agent_path = r'E:\OneDrive\DataBase\DataBase\合约信息\在网用户-加代理.csv'
    if os.path.exists(agent_path):
        data['agent'] = pd.read_csv(agent_path, usecols=['用户id', '一级', '二级', '三级'])
        print(f"  ✅ 在网用户-加代理.csv: {len(data['agent']):,} 行")
    else:
        print(f"  ⚠️  在网用户-加代理.csv 不存在: {agent_path}")

    # ── 仅保留合约期内用户（剔除已到期）──
    if data.get('detail') is not None and '用户生命周期状态_7d' in data['detail'].columns:
        before = len(data['detail'])
        active_mask = data['detail']['用户生命周期状态_7d'] != '已到期'
        data['detail'] = data['detail'][active_mask].reset_index(drop=True)
        # 合约期内用户 id 集合，用于过滤其他数据源
        active_ids = set(data['detail']['用户id'])
        for key in ['full', 'attendance', 'risk_alerts', 'agent']:
            if data.get(key) is not None and '用户id' in data[key].columns:
                data[key] = data[key][data[key]['用户id'].isin(active_ids)].reset_index(drop=True)
        expired_n = before - len(data['detail'])
        print(f"  🔄 已剔除已到期用户 {expired_n:,} 人，合约期内剩 {len(data['detail']):,} 人")

    return data


# ═════════════════════════════════════════════════════════════
#  统计函数
# ═════════════════════════════════════════════════════════════

def fmt_pct(numerator, denominator):
    """计算占比"""
    if denominator == 0:
        return 0
    return numerator / denominator * 100


def safe_mean(series):
    """安全均值"""
    s = series.dropna()
    return s.mean() if len(s) > 0 else 0


def safe_median(series):
    """安全中位数"""
    s = series.dropna()
    return s.median() if len(s) > 0 else 0


def safe_quantile(series, q):
    """安全分位数"""
    s = series.dropna()
    return s.quantile(q) if len(s) > 0 else 0


# ═════════════════════════════════════════════════════════════
#  报告生成
# ═════════════════════════════════════════════════════════════

def generate_report(data: dict) -> str:
    """生成完整报告"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []

    def w(line=''):
        lines.append(line)

    # ── 获取核心 DataFrame ──
    df = data.get('detail')
    df_full = data.get('full')
    df_att = data.get('attendance')
    df_agent = data.get('agent')
    shift = data.get('shift', {})

    if df is None:
        return "# 错误：缺少 user_detail.csv 数据源"

    # ── 合并代理信息到 df_full，并仅保留合资公司+自营用户（贯穿总体报告与代理对比）──
    if df_full is not None and df_agent is not None:
        df_full = df_full.merge(df_agent, on='用户id', how='left')
        # 仅保留一级为「合资公司」「自营」的用户，同步过滤 df / df_att
        _mask = df_full['一级'].isin(['合资公司', '自营']) if '一级' in df_full.columns else pd.Series(True, index=df_full.index)
        _agent_ids = set(df_full.loc[_mask, '用户id'])
        df_full = df_full[_mask].reset_index(drop=True)
        df = df[df['用户id'].isin(_agent_ids)].reset_index(drop=True)
        if df_att is not None and '用户id' in df_att.columns:
            df_att = df_att[df_att['用户id'].isin(_agent_ids)].reset_index(drop=True)
        print(f"  🔄 总体报告仅保留合资公司+自营用户，共 {len(df):,} 人")

    total_users = len(df)
    w(f'# 《两轮车换电平台用户运营分析报告》')
    w()
    w(f'> **自动生成时间**：{now}')
    w(f'> **数据来源**：`docs/reports/` 目录下的 CSV/JSON 数据文件')
    w(f'> **总用户数**：{total_users:,} 人')
    w()

    # ═══════════════ 第一章：管理层摘要 ═══════════════
    w('---')
    w()
    w('## 一、管理层摘要')
    w()
    w('### 1.1 核心指标看板')
    w()

    # ── 加权7天出勤估算 ──
    # 原理：出勤率 = 出勤天数 / 有效数据天数，用户只会在这几天内有活动。
    # 若有效数据不满7天，按出勤率将出勤天数推算到满7天。
    # 例：用户只有4天有效数据，出勤4天(100%) → 加权推算7天出勤 = 100% × 7 = 7 天
    # 例：用户只有4天有效数据，出勤2天(50%) → 加权推算7天出勤 = 50% × 7 = 3.5 天
    if df_full is not None and '近7d出勤率' in df_full.columns and '近7天有数据天数' in df_full.columns:
        valid_days = df_full['近7天有数据天数'].fillna(0)
        raw_rate = df_full['近7d出勤率'].fillna(0)
        w7d_attend = (raw_rate * 7).clip(upper=7)  # 加权7天出勤天数
        w7d_attend[valid_days == 0] = 0  # 无有效数据的归零
        avg_valid_days = safe_mean(valid_days)
        avg_w7d_attend = safe_mean(w7d_attend)
        w7d_med = safe_median(w7d_attend)
        # 加权分类（不含新用户/已到期的特殊判断，纯基于出勤行为）
        w7d_active = int((w7d_attend >= 5).sum())
        w7d_mild = int(((w7d_attend >= 1) & (w7d_attend < 5)).sum())
        w7d_silent = int(((w7d_attend == 0) & (valid_days > 0)).sum())
        w7d_churn = int((valid_days == 0).sum())
        # 加权生命周期状态：保留 已到期/新用户（合同和入网日期决定），其余按加权出勤重新分类
        if '用户生命周期状态_7d' in df_full.columns:
            w7d_lc = df_full['用户生命周期状态_7d'].copy()
        else:
            w7d_lc = pd.Series('轻度活跃', index=df_full.index)
        # 对齐到 df_full 的索引
        w7d_attend_aligned = pd.Series(w7d_attend.values, index=df_full.index)
        valid_aligned = pd.Series(valid_days.values, index=df_full.index)
        is_overridable = ~w7d_lc.isin(['已到期', '新用户'])  # 合同和入网信息不动
        w7d_lc.loc[is_overridable & (valid_aligned == 0)] = '流失'
        w7d_lc.loc[is_overridable & (valid_aligned > 0) & (w7d_attend_aligned == 0)] = '沉默'
        w7d_lc.loc[is_overridable & (w7d_attend_aligned >= 5)] = '活跃'
        w7d_lc.loc[is_overridable & (w7d_attend_aligned >= 1) & (w7d_attend_aligned < 5)] = '轻度活跃'
    else:
        avg_valid_days = safe_mean(df.get('近7d出勤天数', pd.Series()))
        avg_w7d_attend = avg_valid_days
        w7d_med = safe_median(df.get('近7d出勤天数', pd.Series()))
        w7d_active = w7d_mild = w7d_silent = w7d_churn = 0
        w7d_lc = pd.Series('未知', index=df_full.index) if df_full is not None else pd.Series()

    # 风险
    risk_total = int((df['风险标签'] != '正常').sum()) if '风险标签' in df.columns else 0

    # 等级
    high_loss = int((df['用户等级_动态'] == '高损耗用户').sum()) if '用户等级_动态' in df.columns else 0
    excellent = int((df['用户等级_动态'] == '优质用户').sum()) if '用户等级_动态' in df.columns else 0
    violent = int((df['用户等级_动态'] == '暴力').sum()) if '用户等级_动态' in df.columns else 0

    # 到期
    expired = int((df['用户生命周期状态_7d'] == '已到期').sum()) if '用户生命周期状态_7d' in df.columns else 0

    # 里程
    avg_km = safe_mean(df.get('近7d总行驶距离_km', pd.Series()))
    med_km = safe_median(df.get('近7d总行驶距离_km', pd.Series()))

    # 出勤
    avg_attend_days = safe_mean(df.get('近7d出勤天数', pd.Series()))
    avg_attend_rate = safe_mean(df.get('近7d出勤率', pd.Series()))

    # 用电
    monthly_energy_mean = safe_mean(df_att.get('单合约月度用电度数预估_kWh', pd.Series())) if df_att is not None else 0
    monthly_energy_med = safe_median(df_att.get('单合约月度用电度数预估_kWh', pd.Series())) if df_att is not None else 0
    monthly_energy_p90 = safe_quantile(df_att.get('单合约月度用电度数预估_kWh', pd.Series()), 0.9) if df_att is not None else 0

    # 超标
    over_150 = int((df_att['单合约月度用电度数预估_kWh'] > MONTHLY_ENERGY_LOSS_THRESHOLD).sum()) if df_att is not None else 0

    # 换电（换电运营核心指标）
    daily_swaps_mean = safe_mean(df_full.get('近7d日均换电次数', pd.Series())) if df_full is not None else 0
    total_swaps_mean = safe_mean(df_full.get('近7d总换电次数', pd.Series())) if df_full is not None else 0
    # SOC 类指标仅统计实际有换电的用户，避免 0/负值拉低均值
    swap_users = df_full[df_full['近7d总换电次数'].fillna(0) > 0] if df_full is not None and '近7d总换电次数' in df_full.columns else df_full
    swap_users_n = len(swap_users) if swap_users is not None else 0
    swap_soc_mean = safe_mean(swap_users.get('近7d单次换电SOC消耗', pd.Series())) if swap_users is not None else 0
    swap_friendly_mean = safe_mean(swap_users.get('近7d换电友好分', pd.Series())) if swap_users is not None else 0
    pick_soc_mean = safe_mean(swap_users.get('近7d取电时平均SOC', pd.Series())) if swap_users is not None else 0

    # 加权均值（按换电次数加权）：避免高频换电用户被低频用户稀释
    def _weighted_mean(val_col, w_col='近7d总换电次数'):
        if swap_users is None or val_col not in swap_users.columns or w_col not in swap_users.columns:
            return 0
        v = swap_users[val_col].astype(float)
        wt = swap_users[w_col].astype(float)
        mask = v.notna() & wt.notna() & (wt > 0)
        if mask.sum() == 0:
            return 0
        return float((v[mask] * wt[mask]).sum() / wt[mask].sum())

    pick_soc_wmean = _weighted_mean('近7d取电时平均SOC')
    return_soc_wmean = _weighted_mean('近7d还电时平均SOC')
    swap_soc_wmean = _weighted_mean('近7d单次换电SOC消耗')
    return_soc_mean = safe_mean(swap_users.get('近7d还电时平均SOC', pd.Series())) if swap_users is not None else 0
    night_swap_pct = safe_mean(swap_users.get('近7d深夜换电占比', pd.Series())) if swap_users is not None else 0
    peak_swap_pct = safe_mean(swap_users.get('近7d平峰换电占比', pd.Series())) if swap_users is not None else 0
    deep_discharge = 0  # 默认值，6.2 章节若数据可用会覆盖

    # 电流/电耗
    avg_cur_mean = safe_mean(df.get('近7d平均骑行电流_A', pd.Series()))
    avg_cur_med = safe_median(df.get('近7d平均骑行电流_A', pd.Series()))
    avg_cur_p90 = safe_quantile(df.get('近7d平均骑行电流_A', pd.Series()), 0.9)
    max_cur_mean = safe_mean(df.get('近7d最大电流_A', pd.Series()))
    max_cur_med = safe_median(df.get('近7d最大电流_A', pd.Series()))
    max_cur_p90 = safe_quantile(df.get('近7d最大电流_A', pd.Series()), 0.9)
    energy_100_mean = safe_mean(df.get('近7d百公里电耗_kWh', pd.Series()))
    energy_100_med = safe_median(df.get('近7d百公里电耗_kWh', pd.Series()))
    energy_100_p90 = safe_quantile(df.get('近7d百公里电耗_kWh', pd.Series()), 0.9)

    daily_divisor = avg_valid_days if avg_valid_days and avg_valid_days > 0 else 7
    daily_km = avg_km / daily_divisor
    daily_med_km = med_km / daily_divisor
    daily_swaps = daily_swaps_mean

    # 设备状态 — 解析离线天数，分档展示
    import re
    def parse_device_status(s):
        """解析 '暂离线(6天)' / '异常(离线16天)' → (类型, 天数)
        按天数重新分档：≤1天=正常在线（换电场景1天内属正常在线），
        2~6天=暂离线，≥7天=异常离线。
        """
        s = str(s)
        m = re.search(r'(\d+)天', s)
        days = int(m.group(1)) if m else 0
        if days <= 1:
            return ('正常在线', days)
        elif days >= 7:
            return ('异常离线', days)
        else:
            return ('暂离线', days)

    if '设备状态监控' in df.columns:
        status_parsed = df['设备状态监控'].apply(parse_device_status).apply(pd.Series)
        status_parsed.columns = ['状态类型', '离线天数']
        # 离线天数分档
        def bucket_days(d):
            if d <= 1: return '≤1天'
            if d <= 3: return '2-3天'
            if d <= 6: return '4-6天'
            if d <= 14: return '7-14天'
            return '>14天'
        status_parsed['离线分档'] = status_parsed['离线天数'].apply(bucket_days)
        # 组合标签
        status_parsed['设备状态标签'] = status_parsed.apply(
            lambda r: f"{r['状态类型']}({r['离线分档']})", axis=1)
        status_detail = status_parsed['设备状态标签'].value_counts()
        # 将状态类型挂到 df 上，供生命周期交叉表使用
        df['_状态类型'] = status_parsed['状态类型'].values
    else:
        status_detail = pd.Series(dtype=int)
        df['_状态类型'] = '其他'

    w('| 指标 | 数值 | 解读 |')
    w('|------|------|------|')
    w(f'| **总用户数** | {total_users:,} 人 | 全平台注册用户总量 |')
    w(f'| **数据窗口** | 近7天，人均有效数据 {avg_valid_days:.1f} 天 | 出勤率 = 出勤天数 ÷ 有效数据天数 |')
    w(f'| **加权7天出勤均值** | {avg_w7d_attend:.1f} 天（中位 {w7d_med:.1f}） | 按出勤率推算至满7天，消除数据窗口偏差 |')
    w(f'| **活跃用户（加权≥5天）** | {w7d_active:,} 人 ({fmt_pct(w7d_active, total_users):.1f}%) | 核心活跃群体 |')
    w(f'| **轻度活跃（加权1~5天）** | {w7d_mild:,} 人 ({fmt_pct(w7d_mild, total_users):.1f}%) | 有活动但频次偏低 |')
    w(f'| **沉默用户** | {w7d_silent:,} 人 ({fmt_pct(w7d_silent, total_users):.1f}%) | 有数据但出勤率为0 |')
    w(f'| **流失用户** | {w7d_churn:,} 人 ({fmt_pct(w7d_churn, total_users):.1f}%) | 近7天无任何有效数据 |')
    w(f'| **整体出勤率** | {avg_attend_rate*100:.1f}% | 出勤天数 ÷ 有效数据天数（非 /7） |')
    w(f'| **人均日均骑行里程** | {daily_km:.1f} km | 中位数 {daily_med_km:.1f} km |')
    w(f'| **人均日均换电次数** | {daily_swaps:.1f} 次 | 中位数 {safe_median(df_full.get("近7d日均换电次数", pd.Series())):.1f} 次 |')
    w(f'| **单次换电SOC消耗** | {swap_soc_mean:.1f}% | 取电{pick_soc_mean:.0f}% → 还电{return_soc_mean:.0f}%（仅换电用户） |')
    w(f'| **换电友好分均值** | {swap_friendly_mean:.1f} | 越高表示换电行为越规范 |')
    w(f'| **深夜换电占比** | {night_swap_pct*100:.1f}% | 夜间换电占比，反映夜间运营需求 |')
    w(f'| **平均骑行电流** | {avg_cur_mean:.2f} A | 中位 {avg_cur_med:.2f}A, P90={avg_cur_p90:.2f}A |')
    w(f'| **平均最大电流** | {max_cur_mean:.2f} A | 中位 {max_cur_med:.2f}A, P90={max_cur_p90:.2f}A |')
    w(f'| **百公里电耗均值** | {energy_100_mean:.2f} kWh | 中位 {energy_100_med:.2f}kWh, P90={energy_100_p90:.2f}kWh |')
    w(f'| **高风险用户占比** | {fmt_pct(risk_total, total_users):.1f}%（{risk_total:,}人） | 风险标签 ≠ 正常 |')
    w(f'| **优质用户占比** | {fmt_pct(excellent, total_users):.1f}%（{excellent:,}人） | 评分 ≥ P75 且无显著风险 |')
    w(f'| **高损耗用户占比** | {fmt_pct(high_loss, total_users):.1f}%（{high_loss:,}人） | 评分低或电池损耗高于均值 |')
    w(f'| **暴力用户** | {fmt_pct(violent, total_users):.1f}%（{violent:,}人） | 存在严重电池损害行为 |')
    w(f'| **人月均用电预估** | {monthly_energy_mean:.1f} 度 | 中位数 {monthly_energy_med:.1f}度, P90={monthly_energy_p90:.1f}度 |')
    w(f'| **月用电超标（>{MONTHLY_ENERGY_LOSS_THRESHOLD:.0f}度）** | {fmt_pct(over_150, len(df_att)):.1f}%（{over_150:,}人） | 超过盈亏线 |')
    # 设备状态汇总
    online_total = int(status_parsed['状态类型'].eq('正常在线').sum()) if len(status_detail) > 0 else 0
    temp_total = int(status_parsed['状态类型'].eq('暂离线').sum()) if len(status_detail) > 0 else 0
    abnormal_total = int(status_parsed['状态类型'].eq('异常离线').sum()) if len(status_detail) > 0 else 0
    w(f'| **设备正常在线** | {online_total:,} 人 ({fmt_pct(online_total, total_users):.1f}%) | ≤1天内有数据上报，正常活跃 |')
    w(f'| **设备暂离线** | {temp_total:,} 人 ({fmt_pct(temp_total, total_users):.1f}%) | 2~6天无新数据，正常休眠态 |')
    w(f'| **设备异常离线** | {abnormal_total:,} 人 ({fmt_pct(abnormal_total, total_users):.1f}%) | ≥7天无新数据，需关注设备状态 |')
    w()
    w(f'> **关于加权出勤的说明**：由于数据窗口不满7天（人均仅 {avg_valid_days:.1f} 天有效数据），直接统计出勤天数会严重低估。')
    w(f'> 加权7天出勤 = `近7d出勤率 × 7`，将出勤行为按规定比例推算至满7天窗口。')
    w(f'> 例：用户在4天有效数据中出勤4天（出勤率100%）→ 加权推算 = 7天全勤。')
    w(f'> 当前加权后人均出勤 **{avg_w7d_attend:.1f} 天**（中位 {w7d_med:.1f} 天），整体出勤率 **{avg_attend_rate*100:.1f}%**。')
    w()

    # ═══════════════ 第二章：用户结构分析 ═══════════════
    w('---')
    w()
    w('## 二、用户整体结构分析')
    w()
    w('### 2.1 用户等级分布')
    w()

    level_order = ['优质用户', '良好用户', '普通用户', '高损耗用户', '暴力', '沉默用户', '观察期']
    level_stats = []
    for lv in level_order:
        sub = df[df['用户等级_动态'] == lv] if '用户等级_动态' in df.columns else pd.DataFrame()
        cnt = len(sub)
        if cnt > 0:
            sc = safe_mean(sub.get('重评分数', pd.Series()))
            cur = safe_mean(sub.get('近7d平均骑行电流_A', pd.Series()))
            mcur = safe_mean(sub.get('近7d最大电流_A', pd.Series()))
            en = safe_mean(sub.get('近7d百公里电耗_kWh', pd.Series()))
            km = safe_mean(sub.get('近7d总行驶距离_km', pd.Series()))
            ar = safe_mean(sub.get('近7d出勤率', pd.Series()))
            level_stats.append((lv, cnt, sc, cur, mcur, en, km, ar))

    w('| 用户等级 | 人数 | 占比 | 平均评分 | 平均电流(A) | 最大电流(A) | 百公里电耗(kWh) | 日均里程(km) | 出勤率 |')
    w('|----------|------|------|----------|------------|------------|----------------|------------|--------|')
    for lv, cnt, sc, cur, mcur, en, km, ar in level_stats:
        w(f'| {lv} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% | {sc:.1f} | {cur:.2f} | {mcur:.2f} | {en:.2f} | {km/daily_divisor:.2f} | {ar*100:.0f}% |')
    w()

    # 各等级风险
    if '风险标签' in df.columns:
        w('### 2.2 各等级风险用户占比')
        w()
        w('| 等级 | 总人数 | 风险用户数 | 风险率 |')
        w('|------|--------|-----------|--------|')
        for lv, cnt, sc, cur, mcur, en, km, ar in level_stats:
            sub = df[df['用户等级_动态'] == lv]
            risk_cnt = int((sub['风险标签'] != '正常').sum())
            w(f'| {lv} | {cnt:,} | {risk_cnt:,} | {fmt_pct(risk_cnt, cnt):.1f}% |')
        w()

    # ═══════════════ 第三章：生命周期（加权） ═══════════════
    w('---')
    w()
    w('## 三、生命周期结构分析（加权）')
    w()
    w('> 加权逻辑：`加权7天出勤 = 近7d出勤率 × 7`，消除有效数据不足7天导致的偏差。')
    w('> `已到期` 和 `新用户` 保留原有判定（基于合约状态和入网日期），其余状态按加权出勤重新分类。')
    w()

    if len(w7d_lc) > 0:
        # 将加权生命周期合并回 df（基于 user_id 对齐 df_full → df）
        if 'user_id' in df.columns and 'user_id' in df_full.columns:
            lc_map = df_full[['user_id']].copy()
            lc_map['加权生命周期'] = w7d_lc.values
            df = df.merge(lc_map, on='user_id', how='left')
            df['加权生命周期'] = df['加权生命周期'].fillna('未知')
        else:
            # fallback: 按行对齐
            df['加权生命周期'] = list(w7d_lc)[:len(df)]

        lc_col = '加权生命周期'
        w('### 3.1 生命周期状态分布')
        w()
        lc_counts = df[lc_col].value_counts()
        w('| 生命周期状态 | 人数 | 占比 |')
        w('|-------------|------|------|')
        for state, cnt in lc_counts.items():
            w(f'| {state} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% |')
        w()

        # 生命周期 × 设备状态
        w('### 3.2 生命周期 × 设备在线状态')
        w()
        w('| 生命周期 | 总人数 | 暂离线 | 异常离线 | 正常在线 |')
        w('|----------|--------|--------|----------|----------|')
        for state in lc_counts.index:
            sub = df[df[lc_col] == state]
            m = sub['_状态类型'].value_counts()
            w(f'| {state} | {len(sub):,} | {m.get("暂离线",0):,}({fmt_pct(m.get("暂离线",0),len(sub)):.0f}%) | {m.get("异常离线",0):,}({fmt_pct(m.get("异常离线",0),len(sub)):.0f}%) | {m.get("正常在线",0):,} |')
        w()

        # 等级 × 生命周期（已剔除已到期用户）
        w('### 3.3 各等级生命周期分布')
        w()
        w('| 等级 | 总数 | 活跃 | 轻度活跃 | 沉默/流失 |')
        w('|------|------|------|----------|-----------|')
        for lv, cnt, sc, cur, mcur, en, km, ar in level_stats:
            sub = df[df['用户等级_动态'] == lv]
            lc = sub[lc_col].value_counts()
            silent = lc.get('沉默', 0) + lc.get('流失', 0)
            w(f'| {lv} | {cnt:,} | {lc.get("活跃",0):,}({fmt_pct(lc.get("活跃",0),cnt):.0f}%) | {lc.get("轻度活跃",0):,}({fmt_pct(lc.get("轻度活跃",0),cnt):.0f}%) | {silent:,}({fmt_pct(silent,cnt):.0f}%) |')
        w()

    # ═══════════════ 第四章：风险标签 ═══════════════
    w('---')
    w()
    w('## 四、风险标签分析')
    w()

    if '风险标签' in df.columns:
        risk_dist = df['风险标签'].value_counts().head(15)
        w('### 4.1 风险标签 TOP 15')
        w()
        w('| 风险标签 | 人数 | 占比 |')
        w('|----------|------|------|')
        for tag, cnt in risk_dist.items():
            w(f'| {tag} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% |')
        w()

        w(f'**风险用户整体**：{risk_total:,}人（{fmt_pct(risk_total, total_users):.1f}%），')
        risk_sub = df[df['风险标签'] != '正常']
        w(f'平均最大电流 {safe_mean(risk_sub.get("近7d最大电流_A", pd.Series())):.2f}A，')
        w(f'平均百公里电耗 {safe_mean(risk_sub.get("近7d百公里电耗_kWh", pd.Series())):.2f}kWh，')
        w(f'平均重评分数 {safe_mean(risk_sub.get("重评分数", pd.Series())):.1f}。')
        w()

    # ═══════════════ 第五章：用户形态 ═══════════════
    w('---')
    w()
    w('## 五、用户形态分析')
    w()

    if '用户形态_综合_7d' in df.columns:
        ctype_order_7d = ['专送骑手', '众包骑手', '地摊/储能', '数据不足']
        w('### 5.1 用户形态分布')
        w()
        w('| 用户形态 | 人数 | 占比 | 平均电流(A) | 百公里电耗(kWh) | 日均里程(km) | 平均评分 |')
        w('|----------|------|------|------------|----------------|------------|----------|')
        for ct in ctype_order_7d:
            sub = df[df['用户形态_综合_7d'] == ct]
            cnt = len(sub)
            if cnt > 0:
                w(f'| {ct} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% | {safe_mean(sub.get("近7d平均骑行电流_A", pd.Series())):.2f} | {safe_mean(sub.get("近7d百公里电耗_kWh", pd.Series())):.2f} | {safe_mean(sub.get("近7d总行驶距离_km", pd.Series())) / daily_divisor:.2f} | {safe_mean(sub.get("重评分数", pd.Series())):.1f} |')
        w()

    if df_full is not None and '车辆形态_7d' in df_full.columns:
        vtype_order_7d = ['电动自行车', '电动轻便摩托车', '电动摩托车', '改装/超速车', '地摊/储能', '数据不足']
        w('### 5.2 车辆形态分布')
        w()
        w('| 车辆形态 | 人数 | 占比 | P90速度(km/h) | 最高速度(km/h) | 峰值功率(W) | 平均功率(W) | 日均里程(km) | 平均电流(A) |')
        w('|----------|------|------|--------------|---------------|------------|------------|------------|------------|')
        for vt in vtype_order_7d:
            sub = df_full[df_full['车辆形态_7d'] == vt]
            cnt = len(sub)
            if cnt > 0:
                # 平均功率 ≈ 峰值功率 × (平均电流 / 最大电流)，假设电压恒定
                pk = sub.get('近7d峰值功率_W', pd.Series())
                ai = sub.get('近7d平均骑行电流_A', pd.Series())
                mi = sub.get('近7d最大电流_A', pd.Series())
                avg_power = (pk * (ai / mi.replace(0, pd.NA))).mean() if (mi > 0).any() else 0
                w(f'| {vt} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% | {safe_mean(sub.get("近7d_P90骑行速度_kmh", pd.Series())):.1f} | {safe_mean(sub.get("近7d最高速度_kmh", pd.Series())):.1f} | {safe_mean(pk):.0f} | {avg_power:.0f} | {safe_mean(sub.get("近7d总行驶距离_km", pd.Series())) / daily_divisor:.2f} | {safe_mean(ai):.2f} |')
        w()
        w('> **分类规则**（基于近7天 P95 速度分位数）：')
        w('>')
        w('> - **电动自行车**：P95 ≤ 25 km/h（国标限速）')
        w('>')
        w('> - **电动轻便摩托车**：P95 在 25~50 km/h 之间')
        w('>')
        w('> - **电动摩托车**：P95 > 50 km/h')
        w('>')
        w('> - **改装/超速车**：高速 + 高电流 同时满足，或峰值功率 ≥ 8000W')
        w('>')
        w('> - **地摊/储能**：非骑行用电场景（速度极低 + 持续耗电）')
        w('>')
        w('> - **数据不足**：无有效骑行数据，无法判定')
        w('>')
        w('> 注：表中"平均功率"由 峰值功率 × (平均电流/最大电流) 推算，反映骑行过程中的实际平均输出功率。')
        w()
        w()

    # ═══════════════ 5.3 众包/专送判定特征 ═══════════════
    has_new_features = df_full is not None and any(col in df_full.columns for col in ['近7d平均上线时间熵值', '近7d平均路线曲折系数', '近7d平均速度变异系数'])
    if has_new_features and '用户形态_综合_7d' in df_full.columns:
        w('### 5.3 众包/专送判定特征对比')
        w()
        w('| 特征指标 | 专送骑手 | 众包骑手 | 全体平均 | 文档标准参考 |')
        w('|----------|---------|---------|---------|-------------|')

        zhuan = df_full[df_full['用户形态_综合_7d'] == '专送骑手']
        zhong = df_full[df_full['用户形态_综合_7d'] == '众包骑手']

        # 上线时间熵值
        if '近7d平均上线时间熵值' in df_full.columns:
            w(f'| 上线时间熵值 | {safe_mean(zhuan.get("近7d平均上线时间熵值", pd.Series())):.2f} | {safe_mean(zhong.get("近7d平均上线时间熵值", pd.Series())):.2f} | {safe_mean(df_full.get("近7d平均上线时间熵值", pd.Series())):.2f} | 专送≤3.85，众包>3.85 |')

        # 路线曲折系数
        if '近7d平均路线曲折系数' in df_full.columns:
            w(f'| 路线曲折系数 | {safe_mean(zhuan.get("近7d平均路线曲折系数", pd.Series())):.2f} | {safe_mean(zhong.get("近7d平均路线曲折系数", pd.Series())):.2f} | {safe_mean(df_full.get("近7d平均路线曲折系数", pd.Series())):.2f} | 专送≤3.0，众包>3.0 |')

        # 速度变异系数
        if '近7d平均速度变异系数' in df_full.columns:
            w(f'| 速度变异系数 | {safe_mean(zhuan.get("近7d平均速度变异系数", pd.Series())):.2f} | {safe_mean(zhong.get("近7d平均速度变异系数", pd.Series())):.2f} | {safe_mean(df_full.get("近7d平均速度变异系数", pd.Series())):.2f} | 专送≤1.0，众包>1.0 |')

        # 跨区域转移次数
        if '近7d最大跨区域转移次数' in df_full.columns:
            w(f'| 跨区域转移次数 | {safe_mean(zhuan.get("近7d最大跨区域转移次数", pd.Series())):.1f} | {safe_mean(zhong.get("近7d最大跨区域转移次数", pd.Series())):.1f} | {safe_mean(df_full.get("近7d最大跨区域转移次数", pd.Series())):.1f} | 专送<3，众包>3 |')

        # 静止时长占比
        if '近7d平均静止时长占比' in df_full.columns:
            w(f'| 静止时长占比 | {safe_mean(zhuan.get("近7d平均静止时长占比", pd.Series())):.2f} | {safe_mean(zhong.get("近7d平均静止时长占比", pd.Series())):.2f} | {safe_mean(df_full.get("近7d平均静止时长占比", pd.Series())):.2f} | 专送<0.5，众包>0.5 |')

        w()
        w('> 注：特征阈值基于《众包专送判断.md》文档标准，专送骑手表现为规律性强、活动集中、路线顺路、速度稳定')
        w()

    # ═══════════════ 第六章：换电行为分析 ═══════════════
    w('---')
    w()
    w('## 六、换电行为分析')
    w()
    w('> 换电是本平台核心业务环节。以下从换电频次、SOC流转、时段偏好三维度刻画用户换电习惯。')
    w()

    # 6.1 换电频次分布
    if df_full is not None and '近7d日均换电次数' in df_full.columns:
        swaps = df_full['近7d日均换电次数'].fillna(0)
        w('### 6.1 换电频次分布')
        w()
        bins = [(-0.01, 1, '低频(<1次/天)'), (1, 2, '中低频(1~2次)'), (2, 3, '中频(2~3次)'),
                (3, 5, '高频(3~5次)'), (5, 999, '超高频(>5次)')]
        w('| 换电频次档位 | 人数 | 占比 | 日均里程(km) |')
        w('|-------------|------|------|-------------|')
        for lo, hi, label in bins:
            mask = (swaps > lo) & (swaps <= hi)
            cnt = int(mask.sum())
            km = safe_mean(df_full.loc[mask, '近7d总行驶距离_km']) / daily_divisor if cnt > 0 else 0
            w(f'| {label} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% | {km:.1f} |')
        w()

    # 6.2 SOC 流转分析（仅统计实际有换电的用户）
    if swap_users is not None and '近7d单次换电SOC消耗' in swap_users.columns:
        w('### 6.2 换电SOC流转分析')
        w()
        w(f'> 统计口径：仅含近7天有换电记录的用户 {swap_users_n:,} 人，剔除 0 次换电用户的空值干扰。')
        w('>')
        w('> **加权均值**按用户近7天换电次数加权（避免高频用户被低频用户稀释）：例如一人换100次SOC=100%、一人换1次SOC=50%，简单均值=75%偏低，加权后≈99.5% 更贴近真实单次水平。')
        w()
        w('| 指标 | 简单均值 | 加权均值 | 中位数 | P90 | 说明 |')
        w('|------|---------|---------|--------|-----|------|')
        w(f'| 取电时平均SOC | {pick_soc_mean:.1f}% | {pick_soc_wmean:.1f}% | {safe_median(swap_users.get("近7d取电时平均SOC", pd.Series())):.1f}% | {safe_quantile(swap_users.get("近7d取电时平均SOC", pd.Series()), 0.9):.1f}% | 换电取走时的电池余量 |')
        w(f'| 还电时平均SOC | {return_soc_mean:.1f}% | {return_soc_wmean:.1f}% | {safe_median(swap_users.get("近7d还电时平均SOC", pd.Series())):.1f}% | {safe_quantile(swap_users.get("近7d还电时平均SOC", pd.Series()), 0.9):.1f}% | 骑行结束归还时的余量 |')
        w(f'| 单次换电SOC消耗 | {swap_soc_mean:.1f}% | {swap_soc_wmean:.1f}% | {safe_median(swap_users.get("近7d单次换电SOC消耗", pd.Series())):.1f}% | {safe_quantile(swap_users.get("近7d单次换电SOC消耗", pd.Series()), 0.9):.1f}% | 单次换电周期内的电量消耗 |')
        w()
        deep_discharge = int((swap_users['近7d还电时平均SOC'].fillna(100) < 10).sum()) if '近7d还电时平均SOC' in swap_users.columns else 0
        w(f'> **深度放电预警**：{deep_discharge:,}人（{fmt_pct(deep_discharge, swap_users_n):.1f}%）还电时SOC<10%，长期深度放电加速电池衰减，建议引导提前换电。')
        w()

    # 6.3 换电时段偏好（仅换电用户）
    if swap_users is not None and any(c in swap_users.columns for c in ['近7d平峰换电占比', '近7d深夜换电占比', '近7d高峰换电总次数']):
        w('### 6.3 换电时段偏好')
        w()
        w('| 时段 | 人均总次数 | 人均日均次数 | 占比均值 | 说明 |')
        w('|------|-----------|-------------|---------|------|')
        if '近7d高峰换电总次数' in swap_users.columns:
            w(f'| 高峰换电 | {safe_mean(swap_users.get("近7d高峰换电总次数", pd.Series())):.1f} | {safe_mean(swap_users.get("近7d日均高峰换电次数", pd.Series())):.2f} | — | 午间+晚间高峰换电 |')
        if '近7d平峰换电总次数' in swap_users.columns:
            w(f'| 平峰换电 | {safe_mean(swap_users.get("近7d平峰换电总次数", pd.Series())):.1f} | {safe_mean(swap_users.get("近7d日均平峰换电次数", pd.Series())):.2f} | {peak_swap_pct*100:.1f}% | 日间非高峰换电 |')
        if '近7d深夜换电总次数' in swap_users.columns:
            w(f'| 深夜换电 | {safe_mean(swap_users.get("近7d深夜换电总次数", pd.Series())):.1f} | {safe_mean(swap_users.get("近7d日均深夜换电次数", pd.Series())):.2f} | {night_swap_pct*100:.1f}% | 22:00后换电，反映夜间运营强度 |')
        w()
        if '换电时段偏好_7d' in swap_users.columns:
            pref = swap_users['换电时段偏好_7d'].value_counts().head(6)
            w('**换电时段偏好分布（近7天主导换电时段）**')
            w()
            w('| 换电时段偏好 | 人数 | 占比 |')
            w('|-------------|------|------|')
            for p, cnt in pref.items():
                w(f'| {p} | {cnt:,} | {fmt_pct(cnt, swap_users_n):.1f}% |')
            w()

    # ═══════════════ 第七章：区域分析 ═══════════════
    w('---')
    w()
    w('## 七、区域运营分析')
    w()

    if '核心活动省份' in df.columns:
        w('### 7.1 主要省份对比（TOP 10）')
        w()
        prov_counts = df['核心活动省份'].value_counts().head(10)
        w('<table class="keep-together">')
        w('<thead><tr><th>省份</th><th>总用户</th><th>优质率</th><th>高损耗率</th></tr></thead>')
        w('<tbody>')
        for prov, cnt in prov_counts.items():
            sub = df[df['核心活动省份'] == prov]
            exc = fmt_pct((sub['用户等级_动态'] == '优质用户').sum(), cnt)
            hloss = fmt_pct((sub['用户等级_动态'] == '高损耗用户').sum(), cnt)
            w(f'<tr><td>{prov}</td><td>{cnt:,}</td><td>{exc:.0f}%</td><td>{hloss:.0f}%</td></tr>')
        w('</tbody></table>')
        w()

    # 7.2 主要城市综合对比（合并用户质量 + 运营参数，TOP 10）
    if '核心活动城市' in df.columns:
        w('### 7.2 主要城市综合对比（TOP 10）')
        w()
        city_counts = df['核心活动城市'].value_counts().head(10)
        w('<table class="keep-together">')
        w('<thead><tr><th>城市</th><th>用户数</th><th>优质率</th><th>高损耗率</th><th>日均里程(km)</th><th>日均换电(次)</th><th>出勤率</th><th>P90速度(km/h)</th><th>单次SOC消耗(%)</th><th>深夜换电占比</th><th>改装车占比</th></tr></thead>')
        w('<tbody>')
        for city, cnt in city_counts.items():
            sub = df[df['核心活动城市'] == city]
            exc = fmt_pct((sub['用户等级_动态'] == '优质用户').sum(), cnt)
            hloss = fmt_pct((sub['用户等级_动态'] == '高损耗用户').sum(), cnt)
            # 运营参数从 df_full 取
            sub_full = df_full[df_full['核心活动城市'] == city] if df_full is not None else sub
            swap_sub = sub_full[sub_full['近7d总换电次数'].fillna(0) > 0] if '近7d总换电次数' in sub_full.columns else sub_full
            daily_km = safe_mean(sub_full.get('近7d总行驶距离_km', pd.Series())) / daily_divisor
            daily_swaps = safe_mean(sub_full.get('近7d日均换电次数', pd.Series()))
            attend = safe_mean(sub_full.get('近7d出勤率', pd.Series())) * 100
            p90_speed = safe_mean(sub_full.get('近7d_P90骑行速度_kmh', pd.Series()))
            soc_consum = safe_mean(swap_sub.get('近7d单次换电SOC消耗', pd.Series()))
            night_pct = safe_mean(swap_sub.get('近7d深夜换电占比', pd.Series())) * 100
            mod_pct = fmt_pct((sub_full.get('车辆形态_7d', pd.Series()) == '改装/超速车').sum(), cnt)
            w(f'<tr><td>{city}</td><td>{cnt:,}</td><td>{exc:.0f}%</td><td>{hloss:.0f}%</td><td>{daily_km:.1f}</td><td>{daily_swaps:.2f}</td><td>{attend:.1f}%</td><td>{p90_speed:.1f}</td><td>{soc_consum:.1f}</td><td>{night_pct:.1f}%</td><td>{mod_pct:.1f}%</td></tr>')
        w('</tbody></table>')
        w()

    # ═══════════════ 第八章：代理商横向对比（合资公司+自营） ═══════════════
    w('---')
    w()
    # 横向页面容器：代理列数多，用横向 A4 以获得足够宽度
    w('<div class="agent-chapter" markdown="1">')
    w()
    w('## 八、代理商横向对比（合资公司 + 自营）')
    w()
    # 仅保留一级为合资公司/自营的用户
    df_agent_full = df_full[df_full['一级'].isin(['合资公司', '自营'])] if df_full is not None and '一级' in df_full.columns else None
    if df_agent_full is not None and len(df_agent_full) > 0 and '三级' in df_agent_full.columns:
        agent_total = len(df_agent_full)
        w(f'> 统计口径：仅含一级代理为「合资公司」与「自营」的用户，共 {agent_total:,} 人。')
        w('>')
        w('> 数据源：`在网用户-加代理.csv` 按用户 id 匹配合并，对比指标与总体报告口径一致。')
        w()

        # ── 预计算各代理核心指标（供转置表格复用，指标与总体 1.1 口径一致）──
        risk_by_id = df.set_index('用户id')['风险标签'] if '风险标签' in df.columns else None
        # 设备状态解析函数复用（与 1.1 一致）
        import re as _re
        def _parse_dev(s):
            s = str(s)
            m = _re.search(r'(\d+)天', s)
            d = int(m.group(1)) if m else 0
            if d <= 1: return '正常在线'
            if d >= 7: return '异常离线'
            return '暂离线'
        dev_col = '设备状态监控'

        agent_metrics = []
        for agent, sub in df_agent_full.groupby('三级'):
            cnt = len(sub)
            lvl1 = sub['一级'].iloc[0] if len(sub) > 0 else ''
            swap_sub = sub[sub['近7d总换电次数'].fillna(0) > 0] if '近7d总换电次数' in sub.columns else sub
            swap_n = len(swap_sub)
            # 有效数据天数与加权出勤
            valid_days = sub.get('近7天有数据天数', pd.Series())
            avg_valid = safe_mean(valid_days)
            raw_rate = sub.get('近7d出勤率', pd.Series()).fillna(0)
            w7d_attend = (raw_rate * 7).clip(upper=7)
            w7d_attend[valid_days.fillna(0) == 0] = 0
            w7d_active = int((w7d_attend >= 5).sum())
            w7d_silent = int(((w7d_attend == 0) & (valid_days.fillna(0) > 0)).sum())
            w7d_churn = int((valid_days.fillna(0) == 0).sum())
            # 等级
            exc = fmt_pct((sub.get('用户等级_动态', pd.Series()) == '优质用户').sum(), cnt)
            good = fmt_pct((sub.get('用户等级_动态', pd.Series()) == '良好用户').sum(), cnt)
            hloss = fmt_pct((sub.get('用户等级_动态', pd.Series()) == '高损耗用户').sum(), cnt)
            mid_risk = fmt_pct((sub.get('用户等级_动态', pd.Series()) == '中风险用户').sum(), cnt)
            violent = fmt_pct((sub.get('用户等级_动态', pd.Series()) == '暴力').sum(), cnt)
            # 骑行
            daily_km = safe_mean(sub.get('近7d总行驶距离_km', pd.Series())) / (avg_valid if avg_valid > 0 else 7)
            med_km = safe_median(sub.get('近7d总行驶距离_km', pd.Series())) / (avg_valid if avg_valid > 0 else 7)
            p90_speed = safe_mean(sub.get('近7d_P90骑行速度_kmh', pd.Series()))
            # 换电
            daily_swaps = safe_mean(sub.get('近7d日均换电次数', pd.Series()))
            soc_consum = safe_mean(swap_sub.get('近7d单次换电SOC消耗', pd.Series()))
            pick_soc = safe_mean(swap_sub.get('近7d取电时平均SOC', pd.Series()))
            return_soc = safe_mean(swap_sub.get('近7d还电时平均SOC', pd.Series()))
            friendly = safe_mean(swap_sub.get('近7d换电友好分', pd.Series()))
            night_pct = safe_mean(swap_sub.get('近7d深夜换电占比', pd.Series())) * 100
            peak_pct = safe_mean(swap_sub.get('近7d平峰换电占比', pd.Series())) * 100
            # 电流/电耗
            avg_cur = safe_mean(sub.get('近7d平均骑行电流_A', pd.Series()))
            max_cur = safe_mean(sub.get('近7d最大电流_A', pd.Series()))
            energy_100 = safe_mean(sub.get('近7d百公里电耗_kWh', pd.Series()))
            # 风险
            if '风险标签' in sub.columns:
                risk_pct = fmt_pct((sub['风险标签'].fillna('正常') != '正常').sum(), cnt)
            elif risk_by_id is not None:
                r = sub['用户id'].map(risk_by_id).fillna('正常')
                risk_pct = fmt_pct((r != '正常').sum(), cnt)
            else:
                risk_pct = 0
            mod_pct = fmt_pct((sub.get('车辆形态_7d', pd.Series()) == '改装/超速车').sum(), cnt)
            # 设备状态
            if dev_col in sub.columns:
                dev = sub[dev_col].apply(_parse_dev)
                online_pct = fmt_pct((dev == '正常在线').sum(), cnt)
                temp_pct = fmt_pct((dev == '暂离线').sum(), cnt)
                abn_pct = fmt_pct((dev == '异常离线').sum(), cnt)
            else:
                online_pct = temp_pct = abn_pct = 0
            agent_metrics.append({
                'agent': agent, 'lvl1': lvl1, 'cnt': cnt, 'swap_n': swap_n,
                'avg_valid': avg_valid, 'w7d_attend': safe_mean(w7d_attend), 'attend': safe_mean(raw_rate) * 100,
                'w7d_active': w7d_active, 'w7d_silent': w7d_silent, 'w7d_churn': w7d_churn,
                'exc': exc, 'good': good, 'hloss': hloss, 'mid_risk': mid_risk, 'violent': violent,
                'daily_km': daily_km, 'med_km': med_km, 'p90_speed': p90_speed,
                'daily_swaps': daily_swaps, 'soc_consum': soc_consum, 'pick_soc': pick_soc, 'return_soc': return_soc,
                'friendly': friendly, 'night_pct': night_pct, 'peak_pct': peak_pct,
                'avg_cur': avg_cur, 'max_cur': max_cur, 'energy_100': energy_100,
                'risk_pct': risk_pct, 'mod_pct': mod_pct,
                'online_pct': online_pct, 'temp_pct': temp_pct, 'abn_pct': abn_pct,
            })
        agent_metrics.sort(key=lambda x: -x['cnt'])
        agent_stats = [(am['agent'], am['lvl1'], am['cnt'], am['exc'], am['hloss'], am['mid_risk']) for am in agent_metrics]

        # 总体合计指标（口径与各代理完全一致）
        all_swap = df_agent_full[df_agent_full['近7d总换电次数'].fillna(0) > 0] if '近7d总换电次数' in df_agent_full.columns else df_agent_full
        _t_valid = df_agent_full.get('近7天有数据天数', pd.Series())
        _t_avg_valid = safe_mean(_t_valid)
        _t_raw_rate = df_agent_full.get('近7d出勤率', pd.Series()).fillna(0)
        _t_w7d = (_t_raw_rate * 7).clip(upper=7)
        _t_w7d[_t_valid.fillna(0) == 0] = 0
        _t_dev = df_agent_full[dev_col].apply(_parse_dev) if dev_col in df_agent_full.columns else pd.Series(dtype=str)
        total_m = {
            'cnt': agent_total, 'swap_n': len(all_swap),
            'avg_valid': _t_avg_valid, 'w7d_attend': safe_mean(_t_w7d), 'attend': safe_mean(_t_raw_rate) * 100,
            'w7d_active': int((_t_w7d >= 5).sum()), 'w7d_silent': int(((_t_w7d == 0) & (_t_valid.fillna(0) > 0)).sum()),
            'w7d_churn': int((_t_valid.fillna(0) == 0).sum()),
            'exc': fmt_pct((df_agent_full.get('用户等级_动态', pd.Series()) == '优质用户').sum(), agent_total),
            'good': fmt_pct((df_agent_full.get('用户等级_动态', pd.Series()) == '良好用户').sum(), agent_total),
            'hloss': fmt_pct((df_agent_full.get('用户等级_动态', pd.Series()) == '高损耗用户').sum(), agent_total),
            'mid_risk': fmt_pct((df_agent_full.get('用户等级_动态', pd.Series()) == '中风险用户').sum(), agent_total),
            'violent': fmt_pct((df_agent_full.get('用户等级_动态', pd.Series()) == '暴力').sum(), agent_total),
            'daily_km': safe_mean(df_agent_full.get('近7d总行驶距离_km', pd.Series())) / (_t_avg_valid if _t_avg_valid > 0 else 7),
            'med_km': safe_median(df_agent_full.get('近7d总行驶距离_km', pd.Series())) / (_t_avg_valid if _t_avg_valid > 0 else 7),
            'p90_speed': safe_mean(df_agent_full.get('近7d_P90骑行速度_kmh', pd.Series())),
            'daily_swaps': safe_mean(df_agent_full.get('近7d日均换电次数', pd.Series())),
            'soc_consum': safe_mean(all_swap.get('近7d单次换电SOC消耗', pd.Series())),
            'pick_soc': safe_mean(all_swap.get('近7d取电时平均SOC', pd.Series())),
            'return_soc': safe_mean(all_swap.get('近7d还电时平均SOC', pd.Series())),
            'friendly': safe_mean(all_swap.get('近7d换电友好分', pd.Series())),
            'night_pct': safe_mean(all_swap.get('近7d深夜换电占比', pd.Series())) * 100,
            'peak_pct': safe_mean(all_swap.get('近7d平峰换电占比', pd.Series())) * 100,
            'avg_cur': safe_mean(df_agent_full.get('近7d平均骑行电流_A', pd.Series())),
            'max_cur': safe_mean(df_agent_full.get('近7d最大电流_A', pd.Series())),
            'energy_100': safe_mean(df_agent_full.get('近7d百公里电耗_kWh', pd.Series())),
            'risk_pct': fmt_pct((df_agent_full['风险标签'].fillna('正常') != '正常').sum(), agent_total) if '风险标签' in df_agent_full.columns else (fmt_pct((df_agent_full['用户id'].map(risk_by_id).fillna('正常') != '正常').sum(), agent_total) if risk_by_id is not None else 0),
            'mod_pct': fmt_pct((df_agent_full.get('车辆形态_7d', pd.Series()) == '改装/超速车').sum(), agent_total),
            'online_pct': fmt_pct((_t_dev == '正常在线').sum(), agent_total) if len(_t_dev) > 0 else 0,
            'temp_pct': fmt_pct((_t_dev == '暂离线').sum(), agent_total) if len(_t_dev) > 0 else 0,
            'abn_pct': fmt_pct((_t_dev == '异常离线').sum(), agent_total) if len(_t_dev) > 0 else 0,
        }

        # ── 8.1 三级代理综合指标横向对比（与总体 1.1 核心看板口径完全一致）──
        w('### 8.1 三级代理综合指标横向对比')
        w()
        # 指标行定义: (显示名, 键, 格式化方式)  fmt: 'int'=整数, 'pct'=百分比, 'f1'=1位小数, 'f2'=2位小数, 'str'=字符串
        metric_rows = [
            ('一级', 'lvl1', 'str'),
            ('用户数', 'cnt', 'int'),
            ('换电用户数', 'swap_n', 'int'),
            ('人均有效数据天数', 'avg_valid', 'f1'),
            ('加权7天出勤(天)', 'w7d_attend', 'f1'),
            ('整体出勤率(%)', 'attend', 'f1'),
            ('活跃用户(≥5天)', 'w7d_active', 'int'),
            ('沉默用户', 'w7d_silent', 'int'),
            ('流失用户', 'w7d_churn', 'int'),
            ('日均骑行里程(km)', 'daily_km', 'f1'),
            ('里程中位数(km)', 'med_km', 'f1'),
            ('P90骑行速度(km/h)', 'p90_speed', 'f1'),
            ('日均换电次数', 'daily_swaps', 'f2'),
            ('单次SOC消耗(%)', 'soc_consum', 'f1'),
            ('取电时SOC(%)', 'pick_soc', 'f1'),
            ('还电时SOC(%)', 'return_soc', 'f1'),
            ('换电友好分', 'friendly', 'f1'),
            ('深夜换电占比(%)', 'night_pct', 'f1'),
            ('平峰换电占比(%)', 'peak_pct', 'f1'),
            ('平均骑行电流(A)', 'avg_cur', 'f2'),
            ('平均最大电流(A)', 'max_cur', 'f2'),
            ('百公里电耗(kWh)', 'energy_100', 'f2'),
            ('优质用户(%)', 'exc', 'pct'),
            ('良好用户(%)', 'good', 'pct'),
            ('高损耗用户(%)', 'hloss', 'pct'),
            ('中风险用户(%)', 'mid_risk', 'pct'),
            ('暴力用户(%)', 'violent', 'pct'),
            ('高风险占比(%)', 'risk_pct', 'pct'),
            ('改装/超速车(%)', 'mod_pct', 'pct'),
            ('设备正常在线(%)', 'online_pct', 'pct'),
            ('设备暂离线(%)', 'temp_pct', 'pct'),
            ('设备异常离线(%)', 'abn_pct', 'pct'),
        ]

        def _fmt_val(v, fmt, cnt=None, total=None):
            if fmt == 'str': return str(v)
            if fmt == 'int': return f'{int(v):,}'
            if fmt == 'pct': return f'{v:.1f}'
            if fmt == 'f1': return f'{v:.1f}'
            if fmt == 'f2': return f'{v:.2f}'
            return str(v)

        _cols = ['指标'] + [am['agent'] for am in agent_metrics] + ['总体']
        w('<table class="agent-compare">')
        w('<thead><tr>' + ''.join(f'<th>{c}</th>' for c in _cols) + '</tr></thead>')
        w('<tbody>')
        for label, key, fmt in metric_rows:
            row = [f'<td class="metric-name">{label}</td>']
            for am in agent_metrics:
                row.append(f'<td>{_fmt_val(am[key], fmt)}</td>')
            # 总体列
            if key == 'cnt':
                tv = f'{total_m["cnt"]:,}'
            elif key == 'swap_n':
                tv = f'{total_m["swap_n"]:,}'
            elif fmt == 'str':
                tv = '—'
            else:
                tv = _fmt_val(total_m[key], fmt)
            row.append(f'<td class="total-col"><strong>{tv}</strong></td>')
            w('<tr>' + ''.join(row) + '</tr>')
        w('</tbody></table>')
        w()

    else:
        w('> ⚠️ 未加载代理数据或无合资公司/自营用户，跳过本章。')
        w()

    # 关闭横向页面容器
    w('</div>')
    w()

    # ═══════════════ 第九章：价值分析（二八法则） ═══════════════
    w('---')
    w()
    w('## 九、用户价值分析（二八法则）')
    w()

    if '近7d总行驶距离_km' in df.columns:
        df_km = df.sort_values('近7d总行驶距离_km', ascending=False)
        total_km = df_km['近7d总行驶距离_km'].sum()
        top10_km = df_km.head(int(len(df_km) * 0.1))['近7d总行驶距离_km'].sum()
        top20_km = df_km.head(int(len(df_km) * 0.2))['近7d总行驶距离_km'].sum()
        w('### 9.1 骑行里程集中度')
        w()
        w('| 用户群体 | 骑行占比 | 平均日均里程 |')
        w('|----------|---------|------------|')
        w(f'| TOP 10% 用户 | {fmt_pct(top10_km, total_km):.1f}% | {safe_mean(df_km.head(int(len(df_km)*0.1)).get("近7d总行驶距离_km", pd.Series())) / daily_divisor:.1f} km |')
        w(f'| TOP 20% 用户 | {fmt_pct(top20_km, total_km):.1f}% | {safe_mean(df_km.head(int(len(df_km)*0.2)).get("近7d总行驶距离_km", pd.Series())) / daily_divisor:.1f} km |')
        w(f'| 全体 | 100% | {daily_km:.1f} km |')
        w()

    if df_att is not None and '单合约月度用电度数预估_kWh' in df_att.columns:
        df_en = df_att.sort_values('单合约月度用电度数预估_kWh', ascending=False)
        total_en = df_en['单合约月度用电度数预估_kWh'].sum()
        top10_en = df_en.head(int(len(df_en) * 0.1))['单合约月度用电度数预估_kWh'].sum()
        top20_en = df_en.head(int(len(df_en) * 0.2))['单合约月度用电度数预估_kWh'].sum()
        w('### 9.2 月度用电集中度')
        w()
        w('| 用户群体 | 用电占比 | 月均用电 |')
        w('|----------|---------|----------|')
        w(f'| TOP 10% 用户 | {fmt_pct(top10_en, total_en):.1f}% | {safe_mean(df_en.head(int(len(df_en)*0.1))["单合约月度用电度数预估_kWh"]):.1f} 度 |')
        w(f'| TOP 20% 用户 | {fmt_pct(top20_en, total_en):.1f}% | {safe_mean(df_en.head(int(len(df_en)*0.2))["单合约月度用电度数预估_kWh"]):.1f} 度 |')
        w(f'| 全体 | 100% | {monthly_energy_mean:.1f} 度 |')
        w()

    # ═══════════════ 第十章：设备状态 ═══════════════
    w('---')
    w()
    w('## 十、设备状态分析')
    w()
    w('> **说明**：`设备状态监控` = 以数据窗口最新统计日期为基准，距该日期的间隔天数。')
    w('> 两轮车换电场景下电池仅在上报数据时"在线"，大部分时间处于休眠。')
    w('> `正常在线`（≤1天无新数据）：近1天内有数据上报，正常活跃。')
    w('> `暂离线`（2~6天无新数据）：正常休眠/无换电，不代表异常。')
    w('> `异常离线`（≥7天无新数据）：长期无换电记录，需关注设备是否已回收/丢失。')
    w()
    w('### 10.1 设备离线天数分档')
    w()
    if len(status_detail) > 0:
        w('| 设备状态 | 人数 | 占比 |')
        w('|----------|------|------|')
        for label, cnt in status_detail.items():
            w(f'| {label} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% |')
        w()
        # 平均离线天数
        avg_offline_days = safe_mean(status_parsed['离线天数'])
        w(f'**平均离线天数**：{avg_offline_days:.1f} 天')
        w()
        w(f'- 正常在线用户 {online_total:,} 人（{fmt_pct(online_total, total_users):.1f}%），≤1天内有数据上报')
        w(f'- 暂离线用户 {temp_total:,} 人（{fmt_pct(temp_total, total_users):.1f}%），2~6天无数据，属正常休眠')
        w(f'- 异常离线用户 {abnormal_total:,} 人（{fmt_pct(abnormal_total, total_users):.1f}%），≥7天无数据记录，需关注')
        w()
    else:
        w('（无设备状态数据）')
        w()

    # ═══════════════ 第九章：策略建议 ═══════════════
    w('---')
    w()
    w('## 十一、策略建议分布')
    w()

    if '策略建议' in df.columns:
        strategy_counts = df['策略建议'].value_counts()
        w('| 策略建议 | 人数 | 占比 |')
        w('|----------|------|------|')
        for s, cnt in strategy_counts.items():
            w(f'| {s} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% |')
        w()

    # ═══════════════ 第十章：核心建议 ═══════════════
    w('---')
    w()
    w('## 十二、核心建议')
    w()

    # 动态分析
    # 动态分析：自动取高损耗率最高 和 优质率最高的城市
    top_high_loss_city = ''
    top_high_loss_pct = 0
    top_excellent_city = ''
    top_excellent_pct = 0
    if '核心活动城市' in df.columns and '用户等级_动态' in df.columns:
        city_stats = df.groupby('核心活动城市').agg(
            total=('用户等级_动态', 'count'),
            high_loss=('用户等级_动态', lambda x: (x == '高损耗用户').sum()),
            excellent=('用户等级_动态', lambda x: (x == '优质用户').sum())
        )
        city_stats = city_stats[city_stats['total'] >= 5]  # 至少5人
        if len(city_stats) > 0:
            city_stats['high_loss_pct'] = city_stats['high_loss'] / city_stats['total'] * 100
            city_stats['excellent_pct'] = city_stats['excellent'] / city_stats['total'] * 100
            top_hl = city_stats['high_loss_pct'].idxmax()
            top_high_loss_city = top_hl
            top_high_loss_pct = city_stats.loc[top_hl, 'high_loss_pct']
            top_ex = city_stats['excellent_pct'].idxmax()
            top_excellent_city = top_ex
            top_excellent_pct = city_stats.loc[top_ex, 'excellent_pct']

    ctype_modified = df[df['车辆形态_7d'] == '改装/超速车'] if '车辆形态_7d' in df.columns else pd.DataFrame()
    modified_cur = safe_mean(ctype_modified.get('近7d平均骑行电流_A', pd.Series())) if len(ctype_modified) > 0 else 0

    violent_max_cur = 0
    if '用户等级_动态' in df.columns:
        v_sub = df[df['用户等级_动态'] == '暴力']
        violent_max_cur = safe_mean(v_sub.get('近7d最大电流_A', pd.Series()))

    # 夜间高损耗
    night_high_loss = 0
    if '策略建议' in df.columns:
        night_high_loss = int((df['策略建议'] == '夜间高损耗预警').sum())

    w('### 12.1 立即行动（P0）')
    w()
    if top_high_loss_pct > 0:
        w(f'1. **{top_high_loss_city}区域深度排查**：该城市 {top_high_loss_pct:.0f}% 用户为高损耗，需逐户排查是否改装车/电池老化/运营政策过松')
    w(f'2. **{violent:,}名暴力用户清退**：平均最大电流 {violent_max_cur:.1f}A，远超安全线，建议限期整改否则终止服务')
    if deep_discharge > 0:
        w(f'3. **{deep_discharge:,}名深度放电用户引导**：还电SOC<10%，长期深度放电加速电池衰减，推送提前换电提醒')
    w()
    w('### 12.2 短期优化（P1）')
    w()
    if len(ctype_modified) > 0:
        w(f'4. **{len(ctype_modified):,}名改装/超速车核查**：平均电流 {modified_cur:.2f}A，人工确认是否改装车辆')
    if top_excellent_pct > 0 and top_high_loss_pct > 0:
        w(f'5. **建立{top_excellent_city}-{top_high_loss_city}对标学习**：将{top_excellent_city} {top_excellent_pct:.0f}% 优质率的运营方法复制到{top_high_loss_city}')
    if night_high_loss > 0:
        w(f'6. **{night_high_loss:,}名夜间高损耗用户预警**：夜间限流+电量阈值告警')
    w()
    w('### 12.3 中长期建设（P2）')
    w()
    w(f'7. **优质用户会员体系**：{excellent:,}名优质用户是基本盘，建立分级权益')
    w('8. **电池寿命预测模型**：基于电流/温度/SOC数据建立电池健康度评估')
    w('9. **区域差异化运营策略**：针对不同城市用户画像制定差异化阈值')
    w()

    # ═══════════════ 第十二章：指标说明 ═══════════════
    w('---')
    w()
    w('## 十三、指标说明')
    w()
    w('> 本附录说明报告中各核心指标的定义与计算口径，便于准确解读。')
    w()
    w('### 13.1 用户分层指标')
    w()
    w('| 指标 | 定义 |')
    w('|------|------|')
    w('| 用户等级_动态 | 综合电流/电耗/出勤/里程等多维度评分后动态划分：优质/良好/普通/高损耗/暴力/沉默/观察期 |')
    w('| 用户生命周期状态_7d | 基于合约状态与近7天出勤：新用户/活跃/轻度活跃/沉默/流失/已到期（本报告已剔除已到期） |')
    w('| 用户形态_综合_7d | 基于日均骑行时长、高峰占比、空间集中度与行为特征的骑手分类：专送/众包/地摊储能/数据不足 |')
    w('| 车辆形态_7d | 基于P95速度分位数：电动自行车≤25km/h、轻便摩托25~50km/h、电摩>50km/h、改装/超速车、地摊/储能 |')
    w('| 重评分数 | 用户综合评分，等级动态评定的核心依据 |')
    w()
    w('### 13.2 出勤与活跃指标')
    w()
    w('| 指标 | 定义 |')
    w('|------|------|')
    w('| 近7d出勤天数 | 近7天中有骑行活动的天数 |')
    w('| 近7d出勤率 | 出勤天数 ÷ 有效数据天数（非 /7） |')
    w('| 近7天有数据天数 | 近7天中有上报数据的日期数 |')
    w('| 加权7天出勤 | 近7d出勤率 × 7，将出勤行为按比例推算至满7天窗口，消除数据窗口不足偏差 |')
    w()
    w('### 13.3 换电与电池指标')
    w()
    w('| 指标 | 定义 |')
    w('|------|------|')
    w('| 近7d日均换电次数 | 近7天总换电次数 ÷ 有效数据天数 |')
    w('| 单次换电SOC消耗 | 单次换电周期内电池SOC的消耗幅度（取电SOC − 还电SOC） |')
    w('| 取电时平均SOC | 换电取走满电电池时刻的电池荷电量 |')
    w('| 还电时平均SOC | 骑行结束归还电池时刻的剩余荷电量，<10%为深度放电 |')
    w('| 近7d换电友好分 | 综合换电SOC健康度与时段合理性的评分，越高越规范 |')
    w('| 近7d深夜换电占比 | 22:00后换电次数占总换电次数比例 |')
    w('| 近7d最低SOC | 近7天观测到的最低电池荷电量 |')
    w()
    w('### 13.4 用电与能耗指标')
    w()
    w('| 指标 | 定义 |')
    w('|------|------|')
    w('| 单合约月度用电度数预估_kWh | 基于近7天日均用电量推算的单合约月度用电量 |')
    w('| 月用电盈亏线 | 150度，超过此值用户用电超出包月套餐，可能产生亏损 |')
    w('| 近7d百公里电耗_kWh | 每100公里消耗电量，衡量能耗效率 |')
    w('| 近7d平均骑行电流_A | 骑行过程中平均放电电流，反映用电强度 |')
    w('| 近7d最大电流_A | 骑行过程中峰值电流，过高提示暴力骑行/改装 |')
    w()
    w('### 13.5 骑行与空间行为指标')
    w()
    w('| 指标 | 定义 |')
    w('|------|------|')
    w('| 近7d总行驶距离_km | 近7天累计骑行里程 |')
    w('| 日均里程 | 近7d总行驶距离 ÷ 有效数据天数 |')
    w('| 近7d高峰骑行占比 | 午间/晚间高峰时段骑行时长占总骑行时长比例 |')
    w('| 近7d平均上线时间熵值 | 骑行时段分布的熵值，越低越规律（专送特征） |')
    w('| 近7d平均路线曲折系数 | 实际轨迹与直线距离比，越低越顺路（专送特征） |')
    w('| 近7d平均速度变异系数 | 速度波动程度，越低越稳定（专送特征） |')
    w('| 近7d_R90活动半径_km | 90%骑行点覆盖的半径，衡量活动范围 |')
    w('| 近7d凸包覆盖面积_km2 | 骑行轨迹凸包面积，衡量空间集中度 |')
    w()
    w('### 13.6 设备与风险指标')
    w()
    w('| 指标 | 定义 |')
    w('|------|------|')
    w('| 设备状态监控 | 以最新统计日为基准，距上次上报天数：正常在线(≤1天)/暂离线(2~6天)/异常离线(≥7天) |')
    w('| 风险标签 | 综合电流/速度/SOC等异常行为判定的风险等级，正常表示无显著风险 |')
    w('| 策略建议 | 基于用户画像自动生成的运营策略标签 |')
    w('| 夜间高速风险 | 夜间高速骑行行为的风险标识 |')
    w()

    # ── 页脚 ──
    w('---')
    w()
    w(f'> **数据说明**：本报告由 `generate_report.py` 自动生成于 {now}，仅含合约期内用户。')
    w(f'> 详细计算逻辑见 [报告逻辑及计算方法.md](报告逻辑及计算方法.md)。')

    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════
#  HTML / PDF 生成（Markdown → 精美 HTML → PDF）
# ═════════════════════════════════════════════════════════════

_REPORT_CSS = """
<style>
  :root {
    --primary: #1a4d72;
    --primary-light: #e8f1f8;
    --accent: #2c7a7b;
    --text: #2d3748;
    --muted: #718096;
    --border: #e2e8f0;
    --stripe: #f7fafc;
    --warn: #c53030;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Microsoft YaHei', 'PingFang SC', 'SimHei', sans-serif;
    color: var(--text);
    line-height: 1.4;
    margin: 0;
    padding: 18px 24px;
    font-size: 12px;
  }
  .report-title {
    text-align: center;
    font-size: 21px;
    color: var(--primary);
    font-weight: 700;
    padding: 14px 0 6px;
    margin: 0 0 4px;
    border-bottom: 3px solid var(--primary);
  }
  .meta {
    text-align: center;
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 6px;
  }
  h2 {
    font-size: 15px;
    color: #fff;
    background: var(--primary);
    padding: 6px 14px;
    border-radius: 4px;
    margin: 12px 0 8px;
    /* 内容自然流动分页，减少大面积留白；标题不孤立在页底 */
    page-break-after: avoid;
    break-after: avoid;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  h2:first-of-type {
    margin-top: 4px;
  }
  h3 {
    font-size: 13px;
    color: var(--primary);
    border-left: 4px solid var(--accent);
    padding-left: 9px;
    margin: 10px 0 5px;
    page-break-after: avoid;
    break-after: avoid;
  }
  p { margin: 4px 0; }
  strong { color: var(--primary); }
  blockquote {
    background: var(--primary-light);
    border-left: 3px solid var(--primary);
    margin: 6px 0;
    padding: 5px 10px;
    color: #4a5568;
    border-radius: 0 4px 4px 0;
    font-size: 11.5px;
  }
  blockquote p { margin: 2px 0; }
  code {
    background: #edf2f7;
    color: var(--warn);
    padding: 1px 5px;
    border-radius: 3px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
  }
  hr {
    border: none;
    border-top: 1px dashed var(--border);
    margin: 8px 0;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 11.5px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    /* 允许长表格跨页断行，避免大留白；行级保持不被切断 */
    page-break-inside: auto;
    break-inside: auto;
  }
  table.agent-compare {
    page-break-inside: auto;
    break-inside: auto;
  }
  /* 需整体保持不跨页的表格（如7.3城市对比表，避免末行溢出留白） */
  table.keep-together {
    page-break-inside: avoid;
    break-inside: avoid;
  }
  tr {
    page-break-inside: avoid;
    break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  th {
    background: var(--primary);
    color: #fff;
    font-weight: 600;
    padding: 5px 8px;
    text-align: center;
    border: 1px solid var(--primary);
    white-space: nowrap;
  }
  td {
    padding: 4px 8px;
    border: 1px solid var(--border);
    text-align: center;
  }
  tbody tr:nth-child(even) { background: var(--stripe); }
  tbody tr:hover { background: var(--primary-light); }
  td:first-child, th:first-child { text-align: left; }
  /* 代理横向对比表：列数多，固定布局+小字号自适应宽度 */
  table.agent-compare {
    width: 100%;
    table-layout: fixed;
    font-size: 8.5px;
    margin: 6px 0;
  }
  table.agent-compare th,
  table.agent-compare td {
    padding: 3px 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center;
    font-size: 8.5px;
  }
  table.agent-compare th {
    word-break: break-all;
    white-space: normal;
    line-height: 1.1;
  }
  table.agent-compare td.metric-name,
  table.agent-compare th.metric-name {
    width: 105px;
    min-width: 105px;
    text-align: left;
    font-weight: 600;
    color: var(--primary);
    white-space: normal;
    word-break: break-word;
    background: var(--primary-light);
  }
  table.agent-compare td.total-col {
    background: #e6f4ea;
  }
  table.agent-compare tbody tr:nth-child(even) { background: var(--stripe); }
  ul, ol { padding-left: 20px; margin: 6px 0; }
  li { margin: 2px 0; }
  @page { margin: 9mm 9mm; }
  /* 代理商横向对比章节：横向 A4，获得足够宽度容纳多列代理 */
  @page agent-landscape {
    size: A4 landscape;
    margin: 8mm 8mm;
  }
  .agent-chapter {
    page: agent-landscape;
  }
  /* 横向章节起始处强制分页，确保进入横向页面 */
  .agent-chapter h2 {
    page-break-before: always;
    break-before: page;
  }
  /* 横向章节内表格：指标行多，用更紧凑样式让整表放进单页 */
  .agent-chapter table.agent-compare {
    font-size: 8.5px;
    margin: 4px 0;
  }
  .agent-chapter table.agent-compare th,
  .agent-chapter table.agent-compare td {
    font-size: 8.5px;
    padding: 2px 3px;
    line-height: 1.15;
  }
</style>
"""


def md_to_html(md_content: str) -> str:
    """将 Markdown 转换为带精美样式的独立 HTML 文件内容"""
    if not _HTML_AVAILABLE:
        print("   ⚠️  markdown 库未安装，跳过 HTML 生成")
        return ''

    html_body = _md.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'sane_lists', 'attr_list', 'md_in_html'],
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>用户运营分析报告</title>
{_REPORT_CSS}
</head>
<body>
{html_body}
</body>
</html>"""


def html_to_pdf(html_content: str, pdf_path: str) -> bool:
    """用 Playwright (Chromium) 将 HTML 渲染为 PDF"""
    if not _PDF_AVAILABLE:
        print("   ⚠️  playwright 未安装，跳过 PDF 生成（HTML 仍可用）")
        return False

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content, wait_until='networkidle')
        # 等待中文字体就绪
        page.evaluate('document.fonts.ready')
        page.pdf(
                    path=pdf_path,
                    format='A4',
                    print_background=True,
                    margin={'top': '9mm', 'bottom': '9mm', 'left': '9mm', 'right': '9mm'},
                )
        browser.close()
    return True


# ═════════════════════════════════════════════════════════════
#  主入口
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='自动生成用户运营分析报告')
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT_DIR,
                        help=f'输入目录（默认: {DEFAULT_INPUT_DIR}）')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_FILE,
                        help=f'输出文件（默认: {DEFAULT_OUTPUT_FILE}）')
    args = parser.parse_args()

    print('=' * 60)
    print('  用户运营分析报告 - 自动生成')
    print('=' * 60)
    print(f'\n📂 输入目录: {args.input}')
    print()

    # 加载数据
    print('[1/3] 加载数据...')
    data = load_data(args.input)

    if 'detail' not in data:
        print('\n❌ 缺少核心数据文件，无法生成报告。')
        sys.exit(1)

    # 生成 Markdown 报告
    print(f'\n[2/3] 生成报告内容...')
    report = generate_report(data)

    # 写入 Markdown
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'✅ Markdown: {args.output} ({os.path.getsize(args.output):,} 字节)')

    # Markdown → 精美 HTML
    print(f'\n[3/3] 生成 HTML / PDF...')
    html_path = args.output.replace('.md', '.html')
    html_content = md_to_html(report)
    if html_content:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f'✅ HTML:    {html_path} ({os.path.getsize(html_path):,} 字节)')

        # HTML → PDF
        pdf_path = args.output.replace('.md', '.pdf')
        if html_to_pdf(html_content, pdf_path):
            print(f'✅ PDF:     {pdf_path} ({os.path.getsize(pdf_path):,} 字节)')

    print(f'\n   用户总数: {len(data["detail"]):,} 人（合约期内）')
    if data.get('agent') is not None and '一级' in data['agent'].columns:
        _sub = data['agent'][data['agent']['一级'].isin(['合资公司', '自营'])]
        print(f'   其中合资公司+自营: {len(_sub):,} 人（报告统计口径）')
    print()


if __name__ == '__main__':
    main()