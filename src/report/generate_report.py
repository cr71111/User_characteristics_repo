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

# PDF 生成依赖
try:
    from fpdf import FPDF
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# ── 路径配置 ────────────────────────────────────────────────
DATA_ROOT = r'E:\OneDrive\DataBase\DataBase\用户行为习惯\用户特征画像'
DEFAULT_INPUT_DIR = os.path.join(DATA_ROOT, 'reports')
DEFAULT_OUTPUT_FILE = os.path.join(DATA_ROOT, '用户分析报告', '用户运营分析报告.md')


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
    shift = data.get('shift', {})

    if df is None:
        return "# 错误：缺少 user_detail.csv 数据源"

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
    over_150 = int((df_att['单合约月度用电度数预估_kWh'] > 150).sum()) if df_att is not None else 0

    # 换电
    daily_swaps_mean = safe_mean(df_full.get('近7d日均换电次数', pd.Series())) if df_full is not None else 0

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
        """解析 '暂离线(6天)' / '异常(离线16天)' → (类型, 天数)"""
        s = str(s)
        m = re.search(r'(\d+)天', s)
        days = int(m.group(1)) if m else 0
        if '异常' in s:
            return ('异常离线', days)
        elif '暂离线' in s:
            return ('暂离线', days)
        elif '正常在线' in s:
            return ('正常在线', 0)
        else:
            return ('其他', 0)

    if '设备状态监控' in df.columns:
        status_parsed = df['设备状态监控'].apply(parse_device_status).apply(pd.Series)
        status_parsed.columns = ['状态类型', '离线天数']
        # 离线天数分档
        def bucket_days(d):
            if d <= 1: return '≤1天'
            if d <= 3: return '2-3天'
            if d <= 7: return '4-7天'
            if d <= 14: return '8-14天'
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
    w(f'| **平均骑行电流** | {avg_cur_mean:.2f} A | 中位 {avg_cur_med:.2f}A, P90={avg_cur_p90:.2f}A |')
    w(f'| **平均最大电流** | {max_cur_mean:.2f} A | 中位 {max_cur_med:.2f}A, P90={max_cur_p90:.2f}A |')
    w(f'| **百公里电耗均值** | {energy_100_mean:.2f} kWh | 中位 {energy_100_med:.2f}kWh, P90={energy_100_p90:.2f}kWh |')
    w(f'| **高风险用户占比** | {fmt_pct(risk_total, total_users):.1f}%（{risk_total:,}人） | 风险标签 ≠ 正常 |')
    w(f'| **优质用户占比** | {fmt_pct(excellent, total_users):.1f}%（{excellent:,}人） | 评分 ≥ P75 且无显著风险 |')
    w(f'| **高损耗用户占比** | {fmt_pct(high_loss, total_users):.1f}%（{high_loss:,}人） | 评分低或电池损耗高于均值 |')
    w(f'| **暴力用户** | {fmt_pct(violent, total_users):.1f}%（{violent:,}人） | 存在严重电池损害行为 |')
    w(f'| **已到期合约** | {fmt_pct(expired, total_users):.1f}%（{expired:,}人） | 合约到期需关注续费 |')
    w(f'| **人月均用电预估** | {monthly_energy_mean:.1f} 度 | 中位数 {monthly_energy_med:.1f}度, P90={monthly_energy_p90:.1f}度 |')
    w(f'| **月用电超标（>150度）** | {fmt_pct(over_150, len(df_att)):.1f}%（{over_150:,}人） | 超过盈亏线 |')
    # 设备状态汇总
    temp_total = int(status_parsed['状态类型'].eq('暂离线').sum()) if len(status_detail) > 0 else 0
    abnormal_total = int(status_parsed['状态类型'].eq('异常离线').sum()) if len(status_detail) > 0 else 0
    w(f'| **设备暂离线** | {temp_total:,} 人 ({fmt_pct(temp_total, total_users):.1f}%) | 1~6天无新数据记录，正常休眠态 |')
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

        # 等级 × 生命周期
        w('### 3.3 各等级生命周期分布')
        w()
        w('| 等级 | 总数 | 活跃 | 轻度活跃 | 已到期 |')
        w('|------|------|------|----------|--------|')
        for lv, cnt, sc, cur, mcur, en, km, ar in level_stats:
            sub = df[df['用户等级_动态'] == lv]
            lc = sub[lc_col].value_counts()
            w(f'| {lv} | {cnt:,} | {lc.get("活跃",0):,}({fmt_pct(lc.get("活跃",0),cnt):.0f}%) | {lc.get("轻度活跃",0):,}({fmt_pct(lc.get("轻度活跃",0),cnt):.0f}%) | {lc.get("已到期",0):,}({fmt_pct(lc.get("已到期",0),cnt):.0f}%) |')
        w()

    # ═══════════════ 第四章：风险标签 ═══════════════
    w('---')
    w()
    w('## 四、风险标签分析')
    w()

    if '风险标签' in df.columns:
        risk_dist = df['风险标签'].value_counts().head(20)
        w('### 4.1 风险标签 TOP 15')
        w()
        w('| 风险标签 | 人数 | 占比 |')
        w('|----------|------|------|')
        for tag, cnt in risk_dist.head(15).items():
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
        ctype_order_7d = ['专送骑手', '众包骑手', '标准骑手', '普通骑手', '地摊/储能', '数据不足']
        w('### 5.1 用户形态分布')
        w()
        w('| 用户形态 | 人数 | 占比 | 平均电流(A) | 百公里电耗(kWh) | 日均里程(km) | 平均评分 |')
        w('|----------|------|------|------------|----------------|------------|----------|')
        for ct in ctype_order_7d:
            sub = df[df['用户形态_综合_7d'] == ct]
            cnt = len(sub)
            if cnt > 0:
                w(f'| {ct} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% | {safe_mean(sub.get("近7d平均骑行电流_A", pd.Series())):.2f} | {safe_mean(sub.get("近7d百公里电耗_kWh", pd.Series())):.2f} | {safe_mean(sub.get("近7d日均行驶距离_km", pd.Series())):.2f} | {safe_mean(sub.get("重评分数", pd.Series())):.1f} |')
        w()

    if '车辆形态_7d' in df.columns:
        vtype_order_7d = ['电动自行车', '电动轻便摩托车', '电动摩托车', '改装/超速车', '数据不足']
        w('### 5.2 车辆形态分布')
        w()
        w('| 车辆形态 | 人数 | 占比 | 最高车速(km/h) | 峰值功率(W) | 日均里程(km) |')
        w('|----------|------|------|---------------|------------|------------|')
        for vt in vtype_order_7d:
            sub = df[df['车辆形态_7d'] == vt]
            cnt = len(sub)
            if cnt > 0:
                w(f'| {vt} | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% | {safe_mean(sub.get("近7d最高速度_kmh", pd.Series())):.1f} | {safe_mean(sub.get("近7d峰值功率_W", pd.Series())):.0f} | {safe_mean(sub.get("近7d日均行驶距离_km", pd.Series())):.2f} |')
        w(f'> 注：车辆形态基于近7天骑行数据判定，改装/超速车判定标准为最高速度≥50km/h且平均骑行电流≥24A，或峰值功率≥8000W，或持续大电流放电≥40A')
        w()

    # ═══════════════ 5.3 众包/专送判定特征 ═══════════════
    has_new_features = any(col in df.columns for col in ['近7d平均上线时间熵值', '近7d平均路线曲折系数', '近7d平均速度变异系数'])
    if has_new_features and '用户形态_综合_7d' in df.columns:
        w('### 5.3 众包/专送判定特征对比')
        w()
        w('| 特征指标 | 专送骑手 | 众包骑手 | 全体平均 | 文档标准参考 |')
        w('|----------|---------|---------|---------|-------------|')
        
        # 上线时间熵值
        if '近7d平均上线时间熵值' in df.columns:
            zhuan = df[df['用户形态_综合_7d'] == '专送骑手']
            zhong = df[df['用户形态_综合_7d'] == '众包骑手']
            w(f'| 上线时间熵值 | {safe_mean(zhuan.get("近7d平均上线时间熵值", pd.Series())):.2f} | {safe_mean(zhong.get("近7d平均上线时间熵值", pd.Series())):.2f} | {safe_mean(df.get("近7d平均上线时间熵值", pd.Series())):.2f} | 专送<3.5，众包>3.5 |')
        
        # 路线曲折系数
        if '近7d平均路线曲折系数' in df.columns:
            w(f'| 路线曲折系数 | {safe_mean(zhuan.get("近7d平均路线曲折系数", pd.Series())):.2f} | {safe_mean(zhong.get("近7d平均路线曲折系数", pd.Series())):.2f} | {safe_mean(df.get("近7d平均路线曲折系数", pd.Series())):.2f} | 专送<2.0，众包>2.0 |')
        
        # 速度变异系数
        if '近7d平均速度变异系数' in df.columns:
            w(f'| 速度变异系数 | {safe_mean(zhuan.get("近7d平均速度变异系数", pd.Series())):.2f} | {safe_mean(zhong.get("近7d平均速度变异系数", pd.Series())):.2f} | {safe_mean(df.get("近7d平均速度变异系数", pd.Series())):.2f} | 专送<0.4，众包>0.4 |')
        
        # 跨区域转移次数
        if '近7d最大跨区域转移次数' in df.columns:
            w(f'| 跨区域转移次数 | {safe_mean(zhuan.get("近7d最大跨区域转移次数", pd.Series())):.1f} | {safe_mean(zhong.get("近7d最大跨区域转移次数", pd.Series())):.1f} | {safe_mean(df.get("近7d最大跨区域转移次数", pd.Series())):.1f} | 专送<3，众包>3 |')
        
        # 静止时长占比
        if '近7d平均静止时长占比' in df.columns:
            w(f'| 静止时长占比 | {safe_mean(zhuan.get("近7d平均静止时长占比", pd.Series())):.2f} | {safe_mean(zhong.get("近7d平均静止时长占比", pd.Series())):.2f} | {safe_mean(df.get("近7d平均静止时长占比", pd.Series())):.2f} | 专送<0.5，众包>0.5 |')
        
        w()
        w('> 注：特征阈值基于《众包专送判断.md》文档标准，专送骑手表现为规律性强、活动集中、路线顺路、速度稳定')
        w()

    # ═══════════════ 第六章：区域分析 ═══════════════
    w('---')
    w()
    w('## 六、区域运营分析')
    w()

    if '核心活动省份' in df.columns:
        w('### 6.1 主要省份对比')
        w()
        prov_counts = df['核心活动省份'].value_counts().head(10)
        w('| 省份 | 总用户 | 优质率 | 高损耗率 |')
        w('|------|--------|--------|---------|')
        for prov, cnt in prov_counts.items():
            sub = df[df['核心活动省份'] == prov]
            exc = fmt_pct((sub['用户等级_动态'] == '优质用户').sum(), cnt)
            hloss = fmt_pct((sub['用户等级_动态'] == '高损耗用户').sum(), cnt)
            w(f'| {prov} | {cnt:,} | {exc:.0f}% | {hloss:.0f}% |')
        w()

    if '核心活动城市' in df.columns:
        w('### 6.2 主要城市对比')
        w()
        city_counts = df['核心活动城市'].value_counts().head(10)
        w('| 城市 | 总用户 | 优质率 | 高损耗率 |')
        w('|------|--------|--------|---------|')
        for city, cnt in city_counts.items():
            sub = df[df['核心活动城市'] == city]
            exc = fmt_pct((sub['用户等级_动态'] == '优质用户').sum(), cnt)
            hloss = fmt_pct((sub['用户等级_动态'] == '高损耗用户').sum(), cnt)
            w(f'| {city} | {cnt:,} | {exc:.0f}% | {hloss:.0f}% |')
        w()

    # ═══════════════ 第七章：价值分析（二八法则） ═══════════════
    w('---')
    w()
    w('## 七、用户价值分析（二八法则）')
    w()

    if '近7d总行驶距离_km' in df.columns:
        df_km = df.sort_values('近7d总行驶距离_km', ascending=False)
        total_km = df_km['近7d总行驶距离_km'].sum()
        top10_km = df_km.head(int(len(df_km) * 0.1))['近7d总行驶距离_km'].sum()
        top20_km = df_km.head(int(len(df_km) * 0.2))['近7d总行驶距离_km'].sum()
        w('### 7.1 骑行里程集中度')
        w()
        w('| 用户群体 | 骑行占比 | 平均日均里程 |')
        w('|----------|---------|------------|')
        w(f'| TOP 10% 用户 | {fmt_pct(top10_km, total_km):.1f}% | {safe_mean(df_km.head(int(len(df_km)*0.1)).get("近7d日均行驶距离_km", pd.Series())):.1f} km |')
        w(f'| TOP 20% 用户 | {fmt_pct(top20_km, total_km):.1f}% | {safe_mean(df_km.head(int(len(df_km)*0.2)).get("近7d日均行驶距离_km", pd.Series())):.1f} km |')
        w(f'| 全体 | 100% | {daily_km:.1f} km |')
        w()

    if df_att is not None and '单合约月度用电度数预估_kWh' in df_att.columns:
        df_en = df_att.sort_values('单合约月度用电度数预估_kWh', ascending=False)
        total_en = df_en['单合约月度用电度数预估_kWh'].sum()
        top10_en = df_en.head(int(len(df_en) * 0.1))['单合约月度用电度数预估_kWh'].sum()
        top20_en = df_en.head(int(len(df_en) * 0.2))['单合约月度用电度数预估_kWh'].sum()
        w('### 7.2 月度用电集中度')
        w()
        w('| 用户群体 | 用电占比 | 月均用电 |')
        w('|----------|---------|----------|')
        w(f'| TOP 10% 用户 | {fmt_pct(top10_en, total_en):.1f}% | {safe_mean(df_en.head(int(len(df_en)*0.1))["单合约月度用电度数预估_kWh"]):.1f} 度 |')
        w(f'| TOP 20% 用户 | {fmt_pct(top20_en, total_en):.1f}% | {safe_mean(df_en.head(int(len(df_en)*0.2))["单合约月度用电度数预估_kWh"]):.1f} 度 |')
        w(f'| 全体 | 100% | {monthly_energy_mean:.1f} 度 |')
        w()

    # ═══════════════ 第八章：设备状态 ═══════════════
    w('---')
    w()
    w('## 八、设备状态分析')
    w()
    w('> **说明**：`设备状态监控` = 以数据窗口最新统计日期为基准，距该日期的间隔天数。')
    w('> 两轮车换电场景下电池仅在上报数据时"在线"，大部分时间处于休眠。')
    w('> `暂离线`（1~6天无新数据）：正常休眠/无换电，不代表异常。')
    w('> `异常离线`（≥7天无新数据）：长期无换电记录，需关注设备是否已回收/丢失。')
    w()
    w('### 8.1 设备离线天数分档')
    w()
    if len(status_detail) > 0:
        w('| 状态类型 | 离线天数 | 人数 | 占比 |')
        w('|----------|---------|------|------|')
        for label, cnt in status_detail.items():
            w(f'| {label} | | {cnt:,} | {fmt_pct(cnt, total_users):.1f}% |')
        w()
        # 平均离线天数
        avg_offline_days = safe_mean(status_parsed['离线天数'])
        w(f'**平均离线天数**：{avg_offline_days:.1f} 天')
        w()
        w(f'- 暂离线用户 {temp_total:,} 人（{fmt_pct(temp_total, total_users):.1f}%），最近1~6天无数据，属正常休眠')
        w(f'- 异常离线用户 {abnormal_total:,} 人（{fmt_pct(abnormal_total, total_users):.1f}%），≥7天无数据记录，需关注')
        w()
    else:
        w('（无设备状态数据）')
        w()

    # ═══════════════ 第九章：策略建议 ═══════════════
    w('---')
    w()
    w('## 九、策略建议分布')
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
    w('## 十、核心建议')
    w()

    # 动态分析
    shanghai_high_loss_pct = 0
    if '核心活动城市' in df.columns and '用户等级_动态' in df.columns:
        sh_sub = df[df['核心活动城市'] == '上海市']
        if len(sh_sub) > 0:
            shanghai_high_loss_pct = fmt_pct((sh_sub['用户等级_动态'] == '高损耗用户').sum(), len(sh_sub))

    yichang_excellent_pct = 0
    if '核心活动城市' in df.columns and '用户等级_动态' in df.columns:
        yc_sub = df[df['核心活动城市'] == '宜昌市']
        if len(yc_sub) > 0:
            yichang_excellent_pct = fmt_pct((yc_sub['用户等级_动态'] == '优质用户').sum(), len(yc_sub))

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

    w('### 10.1 立即行动（P0）')
    w()
    if shanghai_high_loss_pct > 0:
        w(f'1. **上海区域深度排查**：上海 {shanghai_high_loss_pct:.0f}% 用户为高损耗，需逐户排查是否改装车/电池老化/运营政策过松')
    w(f'2. **{violent:,}名暴力用户清退**：平均最大电流 {violent_max_cur:.1f}A，远超安全线，建议限期整改否则终止服务')
    if expired > 0:
        w(f'3. **{expired:,}名到期用户续费**：{fmt_pct(int((df[df["用户生命周期状态_7d"]=="已到期"]["_状态类型"]=="异常离线").sum()), expired):.0f}% 已异常离线，需尽快推出续费优惠方案')
    w()
    w('### 10.2 短期优化（P1）')
    w()
    if len(ctype_modified) > 0:
        w(f'4. **{len(ctype_modified):,}名改装/超速车核查**：平均电流 {modified_cur:.2f}A，人工确认是否改装车辆')
    if yichang_excellent_pct > 0 and shanghai_high_loss_pct > 0:
        w(f'5. **建立宜昌-上海对标学习**：将宜昌 {yichang_excellent_pct:.0f}% 优质率的运营方法复制到上海地区')
    if night_high_loss > 0:
        w(f'6. **{night_high_loss:,}名夜间高损耗用户预警**：夜间限流+电量阈值告警')
    w()
    w('### 10.3 中长期建设（P2）')
    w()
    w(f'7. **优质用户会员体系**：{excellent:,}名优质用户是基本盘，建立分级权益')
    w('8. **电池寿命预测模型**：基于电流/温度/SOC数据建立电池健康度评估')
    w('9. **区域差异化运营策略**：针对不同城市用户画像制定差异化阈值')
    w()

    # ── 页脚 ──
    w('---')
    w()
    w(f'> **数据说明**：本报告由 `generate_report.py` 自动生成于 {now}。')
    w(f'> 详细计算逻辑见 [报告逻辑及计算方法.md](报告逻辑及计算方法.md)。')

    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════
#  PDF 生成
# ═════════════════════════════════════════════════════════════

# 中文字体路径
_CJK_FONT_PATH = r'C:\Windows\Fonts\simhei.ttf'


def _strip_inline_md(text: str) -> str:
    """去除行内 Markdown 格式标记"""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.+?\)', '', text)
    return text


def _render_table(pdf: FPDF, header: list, rows: list, col_widths: list = None):
    """用 fpdf2 原生 table() 渲染表格"""
    n_cols = len(header)
    if not col_widths:
        col_widths = [190 / n_cols] * n_cols

    pdf.set_font('SimHei', '', 8)
    col_widths = tuple(col_widths)

    with pdf.table(
        first_row_as_headings=True,
        borders_layout='ALL',
        text_align='CENTER',
        col_widths=col_widths,
        width=190,
    ) as table:
        row = table.row()
        for h in header:
            row.cell(h)
        for data_row in rows:
            row = table.row()
            for j in range(n_cols):
                row.cell(str(data_row[j]) if j < len(data_row) else '')


def _md_to_pdf(md_content: str, pdf_path: str) -> bool:
    """将 Markdown 内容直接渲染为 PDF（fpdf2 原生 API，无 HTML 中间层）"""
    if not _PDF_AVAILABLE:
        print("   ⚠️  fpdf2 未安装，跳过 PDF 生成")
        return False

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.add_font('SimHei', '', _CJK_FONT_PATH)
    pdf.add_font('SimHei', 'B', _CJK_FONT_PATH)
    pdf.add_font('SimHei', 'I', _CJK_FONT_PATH)
    pdf.add_font('SimHei', 'BI', _CJK_FONT_PATH)

    lines = md_content.split('\n')
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 水平分隔线
        if stripped == '---':
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y() + 1, 200, pdf.get_y() + 1)
            pdf.ln(5)
            i += 1
            continue

        # 块引用
        if line.startswith('> '):
            bq_lines = []
            while i < n and lines[i].startswith('> '):
                bq_lines.append(lines[i][2:].strip())
                i += 1
            text = _strip_inline_md(' '.join(bq_lines))
            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(80, 80, 80)
            pdf.set_font('SimHei', '', 9)
            x0 = pdf.get_x()
            pdf.set_x(x0 + 4)
            pdf.multi_cell(182, 5, text, fill=True)
            pdf.ln(2)
            pdf.set_text_color(0, 0, 0)
            continue

        # 一级标题
        if stripped.startswith('# ') and not stripped.startswith('## '):
            pdf.set_font('SimHei', 'B', 18)
            pdf.ln(4)
            pdf.multi_cell(0, 10, _strip_inline_md(stripped[2:]), align='C')
            pdf.ln(2)
            i += 1
            continue

        # 二级标题
        if stripped.startswith('## ') and not stripped.startswith('### '):
            pdf.set_font('SimHei', 'B', 14)
            pdf.ln(3)
            y0 = pdf.get_y()
            pdf.multi_cell(0, 8, _strip_inline_md(stripped[3:]))
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y() + 1, 200, pdf.get_y() + 1)
            pdf.ln(3)
            i += 1
            continue

        # 三级标题
        if stripped.startswith('### '):
            pdf.set_font('SimHei', 'B', 12)
            pdf.ln(2)
            pdf.multi_cell(0, 7, _strip_inline_md(stripped[4:]))
            pdf.ln(2)
            i += 1
            continue

        # Markdown 表格
        if stripped.startswith('|') and '|' in stripped:
            header = [c.strip() for c in stripped.split('|')[1:-1]]
            i += 1
            if i < n and lines[i].strip().startswith('|') and '---' in lines[i]:
                i += 1
            rows = []
            while i < n and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].split('|')[1:-1]]
                rows.append(row)
                i += 1

            # 计算列宽
            n_cols = len(header)
            max_w = [pdf.get_string_width(h) + 4 for h in header]
            for row in rows:
                for j in range(min(n_cols, len(row))):
                    w = pdf.get_string_width(str(row[j])) + 4
                    if w > max_w[j]:
                        max_w[j] = w

            total_w = sum(max_w)
            if total_w > 190:
                scale = 190 / total_w
                col_widths = [w * scale for w in max_w]
            else:
                col_widths = max_w

            _render_table(pdf, header, rows, col_widths)
            pdf.ln(4)
            continue

        # 空行
        if not stripped:
            pdf.ln(3)
            i += 1
            continue

        # 普通段落 / 列表项
        para_lines = []
        while (i < n and lines[i].strip()
               and not lines[i].strip().startswith('#')
               and not lines[i].strip().startswith('|')
               and not lines[i].startswith('> ')
               and lines[i].strip() != '---'):
            para_lines.append(lines[i].strip())
            i += 1

        if para_lines:
            text = _strip_inline_md(' '.join(para_lines))
            pdf.set_font('SimHei', '', 10)
            pdf.multi_cell(0, 5.5, text)
            pdf.ln(1)

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    pdf.output(pdf_path)
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
    print('[1/2] 加载数据...')
    data = load_data(args.input)

    if 'detail' not in data:
        print('\n❌ 缺少核心数据文件，无法生成报告。')
        sys.exit(1)

    # 生成报告
    print(f'\n[2/2] 生成报告...')
    report = generate_report(data)

    # 写入
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)

    file_size = os.path.getsize(args.output)
    print(f'\n✅ 报告已生成: {args.output}')
    print(f'   文件大小: {file_size:,} 字节')

    # PDF 导出
    pdf_path = args.output.replace('.md', '.pdf')
    if _md_to_pdf(report, pdf_path):
        pdf_size = os.path.getsize(pdf_path)
        print(f'✅ PDF 已生成:  {pdf_path}')
        print(f'   文件大小: {pdf_size:,} 字节')

    print(f'   用户总数: {len(data["detail"]):,} 人')
    print()


if __name__ == '__main__':
    main()