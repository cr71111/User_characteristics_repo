# -*- coding: utf-8 -*-
"""
L0 Raw Ingest - 原始数据摄入层
职责：把原始 CSV 转成干净的 Parquet，不做任何业务推断。

输入：IoT CSV 文件
输出：raw/{date}.parquet
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import (
    EXPORT_PATH_BATTERY_STATUS_30D,
    EXPORT_PATH_BATTERY_STATUS_3D,
    BASE_EXPORT_PATH,
    EXPORT_PATH_RAW,
)

# 字段标准化映射
COLUMN_NORMALIZE_MAP = {
    '统计日期': '统计日期',
    '合约id': '合约id',
    '用户id': '用户id',
    '时间戳': '时间戳',
    '纬度': '纬度',
    '经度': '经度',
    '电流': '电流',
    '温度': '温度',
    '速度': '速度',
    '电池SOC': '电池SOC',
    '电池度数': '电池度数',
    '电池id': '电池id',
    '是否在线': '是否在线',
    '中心纬度': '中心纬度',
    '中心经度': '中心经度',
    '用电量_kWh': '用电量_kWh',
    '换电标记': '换电标记',
}

REQUIRED_COLUMNS_RAW = ['时间戳', '合约id', '用户id', '纬度', '经度', '电流', '温度', '速度', '电池SOC']

VALID_ONLINE_VALUES = ['在线', '1', 'true', 'online', '是']

CHINA_LAT_MIN, CHINA_LAT_MAX = 3, 54
CHINA_LON_MIN, CHINA_LON_MAX = 73, 136


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """按 COLUMN_NORMALIZE_MAP 统一字段名"""
    rename_map = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in COLUMN_NORMALIZE_MAP:
            rename_map[col] = COLUMN_NORMALIZE_MAP[col_stripped]
    return df.rename(columns=rename_map)


def ingest_raw_data(target_date: Optional[str] = None, base_path: str = BASE_EXPORT_PATH) -> dict:
    """
    执行 L0 原始数据摄入

    Args:
        target_date: 目标日期 YYYY-MM-DD，None 则处理所有日期
        base_path: 基础导出路径

    Returns:
        dict: {date_str: parquet_path} 已处理的文件映射
    """
    print("\n" + "=" * 80)
    print("L0 Raw Ingest - 原始数据摄入层")
    print("=" * 80)

    raw_data_paths = {
        "历史30天-前4天": os.path.join(base_path, EXPORT_PATH_BATTERY_STATUS_30D),
        "近3天-昨天": os.path.join(base_path, EXPORT_PATH_BATTERY_STATUS_3D),
    }

    output_dir = EXPORT_PATH_RAW
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有原始数据文件（递归扫描日期子目录）
    all_files = []
    for folder_name, folder_path in raw_data_paths.items():
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.csv'):
                        all_files.append(os.path.join(root, file))

    if not all_files:
        print("❌ 未找到原始数据文件")
        return {}

    print(f"📂 找到 {len(all_files)} 个原始数据文件")

    processed = {}

    # 先按日期分组收集所有数据
    date_data = {}
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except Exception as e:
            print(f"⚠️ 读取文件失败 {file_path}: {e}")
            continue

        # 字段标准化
        df = normalize_column_names(df)

        # 检查核心字段
        missing_cols = [c for c in REQUIRED_COLUMNS_RAW if c not in df.columns]
        if missing_cols:
            print(f"⚠️ 文件 {file_path} 缺少核心字段: {missing_cols}")
            continue

        # 过滤离线记录
        if '是否在线' in df.columns:
            df = df[df['是否在线'].astype(str).str.lower().isin([v.lower() for v in VALID_ONLINE_VALUES])]

        # 数值转换
        for col in ['时间戳', '纬度', '经度', '电流', '温度', '速度']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 中国境内粗过滤
        df = df[
            (df['纬度'] >= CHINA_LAT_MIN) & (df['纬度'] <= CHINA_LAT_MAX) &
            (df['经度'] >= CHINA_LON_MIN) & (df['经度'] <= CHINA_LON_MAX)
        ]

        # 提取统计日期
        if '统计日期' not in df.columns:
            df['统计日期'] = pd.to_datetime(df['时间戳'], unit='s').dt.strftime('%Y-%m-%d')
        else:
            df['统计日期'] = pd.to_datetime(df['统计日期']).dt.strftime('%Y-%m-%d')

        # 确保类型
        df['用户id'] = df['用户id'].astype(str)
        df['合约id'] = df['合约id'].astype(str)
        df['时间戳'] = df['时间戳'].astype('int64')

        # 按日期收集
        for date_str, df_day in df.groupby('统计日期'):
            if date_str not in date_data:
                date_data[date_str] = []
            date_data[date_str].append(df_day)

    # 合并同一天所有数据并保存
    for date_str, df_list in date_data.items():
        if target_date and date_str != target_date:
            continue

        output_path = os.path.join(output_dir, f"{date_str}.parquet")

        # 幂等检查
        if os.path.exists(output_path):
            print(f"⏭️  L0 已存在，跳过: {output_path}")
            processed[date_str] = output_path
            continue

        # 合并同一天所有文件的数据
        df_merged = pd.concat(df_list, ignore_index=True)

        # 去重（合并后统一去重）
        df_merged = df_merged.drop_duplicates(subset=['合约id', '用户id', '时间戳'], keep='last')

        df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
        print(f"✅ L0 保存: {output_path} ({len(df_merged)} 行, 合并 {len(df_list)} 个文件)")
        processed[date_str] = output_path

    print(f"\n✅ L0 Raw Ingest 完成，处理 {len(processed)} 个日期")
    return processed


if __name__ == "__main__":
    ingest_raw_data(target_date=None)
