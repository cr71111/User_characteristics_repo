# -*- coding: utf-8 -*-
"""
L0 Raw Ingest - 原始数据摄入层（超高性能版）
职责：把原始 CSV 转成干净的 Parquet，不做任何业务推断。

性能优化策略：
1. 批处理：每100个文件为一批，避免内存爆炸
2. 按日期拆分写入：每个批次直接按日期写入临时文件，避免主进程groupby
3. 最终合并：只需简单concat同一天文件，无需复杂操作
4. 列选择：只读取需要的列，减少内存占用

输入：IoT CSV 文件
输出：raw/{date}.parquet
"""

import os
import sys
import time
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple

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

# 预计算在线状态的有效值集合（小写）
VALID_ONLINE_VALUES_SET = {'在线', '1', 'true', 'online', '是'}

CHINA_LAT_MIN, CHINA_LAT_MAX = 3, 54
CHINA_LON_MIN, CHINA_LON_MAX = 73, 136

# 并行和批处理配置
MAX_WORKERS = min(8, os.cpu_count() or 4)  # 最多8个进程
BATCH_SIZE = 100  # 每批处理的文件数


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """按 COLUMN_NORMALIZE_MAP 统一字段名"""
    rename_map = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in COLUMN_NORMALIZE_MAP:
            rename_map[col] = COLUMN_NORMALIZE_MAP[col_stripped]
    return df.rename(columns=rename_map)


def get_existing_dates(output_dir: str) -> set:
    """获取已存在的日期parquet文件"""
    existing = set()
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith('.parquet'):
                date_str = f.replace('.parquet', '')
                existing.add(date_str)
    return existing


def quick_scan_file_dates(file_path: str, max_rows: int = 50) -> set:
    """
    快速扫描文件前几行，提取包含的日期集合
    
    Args:
        file_path: CSV文件路径
        max_rows: 最大扫描行数
        
    Returns:
        该文件包含的日期集合
    """
    dates = set()
    try:
        df = pd.read_csv(file_path, encoding='utf-8', nrows=max_rows, low_memory=False)
        
        if len(df) == 0:
            return dates
            
        df = normalize_column_names(df)
        
        if '统计日期' in df.columns:
            date_series = pd.to_datetime(df['统计日期'], errors='coerce').dt.strftime('%Y-%m-%d')
            dates = set(date_series.dropna().unique())
        elif '时间戳' in df.columns:
            ts_col = pd.to_numeric(df['时间戳'], errors='coerce').dropna()
            if len(ts_col) > 0:
                date_series = pd.to_datetime(ts_col, unit='s', errors='coerce').dt.strftime('%Y-%m-%d')
                dates = set(date_series.dropna().unique())
                
    except Exception:
        pass
    
    return dates


def filter_files_by_existing_dates(all_files: List[str], existing_dates: set) -> Tuple[List[str], Dict[str, int]]:
    """
    根据已有日期过滤文件列表
    
    Args:
        all_files: 所有原始文件路径
        existing_dates: 已存在的日期集合
        
    Returns:
        (需要处理的文件列表, {日期: 文件数} 统计)
    """
    if not existing_dates:
        return all_files, {}
    
    files_to_process = []
    date_file_count = {}
    skipped_by_date = {}
    unknown_dates = []
    
    print(f"🔍 快速扫描文件日期信息（跳过已有数据的日期）...")
    scan_start = time.time()
    
    for i, file_path in enumerate(all_files):
        file_dates = quick_scan_file_dates(file_path)
        
        if not file_dates:
            unknown_dates.append(file_path)
            files_to_process.append(file_path)
        else:
            relevant_dates = file_dates - existing_dates
            if relevant_dates:
                files_to_process.append(file_path)
                for d in relevant_dates:
                    date_file_count[d] = date_file_count.get(d, 0) + 1
            else:
                for d in file_dates:
                    skipped_by_date[d] = skipped_by_date.get(d, 0) + 1
        
        if (i + 1) % 1000 == 0 or i == len(all_files) - 1:
            elapsed = time.time() - scan_start
            print(f"   📂 扫描进度: {i+1}/{len(all_files)} ({elapsed:.1f}s)")
    
    scan_time = time.time() - scan_start
    total_skipped = sum(skipped_by_date.values())
    
    print(f"\n✅ 文件过滤完成:")
    print(f"   📊 总文件数: {len(all_files)}")
    print(f"   ⏭️  跳过已有数据: {total_skipped} 个文件")
    print(f"   ✅ 需要处理: {len(files_to_process)} 个文件")
    print(f"   ⏱️  扫描耗时: {scan_time:.2f}s")
    
    if skipped_by_date:
        print(f"   📅 按日期跳过详情:")
        for d, count in sorted(skipped_by_date.items()):
            print(f"      - {d}: {count} 个文件")
    
    return files_to_process, date_file_count


def process_batch(file_batch: List[str], temp_dir: str, batch_idx: int, existing_dates: set = None) -> Optional[List[str]]:
    """
    处理一批CSV文件，按日期拆分写入临时parquet文件
    
    Args:
        file_batch: 文件路径列表
        temp_dir: 临时文件目录
        batch_idx: 批次索引
        existing_dates: 已存在的日期集合（跳过这些日期）
    
    Returns:
        生成的临时文件路径列表
    """
    if existing_dates is None:
        existing_dates = set()
    
    try:
        # 按日期收集数据
        date_data: Dict[str, List[pd.DataFrame]] = {}
        
        for file_path in file_batch:
            try:
                # 读取所有原始字段（不做列过滤，保留全部原始数据）
                df = pd.read_csv(
                    file_path, 
                    encoding='utf-8', 
                    low_memory=False
                )
                
                if len(df) == 0:
                    continue
                
                # 字段标准化
                df = normalize_column_names(df)
                
                # 检查核心字段
                missing_cols = [c for c in REQUIRED_COLUMNS_RAW if c not in df.columns]
                if missing_cols:
                    continue
                
                # 在线状态过滤
                if '是否在线' in df.columns:
                    online_col = df['是否在线'].astype(str).str.strip().str.lower()
                    mask = online_col.isin(VALID_ONLINE_VALUES_SET)
                    df = df[mask]
                    
                    if len(df) == 0:
                        continue
                
                # 批量数值转换
                numeric_cols = ['时间戳', '纬度', '经度', '电流', '温度', '速度']
                cols_to_convert = [c for c in numeric_cols if c in df.columns]
                if cols_to_convert:
                    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
                
                # 中国境内粗过滤
                df = df[
                    (df['纬度'] >= CHINA_LAT_MIN) & (df['纬度'] <= CHINA_LAT_MAX) &
                    (df['经度'] >= CHINA_LON_MIN) & (df['经度'] <= CHINA_LON_MAX)
                ]
                
                if len(df) == 0:
                    continue
                
                # 提取统计日期
                if '统计日期' not in df.columns:
                    df['统计日期'] = pd.to_datetime(df['时间戳'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d')
                else:
                    df['统计日期'] = pd.to_datetime(df['统计日期'], errors='coerce').dt.strftime('%Y-%m-%d')
                
                # 确保类型
                df['用户id'] = df['用户id'].astype(str)
                df['合约id'] = df['合约id'].astype(str)
                df['时间戳'] = df['时间戳'].astype('int64')
                
                # 按日期分组收集（跳过已存在的日期）
                for date_str, df_day in df.groupby('统计日期'):
                    if date_str in existing_dates:
                        continue  # 跳过已存在的日期
                    if date_str not in date_data:
                        date_data[date_str] = []
                    date_data[date_str].append(df_day)
                
            except Exception:
                continue
        
        # 按日期写入临时文件
        temp_files = []
        for date_str, df_list in date_data.items():
            df_date = pd.concat(df_list, ignore_index=True)
            temp_file = os.path.join(temp_dir, f"{date_str}_batch{batch_idx}.parquet")
            df_date.to_parquet(temp_file, engine='pyarrow', compression='snappy')
            temp_files.append(temp_file)
        
        return temp_files
        
    except Exception as e:
        print(f"⚠️ 批次 {batch_idx} 处理失败: {e}")
        return None


def ingest_raw_data(target_date: Optional[str] = None, base_path: str = BASE_EXPORT_PATH) -> dict:
    """
    执行 L0 原始数据摄入（超高性能版）

    Args:
        target_date: 目标日期 YYYY-MM-DD，None 则处理所有日期
        base_path: 基础导出路径

    Returns:
        dict: {date_str: parquet_path} 已处理的文件映射
    """
    print("\n" + "=" * 80)
    print("L0 Raw Ingest - 原始数据摄入层（超高性能版）")
    print("=" * 80)
    
    start_time = time.time()

    raw_data_paths = {
        "历史30天-前4天": os.path.join(base_path, EXPORT_PATH_BATTERY_STATUS_30D),
        "近3天-昨天": os.path.join(base_path, EXPORT_PATH_BATTERY_STATUS_3D),
    }

    output_dir = EXPORT_PATH_RAW
    os.makedirs(output_dir, exist_ok=True)

    # 获取已存在的日期（跳过已处理的日期）
    existing_dates = get_existing_dates(output_dir)
    if existing_dates:
        print(f"📋 已存在 {len(existing_dates)} 个日期的数据: {sorted(existing_dates)}")

    # 收集所有原始数据文件
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

    total_files = len(all_files)
    print(f"📂 找到 {total_files} 个原始数据文件")

    # 根据已有日期过滤文件（避免处理已有数据的日期）
    if existing_dates:
        all_files, date_file_count = filter_files_by_existing_dates(all_files, existing_dates)
        
        if not all_files:
            print("✅ 所有日期数据已存在，无需处理")
            return {d: os.path.join(output_dir, f"{d}.parquet") for d in existing_dates}
        
        total_files = len(all_files)
    
    print(f"⚡ 使用 {MAX_WORKERS} 个并行进程，每批 {BATCH_SIZE} 个文件")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="l0_ingest_")
    print(f"📁 临时目录: {temp_dir}")
    
    try:
        # 分批处理
        batches = [all_files[i:i+BATCH_SIZE] for i in range(0, total_files, BATCH_SIZE)]
        total_batches = len(batches)
        print(f"📦 分为 {total_batches} 批处理\n")
        
        batch_start = time.time()
        date_temp_files: Dict[str, List[str]] = {}
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_batch, batch, temp_dir, i, existing_dates): i for i, batch in enumerate(batches)}
            
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    temp_files = future.result()
                    if temp_files:
                        # 按日期收集临时文件
                        for temp_file in temp_files:
                            # 从文件名提取日期: 2026-03-20_batch0.parquet
                            date_str = os.path.basename(temp_file).split('_batch')[0]
                            if date_str not in date_temp_files:
                                date_temp_files[date_str] = []
                            date_temp_files[date_str].append(temp_file)
                    
                    # 进度显示
                    elapsed = time.time() - batch_start
                    speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                    eta = (total_batches - batch_idx - 1) / speed if speed > 0 else 0
                    print(f"   📊 批次 {batch_idx+1}/{total_batches} 完成 ({speed:.1f} 批/秒, 预计剩余 {eta:.0f}s)")
                    
                except Exception as e:
                    print(f"⚠️ 批次 {batch_idx} 处理失败: {e}")
        
        print(f"\n✅ 所有批次完成，共 {len(date_temp_files)} 个待处理日期，耗时 {time.time()-batch_start:.2f}s")
        
        if not date_temp_files:
            print("✅ 所有日期数据已存在，无需处理")
            return {d: os.path.join(output_dir, f"{d}.parquet") for d in existing_dates}
        
        # 合并每个日期的临时文件
        merge_start = time.time()
        print("🔄 开始合并每个日期的数据...")
        
        processed = {}
        for date_str, temp_files in date_temp_files.items():
            if target_date and date_str != target_date:
                continue

            output_path = os.path.join(output_dir, f"{date_str}.parquet")

            # 幂等检查（双重保险）
            if os.path.exists(output_path):
                print(f"⏭️  L0 已存在，跳过: {output_path}")
                processed[date_str] = output_path
                continue

            # 读取并合并同一天的所有临时文件
            df_list = []
            for temp_file in temp_files:
                df_temp = pd.read_parquet(temp_file)
                df_list.append(df_temp)
            
            df_merged = pd.concat(df_list, ignore_index=True)
            
            # 去重
            df_merged = df_merged.drop_duplicates(subset=['合约id', '用户id', '时间戳'], keep='last')

            df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
            print(f"✅ L0 保存: {output_path} ({len(df_merged):,} 行, 合并 {len(temp_files)} 个批次)")
            processed[date_str] = output_path
        
        print(f"\n✅ L0 Raw Ingest 完成")
        print(f"   处理日期数: {len(processed)}")
        print(f"   总耗时: {time.time()-start_time:.2f}s")
        print(f"     - 批处理: {batch_start-start_time:.2f}s")
        print(f"     - 合并保存: {time.time()-merge_start:.2f}s")
        
        return processed
        
    finally:
        # 清理临时文件
        print(f"🧹 清理临时目录...")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    ingest_raw_data(target_date=None)
