# -*- coding: utf-8 -*-
"""
数据流水线模块
实现：原始数据 → 每日汇总 → 每日快照 → 生命周期档案 → 报告
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from User_characteristics_repo.config.config import (
    BASE_EXPORT_PATH,
    EXPORT_PATH_BATTERY_STATUS_30D,
    EXPORT_PATH_BATTERY_STATUS_3D,
    EXPORT_PATH_USER_BEHAVIOR,
    EXPORT_PATH_DAILY_USER_DATA_RAW,
    EXPORT_PATH_DAILY_SNAPSHOT,
    EXPORT_FILE_USER_LIFECYCLE_7D,
    EXPORT_FILE_USER_DAILY_SNAPSHOT
)


class DataPipeline:
    """数据流水线管理器"""
    
    def __init__(self, base_path: str = BASE_EXPORT_PATH):
        self.base_path = base_path
        
        # 原始数据路径
        self.raw_data_paths = {
            "历史30天-前4天": os.path.join(base_path, EXPORT_PATH_BATTERY_STATUS_30D),
            "近3天-昨天": os.path.join(base_path, EXPORT_PATH_BATTERY_STATUS_3D)
        }
        
        # 每日汇总数据路径（Parquet，level3压缩）
        self.daily_summary_dir = os.path.join(base_path, EXPORT_PATH_DAILY_USER_DATA_RAW)
        os.makedirs(self.daily_summary_dir, exist_ok=True)
        
        # 每日快照数据路径（Parquet+CSV）
        self.daily_snapshot_dir = os.path.join(base_path, EXPORT_PATH_DAILY_SNAPSHOT)
        os.makedirs(self.daily_snapshot_dir, exist_ok=True)
        
        # 生命周期档案路径（Parquet+CSV）
        self.lifecycle_dir = os.path.join(base_path, EXPORT_PATH_USER_BEHAVIOR)
        os.makedirs(self.lifecycle_dir, exist_ok=True)
        
        # 报告路径（Markdown）
        self.report_dir = os.path.join(base_path, EXPORT_PATH_USER_BEHAVIOR, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
    
    def stage1_raw_to_daily_summary(self, target_date: Optional[str] = None) -> str:
        """
        阶段1：原始数据 → 每日用户汇总（Parquet，level3压缩）
        不更改任何数据，只是把每天的数据汇总起来，方便存档
        """
        print("\n" + "="*80)
        print("📊 阶段1：原始数据 → 每日用户汇总（Parquet，level3压缩）")
        print("="*80)
        
        # 收集所有原始数据文件
        all_files = []
        for folder_name, folder_path in self.raw_data_paths.items():
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        all_files.append(os.path.join(folder_path, file))
        
        if not all_files:
            print("❌ 未找到原始数据文件")
            return ""
        
        print(f"📂 找到 {len(all_files)} 个原始数据文件")
        
        # 读取并汇总数据
        dfs = []
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ 读取文件失败 {file_path}: {e}")
        
        if not dfs:
            print("❌ 无法读取任何原始数据文件")
            return ""
        
        # 合并所有数据
        df_raw = pd.concat(dfs, ignore_index=True)
        print(f"📊 合并后数据量: {len(df_raw)} 行")
        
        # 按日期分组
        if '统计日期' in df_raw.columns:
            df_raw['统计日期'] = pd.to_datetime(df_raw['统计日期']).dt.strftime('%Y-%m-%d')
            
            # 如果指定了目标日期，只处理该日期
            if target_date:
                df_raw = df_raw[df_raw['统计日期'] == target_date]
            
            # 按日期保存汇总数据
            for date, df_day in df_raw.groupby('统计日期'):
                output_path = os.path.join(self.daily_summary_dir, f"daily_summary_{date}.parquet")
                df_day.to_parquet(
                    output_path, 
                    engine='pyarrow',
                    compression='snappy'
                )
                print(f"✅ 保存每日汇总: {output_path} ({len(df_day)} 行)")
            
            return self.daily_summary_dir
        else:
            print("❌ 原始数据缺少'统计日期'列")
            return ""
    
    def stage2_daily_summary_to_snapshot(self, target_date: Optional[str] = None) -> str:
        """
        阶段2：每日用户汇总 → 每日数据快照（Parquet+CSV）
        从每日汇总数据中得到每日数据快照
        """
        print("\n" + "="*80)
        print("📸 阶段2：每日用户汇总 → 每日数据快照（Parquet+CSV）")
        print("="*80)
        
        # 获取所有每日汇总文件
        summary_files = []
        for file in os.listdir(self.daily_summary_dir):
            if file.endswith('.parquet'):
                if target_date and target_date in file:
                    summary_files.append(os.path.join(self.daily_summary_dir, file))
                elif not target_date:
                    summary_files.append(os.path.join(self.daily_summary_dir, file))
        
        if not summary_files:
            print("❌ 未找到每日汇总文件")
            return ""
        
        print(f"📂 找到 {len(summary_files)} 个每日汇总文件")
        
        # 处理每个日期的汇总数据
        for summary_path in summary_files:
            try:
                df_summary = pd.read_parquet(summary_path)
                
                # 提取日期
                date_str = os.path.basename(summary_path).replace('daily_summary_', '').replace('.parquet', '')
                
                # 生成快照数据
                if '用户id' in df_summary.columns:
                    df_snapshot = df_summary.groupby('用户id').agg({
                        '电流': ['mean', 'max', 'min'],
                        '速度': ['mean', 'max'],
                        '电池SOC': ['mean', 'min'],
                        '电池度数': ['sum', 'mean'],
                        '是否在线': 'max'
                    }).reset_index()
                    
                    df_snapshot.columns = ['用户id', '平均电流', '最大电流', '最小电流', 
                                          '平均速度', '最大速度', '平均SOC', '最低SOC',
                                          '总电池度数', '平均电池度数', '最后在线状态']
                    
                    # 添加统计日期
                    df_snapshot['统计日期'] = date_str
                    
                    # 保存为Parquet
                    parquet_path = os.path.join(self.daily_snapshot_dir, f"daily_snapshot_{date_str}.parquet")
                    df_snapshot.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
                    
                    # 保存为CSV
                    csv_path = os.path.join(self.daily_snapshot_dir, f"daily_snapshot_{date_str}.csv")
                    df_snapshot.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    
                    print(f"✅ 保存每日快照: {parquet_path} ({len(df_snapshot)} 用户)")
                
            except Exception as e:
                print(f"⚠️ 处理文件失败 {summary_path}: {e}")
        
        return self.daily_snapshot_dir
    
    def stage3_snapshot_to_lifecycle(self, target_date: Optional[str] = None) -> str:
        """
        阶段3：每日数据快照 → 生命周期档案（Parquet+CSV）
        调用用户生命周期管理逻辑
        """
        print("\n" + "="*80)
        print("🔄 阶段3：每日数据快照 → 生命周期档案（Parquet+CSV）")
        print("="*80)
        
        # 导入用户生命周期管理模块
        try:
            # 添加当前目录到路径
            sys.path.insert(0, os.path.dirname(__file__))
            from User_characteristics_repo.src.用户生命周期管理 import main as lifecycle_main
            
            # 运行生命周期管理，传入目标日期参数
            lifecycle_main(target_date=target_date)
            
            # 读取生成的CSV并转换为Parquet
            lifecycle_csv = os.path.join(self.base_path, EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_LIFECYCLE_7D)
            if os.path.exists(lifecycle_csv):
                df_lifecycle = pd.read_csv(lifecycle_csv, encoding='utf-8-sig')
                
                # 保存为Parquet
                parquet_path = os.path.join(self.lifecycle_dir, "user_lifecycle.parquet")
                df_lifecycle.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
                
                print(f"✅ 保存生命周期档案: {parquet_path} ({len(df_lifecycle)} 用户)")
                
                return self.lifecycle_dir
            else:
                print("❌ 未找到生命周期档案CSV文件")
                return ""
                
        except ImportError as e:
            print(f"⚠️  用户生命周期管理模块导入失败: {e}")
            print("   使用基础生命周期计算")
            
            # 基础生命周期计算（备用方案）
            snapshot_files = []
            for file in os.listdir(self.daily_snapshot_dir):
                if file.endswith('.parquet'):
                    if target_date and target_date in file:
                        snapshot_files.append(os.path.join(self.daily_snapshot_dir, file))
                    elif not target_date:
                        snapshot_files.append(os.path.join(self.daily_snapshot_dir, file))
            
            if not snapshot_files:
                print("❌ 未找到每日快照文件")
                return ""
            
            # 读取所有快照数据
            dfs = []
            for snapshot_path in snapshot_files:
                try:
                    df_snapshot = pd.read_parquet(snapshot_path)
                    dfs.append(df_snapshot)
                except Exception as e:
                    print(f"⚠️ 读取快照失败 {snapshot_path}: {e}")
            
            if not dfs:
                print("❌ 无法读取任何快照文件")
                return ""
            
            # 合并所有快照数据
            df_all_snapshots = pd.concat(dfs, ignore_index=True)
            
            # 生成生命周期档案
            if '用户id' in df_all_snapshots.columns and '统计日期' in df_all_snapshots.columns:
                df_all_snapshots['统计日期'] = pd.to_datetime(df_all_snapshots['统计日期'])
                
                # 按用户计算生命周期指标
                df_lifecycle = df_all_snapshots.groupby('用户id').agg({
                    '平均电流': 'mean',
                    '最大电流': 'max',
                    '平均速度': 'mean',
                    '平均SOC': 'mean',
                    '总电池度数': 'sum',
                    '统计日期': ['min', 'max', 'count']
                }).reset_index()
                
                df_lifecycle.columns = ['用户id', '7日平均电流', '7日最大电流', '7日平均速度',
                                       '7日平均SOC', '7日总电池度数', '首次使用日期', '最后使用日期', '活跃天数']
                
                # 保存为Parquet
                parquet_path = os.path.join(self.lifecycle_dir, "user_lifecycle.parquet")
                df_lifecycle.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
                
                # 保存为CSV
                csv_path = os.path.join(self.lifecycle_dir, "user_lifecycle.csv")
                df_lifecycle.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                print(f"✅ 保存生命周期档案: {parquet_path} ({len(df_lifecycle)} 用户)")
                
                return self.lifecycle_dir
            else:
                print("❌ 快照数据缺少必要列")
                return ""
    
    def stage4_lifecycle_to_report(self) -> str:
        """
        阶段4：生命周期档案 → 报告（Markdown）
        """
        print("\n" + "="*80)
        print("📝 阶段4：生命周期档案 → 报告（Markdown）")
        print("="*80)
        
        # 读取生命周期档案
        lifecycle_path = os.path.join(self.lifecycle_dir, "user_lifecycle.parquet")
        if not os.path.exists(lifecycle_path):
            print("❌ 未找到生命周期档案")
            return ""
        
        df_lifecycle = pd.read_parquet(lifecycle_path)
        
        # 生成报告
        report_path = os.path.join(self.report_dir, f"user_lifecycle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 用户生命周期分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基础统计
            f.write("## 基础统计\n\n")
            f.write(f"- **总用户数**: {len(df_lifecycle)}\n")
            f.write(f"- **平均活跃天数**: {df_lifecycle['活跃天数'].mean():.1f}\n")
            f.write(f"- **7日平均电流**: {df_lifecycle['7日平均电流'].mean():.2f}A\n")
            f.write(f"- **7日平均SOC**: {df_lifecycle['7日平均SOC'].mean():.1f}%\n\n")
            
            # 用户分群
            f.write("## 用户分群\n\n")
            
            # 按电流分群
            high_current = len(df_lifecycle[df_lifecycle['7日平均电流'] > 30])
            normal_current = len(df_lifecycle[(df_lifecycle['7日平均电流'] >= 20) & (df_lifecycle['7日平均电流'] <= 30)])
            low_current = len(df_lifecycle[df_lifecycle['7日平均电流'] < 20])
            
            f.write("### 按电流分群\n\n")
            f.write(f"- **高电流用户** (>30A): {high_current} ({high_current/len(df_lifecycle)*100:.1f}%)\n")
            f.write(f"- **正常电流用户** (20-30A): {normal_current} ({normal_current/len(df_lifecycle)*100:.1f}%)\n")
            f.write(f"- **低电流用户** (<20A): {low_current} ({low_current/len(df_lifecycle)*100:.1f}%)\n\n")
            
            # 按SOC分群
            high_soc = len(df_lifecycle[df_lifecycle['7日平均SOC'] > 50])
            normal_soc = len(df_lifecycle[(df_lifecycle['7日平均SOC'] >= 30) & (df_lifecycle['7日平均SOC'] <= 50)])
            low_soc = len(df_lifecycle[df_lifecycle['7日平均SOC'] < 30])
            
            f.write("### 按SOC分群\n\n")
            f.write(f"- **高SOC用户** (>50%): {high_soc} ({high_soc/len(df_lifecycle)*100:.1f}%)\n")
            f.write(f"- **正常SOC用户** (30-50%): {normal_soc} ({normal_soc/len(df_lifecycle)*100:.1f}%)\n")
            f.write(f"- **低SOC用户** (<30%): {low_soc} ({low_soc/len(df_lifecycle)*100:.1f}%)\n\n")
            
            # 活跃度分析
            f.write("## 活跃度分析\n\n")
            active_users = len(df_lifecycle[df_lifecycle['活跃天数'] >= 5])
            normal_users = len(df_lifecycle[(df_lifecycle['活跃天数'] >= 3) & (df_lifecycle['活跃天数'] < 5)])
            inactive_users = len(df_lifecycle[df_lifecycle['活跃天数'] < 3])
            
            f.write(f"- **高活跃用户** (≥5天): {active_users} ({active_users/len(df_lifecycle)*100:.1f}%)\n")
            f.write(f"- **正常活跃用户** (3-5天): {normal_users} ({normal_users/len(df_lifecycle)*100:.1f}%)\n")
            f.write(f"- **低活跃用户** (<3天): {inactive_users} ({inactive_users/len(df_lifecycle)*100:.1f}%)\n\n")
            
            # 建议
            f.write("## 建议\n\n")
            f.write("1. 关注高电流用户，可能存在电池损耗风险\n")
            f.write("2. 关注低SOC用户，可能需要优化换电策略\n")
            f.write("3. 关注低活跃用户，可能存在流失风险\n")
        
        print(f"✅ 保存报告: {report_path}")
        return report_path
    
    def run_full_pipeline(self, target_date: Optional[str] = None):
        """运行完整流水线"""
        print("🚀 开始运行完整数据流水线")
        
        # 阶段1：原始数据 → 每日汇总
        stage1_result = self.stage1_raw_to_daily_summary(target_date)
        if not stage1_result:
            print("❌ 阶段1失败，终止流水线")
            return
        
        # 阶段2：每日汇总 → 每日快照
        stage2_result = self.stage2_daily_summary_to_snapshot(target_date)
        if not stage2_result:
            print("❌ 阶段2失败，终止流水线")
            return
        
        # 阶段3：每日快照 → 生命周期档案
        stage3_result = self.stage3_snapshot_to_lifecycle(target_date)
        if not stage3_result:
            print("❌ 阶段3失败，终止流水线")
            return
        
        # 阶段4：生命周期档案 → 报告
        stage4_result = self.stage4_lifecycle_to_report()
        if not stage4_result:
            print("❌ 阶段4失败，终止流水线")
            return
        
        print("\n" + "="*80)
        print("✅ 完整数据流水线运行完成")
        print("="*80)
        print(f"📊 每日汇总数据: {stage1_result}")
        print(f"📸 每日快照数据: {stage2_result}")
        print(f"🔄 生命周期档案: {stage3_result}")
        print(f"📝 分析报告: {stage4_result}")


if __name__ == "__main__":
    # 创建流水线实例
    pipeline = DataPipeline()
    
    # 运行完整流水线（可以指定target_date来处理特定日期）
    pipeline.run_full_pipeline(target_date=None)
