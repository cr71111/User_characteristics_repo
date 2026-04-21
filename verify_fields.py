import pandas as pd
import numpy as np

print("=" * 80)
print("输出字段验证报告")
print("=" * 80)

# 验证L1 Fact Layer输出
print("\n【L1 Fact Layer 输出验证】")
df_fact = pd.read_parquet(r"E:\test\data\fact\daily\2026-04-15.parquet")
print(f"总记录数: {len(df_fact):,}")
print(f"字段数: {len(df_fact.columns)}")

# 检查关键字段覆盖率
fact_key_fields = [
    '统计日期', '合约id', '用户id', '行驶距离', '骑行总耗时(小时)',
    '放电总耗时(小时)', '怠速放电耗时(小时)', '平均骑行放电比',
    '用电量(kWh)', '骑行次数', '换电次数', '午间高峰长时骑行次数',
    '晚间高峰长时骑行次数', '平峰长时骑行次数', '夜间长时骑行次数',
    '午间高峰里程_km', '晚间高峰里程_km', '平峰里程_km', '夜间里程_km',
    '电流>60A次数', '电流>80A次数', '超100A连续次数', '超100A累计时长_h',
    '最高速度', '最大电流', '最高温度', '最低SOC', '包月友好评分',
    '活动半径_km', '客户形态', '用户等级', '用户等级说明', '客户形态说明'
]

print("\n关键字段覆盖率:")
for col in fact_key_fields:
    if col in df_fact.columns:
        non_null = df_fact[col].notna().sum()
        non_zero = (df_fact[col] != 0).sum()
        print(f"  ✅ {col}: 非空{non_null:,} ({non_null/len(df_fact)*100:.1f}%), 非零{non_zero:,} ({non_zero/len(df_fact)*100:.1f}%)")
    else:
        print(f"  ❌ {col}: 缺失")

# 验证L2 Snapshot Layer输出
print("\n【L2 Snapshot Layer 输出验证】")
df_snap = pd.read_parquet(r"E:\test\data\snapshot\user_daily.parquet")
print(f"总记录数: {len(df_snap):,}")
print(f"字段数: {len(df_snap.columns)}")

snap_key_fields = [
    '统计日期', '用户id', '关联合约数', '当日总行驶距离_km', '当日总骑行时长_h',
    '当日总放电时长_h', '当日总怠速放电时长_h', '当日平均骑行放电比',
    '当日总用电量_kWh', '当日总骑行次数', '当日总换电次数',
    '单合约日均行驶里程_km', '单合约日均骑行时长_h', '单合约日均放电时长_h',
    '单合约日均怠速放电_h', '单合约日均用电量_kWh',
    '午间高峰长时骑行总次数', '晚间高峰长时骑行总次数', '平峰长时骑行总次数',
    '夜间长时骑行总次数', '午间高峰总里程_km', '晚间高峰总里程_km',
    '平峰总里程_km', '夜间总里程_km', '总电流超60A次数', '总电流超80A次数',
    '总超100A连续次数', '总超100A累计时长_h', '当日最高速度_kmh',
    '当日最大电流_A', '当日最高温度_℃', '当日最低SOC', '最低包月友好分',
    '最大单合约活动半径_km', '当日是否出勤', '平均包月友好分',
    '当日平均骑行SOC', '当日平均骑行速度_kmh', '当日平均骑行电流_A',
    '当日百公里电耗_kWh', '高峰骑行占比', 'SOC低于20%时长占比_当日',
    'SOC低于10%时长占比_当日', '核心活动省份', '核心活动城市', '核心活动区县',
    '风险标签_电流异常', '客户形态_综合', '用户等级_综合',
    '用户等级_综合说明', '客户形态_综合说明'
]

print("\n关键字段覆盖率:")
for col in snap_key_fields:
    if col in df_snap.columns:
        non_null = df_snap[col].notna().sum()
        non_zero = (df_snap[col] != 0).sum()
        print(f"  ✅ {col}: 非空{non_null:,} ({non_null/len(df_snap)*100:.1f}%), 非零{non_zero:,} ({non_zero/len(df_snap)*100:.1f}%)")
    else:
        print(f"  ❌ {col}: 缺失")

# 验证L3 Lifecycle Layer输出
print("\n【L3 Lifecycle Layer 输出验证】")
df_life = pd.read_parquet(r"E:\test\data\lifecycle\user_7d.parquet")
print(f"总记录数: {len(df_life):,}")
print(f"字段数: {len(df_life.columns)}")

life_key_fields = [
    '统计日期', '用户id', '近7d总行驶距离_km', '近7d总骑行时长_h',
    '近7d总放电时长_h', '近7d总怠速放电时长_h', '近7d平均骑行放电比',
    '近7d总用电量_kWh', '近7d总骑行次数', '近7d总换电次数',
    '近7d单合约日均行驶里程_km', '近7d单合约日均骑行时长_h',
    '近7d单合约日均放电时长_h', '近7d单合约日均怠速放电_h',
    '近7d单合约日均用电量_kWh', '近7d午间高峰长时骑行总次数',
    '近7d晚间高峰长时骑行总次数', '近7d平峰长时骑行总次数',
    '近7d夜间长时骑行总次数', '近7d午间高峰总里程_km',
    '近7d晚间高峰总里程_km', '近7d平峰总里程_km', '近7d夜间总里程_km',
    '近7d总电流超60A次数', '近7d总电流超80A次数', '近7d总超100A连续次数',
    '近7d总超100A累计时长_h', '近7d最高速度_kmh', '近7d最大电流_A',
    '近7d最高温度_℃', '近7d最低SOC', '近7d最低包月友好分',
    '近7d最大单合约活动半径_km', '近7d出勤天数', '近7d平均包月友好分',
    '近7d平均骑行SOC', '近7d平均骑行速度_kmh', '近7d平均骑行电流_A',
    '近7d百公里电耗_kWh', '近7d高峰骑行占比', '近7d平均SOC低于20%时长占比',
    '近7d平均SOC低于10%时长占比', '近7d核心活动省份', '近7d核心活动城市',
    '近7d核心活动区县', '近7d风险标签_电流异常', '客户形态_综合_7d',
    '用户等级_综合_7d', '客户形态_综合_7d说明', '设备状态监控',
    '月度用电预估_kWh', '用户生命周期状态', '用户综合评分',
    '评分等级', '风险标签', '动态阈值_电流', '动态阈值_速度',
    '分布漂移检测'
]

print("\n关键字段覆盖率:")
for col in life_key_fields:
    if col in df_life.columns:
        non_null = df_life[col].notna().sum()
        non_zero = (df_life[col] != 0).sum()
        print(f"  ✅ {col}: 非空{non_null:,} ({non_null/len(df_life)*100:.1f}%), 非零{non_zero:,} ({non_zero/len(df_life)*100:.1f}%)")
    else:
        print(f"  ❌ {col}: 缺失")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)
