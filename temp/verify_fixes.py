import pandas as pd
import numpy as np

print("=" * 80)
print("🔍 数据质量修复验证报告")
print("=" * 80)

# 读取新生成的数据（优先使用parquet文件，字段更完整）
import os
DATA_OUTPUT_ROOT = r'E:\test\用户行为习惯/用户特征画像/data'

# 优先尝试parquet文件（包含完整字段）
parquet_file = os.path.join(DATA_OUTPUT_ROOT, 'lifecycle', 'user_7d.parquet')
if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
    print(f"\n✅ 读取新数据: {parquet_file}")
else:
    # 尝试CSV报告
    report_file = os.path.join(DATA_OUTPUT_ROOT, 'reports', 'user_detail.csv')
    if os.path.exists(report_file):
        df = pd.read_csv(report_file)
        print(f"\n✅ 读取新数据: {report_file}")
    else:
        # 回退到旧文件
        df = pd.read_csv('d:/PY代码/用户特征画像/User_characteristics_repo/user_7d.csv')
        print(f"\n⚠️  读取旧数据（未找到新数据）")
print(f"\n总用户数: {len(df)}")

# 验证1: 评分是否在0-100范围内
print("\n" + "-" * 60)
print("✅ 验证1: 评分范围 (0-100)")
print("-" * 60)
score_issues = df[(df['重评分数'] > 100) | (df['重评分数'] < 0)]
if len(score_issues) == 0:
    print("🎉 PASS - 所有评分都在0-100范围内")
    print(f"   最大值: {df['重评分数'].max()}, 最小值: {df['重评分数'].min()}")
else:
    print(f"❌ FAIL - 仍有 {len(score_issues)} 个用户评分超出范围")
    print(f"   最大值: {df['重评分数'].max()}, 最小值: {df['重评分数'].min()}")

# 验证2: 换电友好分（无换电记录时应为0）
print("\n" + "-" * 60)
print("✅ 验证2: 换电友好分 (无换电记录时=0)")
print("-" * 60)
no_swap = df[df['近7d总换电次数'] == 0]
no_swap_with_score = no_swap[no_swap['近7d换电友好分'] > 0]
if len(no_swap_with_score) == 0:
    print("🎉 PASS - 所有无换电记录的用户换电友好分=0")
else:
    print(f"❌ FAIL - 仍有 {len(no_swap_with_score)} 个用户无换电但友好分>0")
    print(f"   示例: 用户ID={no_swap_with_score['用户id'].iloc[0]}, "
          f"换电次数={no_swap_with_score['近7d总换电次数'].iloc[0]}, "
          f"友好分={no_swap_with_score['近7d换电友好分'].iloc[0]}")

# 验证3: 最早骑行时刻（有骑行数据时不为-1）
print("\n" + "-" * 60)
print("✅ 验证3: 骑行时刻完整性 (有骑行数据时!=-1)")
print("-" * 60)
has_ride = df[df['近7d单合约日均行驶里程_km'] > 0]
has_ride_minus1 = has_ride[has_ride['近7d最早骑行时刻_h'] == -1]
if len(has_ride_minus1) == 0:
    print("🎉 PASS - 所有有骑行数据的用户都有有效的最早骑行时刻")
else:
    print(f"⚠️  PARTIAL - 仍有 {len(has_ride_minus1)} 个用户 ({len(has_ride_minus1)/len(has_ride)*100:.1f}%) 缺少骑行时刻")
    print(f"   (这可能是历史数据，需要重新运行L1层才能修复)")

# 验证4: P50速度（有骑行数据时>0）
print("\n" + "-" * 60)
print("✅ 验证4: P50速度合理性 (有平均速度时P50应>0)")
print("-" * 60)
has_avg_speed = df[df['近7d平均骑行速度_kmh'] > 0]
p50_zero_has_speed = has_avg_speed[has_avg_speed['近7d_P50骑行速度_kmh'] == 0]
if len(p50_zero_has_speed) == 0:
    print("🎉 PASS - 所有有平均速度的用户的P50速度>0")
else:
    print(f"⚠️  PARTIAL - 仍有 {len(p50_zero_has_speed)} 个用户 ({len(p50_zero_has_speed)/len(has_avg_speed)*100:.1f}%) P50=0")
    print(f"   (这可能是历史数据，需要重新运行L1层才能修复)")

# 验证5: 完全体画像完整性
print("\n" + "-" * 60)
print("✅ 验证5: 完全体画像完整性 (包含【骑行时间】维度)")
print("-" * 60)
missing_time = df[~df['用户完全体画像'].str.contains('【骑行时间】', na=False)]
has_ride_missing_time = missing_time[missing_time['近7d单合约日均行驶里程_km'] > 0]
if len(has_ride_missing_time) == 0:
    print("🎉 PASS - 所有有骑行数据的用户完全体画像包含【骑行时间】维度")
else:
    print(f"⚠️  PARTIAL - 仍有 {len(has_ride_missing_time)} 个用户缺少【骑行时间】维度")
    print(f"   (需要重新运行L1-L3层才能更新)")

# 统计摘要
print("\n" + "=" * 80)
print("📊 修复前后对比")
print("=" * 80)
print(f"\n当前数据状态:")
print(f"  • 评分范围: [{df['重评分数'].min():.1f}, {df['重评分数'].max():.1f}]")
print(f"  • 无换电记录用户数: {len(no_swap)}")
print(f"  • 无换电但友好分>0: {len(no_swap_with_score)}")
print(f"  • 有骑行但时刻=-1: {len(has_ride_minus1)}")
print(f"  • 有速度但P50=0: {len(p50_zero_has_speed)}")

print("\n" + "=" * 80)
print("💡 提示:")
print("=" * 80)
print("""
对于问题3和问题4（骑行时刻和P50速度）：
- 这些是L1层的修改，当前的user_7d.csv是历史数据
- 需要重新运行完整流水线才能看到修复效果：
  
  python src/run_pipeline.py --mode full
  
或只重新运行受影响的层：
  python src/run_pipeline.py --mode refresh
""")
