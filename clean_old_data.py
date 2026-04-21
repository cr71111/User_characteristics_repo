import os
import glob

# 删除旧的fact层数据
fact_dir = r"E:\test\data\fact\daily"
files = glob.glob(os.path.join(fact_dir, "*.parquet"))
for f in files:
    try:
        os.remove(f)
        print(f"已删除: {f}")
    except Exception as e:
        print(f"删除失败 {f}: {e}")

# 删除旧的snapshot层数据
snapshot_dir = r"E:\test\data\snapshot\daily"
files = glob.glob(os.path.join(snapshot_dir, "*.parquet"))
for f in files:
    try:
        os.remove(f)
        print(f"已删除: {f}")
    except Exception as e:
        print(f"删除失败 {f}: {e}")

# 删除旧的lifecycle层数据
lifecycle_dir = r"E:\test\data\lifecycle"
files = glob.glob(os.path.join(lifecycle_dir, "*.parquet"))
for f in files:
    try:
        os.remove(f)
        print(f"已删除: {f}")
    except Exception as e:
        print(f"删除失败 {f}: {e}")

print("\n旧数据清理完成")
