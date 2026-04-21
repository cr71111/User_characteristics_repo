"""
用户特征画像项目配置模块
包含输入路径、输出路径等基础配置项
"""

import os

# -------------------------- 1. 基础路径配置 --------------------------
# 测试模式开关：True - 使用测试路径，False - 使用正式路径
TEST_MODE = True

if TEST_MODE:
    BASE_EXPORT_PATH = r"E:\test"
else:
    BASE_EXPORT_PATH = r"E:\OneDrive\Powerbi"

def get_full_path(relative_path):
    """根据相对路径生成完整的导出路径"""
    return os.path.join(BASE_EXPORT_PATH, relative_path)


# -------------------------- 2. 输入数据路径 --------------------------
USER_BEHAVIOR_FOLDER = "用户行为习惯/用户特征画像"

BATTERY_STATUS_FOLDER = f"基础数据/用户电池情况"
EXPORT_PATH_BATTERY_STATUS_30D = get_full_path(f"{BATTERY_STATUS_FOLDER}/前30天-前4天")
EXPORT_PATH_BATTERY_STATUS_3D = get_full_path(f"{BATTERY_STATUS_FOLDER}/前3天-昨天")

GEOJSON_FOLDER = "基础数据/省市区围栏"
EXPORT_FILE_PROVINCE_GEOJSON = get_full_path(f"{GEOJSON_FOLDER}/中国_省.geojson")
EXPORT_FILE_CITY_GEOJSON = get_full_path(f"{GEOJSON_FOLDER}/中国_市.geojson")
EXPORT_FILE_DISTRICT_GEOJSON = get_full_path(f"{GEOJSON_FOLDER}/中国_县.geojson")


# -------------------------- 3. 输出数据路径 --------------------------
DATA_OUTPUT_ROOT = get_full_path(f"{USER_BEHAVIOR_FOLDER}/data")

EXPORT_PATH_RAW = os.path.join(DATA_OUTPUT_ROOT, "raw")
EXPORT_PATH_FACT_DAILY = os.path.join(DATA_OUTPUT_ROOT, "fact/daily")
EXPORT_PATH_SNAPSHOT = os.path.join(DATA_OUTPUT_ROOT, "snapshot/user_daily.parquet")
EXPORT_PATH_LIFECYCLE_7D = os.path.join(DATA_OUTPUT_ROOT, "lifecycle/user_7d.parquet")
EXPORT_PATH_REPORTS = os.path.join(DATA_OUTPUT_ROOT, "reports")

EXPORT_PATH_THRESHOLDS_BASELINE = os.path.join(os.path.dirname(__file__), '..', 'src', 'thresholds_baseline.json')
