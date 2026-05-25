"""
用户特征画像项目配置模块
包含输入路径、输出路径等基础配置项
"""

import os

# -------------------------- 1. 基础路径配置 --------------------------
# 测试模式开关：True - 使用测试路径，False - 使用正式路径
TEST_MODE = False

if TEST_MODE:
    BASE_EXPORT_PATH = r"E:\test"
else:
    BASE_EXPORT_PATH = r"E:\OneDrive\DataBase\DataBase"

def get_full_path(relative_path):
    """根据相对路径生成完整的导出路径"""
    return os.path.join(BASE_EXPORT_PATH, relative_path)


# -------------------------- 2. 输入数据路径 --------------------------
USER_BEHAVIOR_FOLDER = "用户行为习惯/用户特征画像"

# 新数据源：每日parquet文件，文件名格式 battery_status_YYYY-MM-DD.parquet
BATTERY_STATUS_FOLDER = f"{USER_BEHAVIOR_FOLDER}/每日用户数据汇总"
EXPORT_PATH_BATTERY_STATUS_DAILY = get_full_path(f"{BATTERY_STATUS_FOLDER}")

# 省市区围栏（项目内 data/ 目录）
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EXPORT_FILE_PROVINCE_GEOJSON = os.path.join(DATA_DIR, "中国_省.geojson")
EXPORT_FILE_CITY_GEOJSON = os.path.join(DATA_DIR, "中国_市.geojson")
EXPORT_FILE_DISTRICT_GEOJSON = os.path.join(DATA_DIR, "中国_县.geojson")

# 合约信息
CONTRACT_FOLDER = "合约信息"
EXPORT_FILE_CONTRACT_EARLY = get_full_path(f"{CONTRACT_FOLDER}/4-5用户最早合约时间.csv")

# 电池信息
BATTERY_INFO_FOLDER = "电池信息"
EXPORT_FILE_BATTERY_CELL_VOLTAGE = get_full_path(f"{BATTERY_INFO_FOLDER}/电池单体电压.csv")


# -------------------------- 3. 输出数据路径 --------------------------
DATA_OUTPUT_ROOT = get_full_path(f"{USER_BEHAVIOR_FOLDER}")

EXPORT_PATH_RAW = os.path.join(DATA_OUTPUT_ROOT, "raw")
EXPORT_PATH_FACT_DAILY = os.path.join(DATA_OUTPUT_ROOT, "fact/daily")
EXPORT_PATH_SNAPSHOT = os.path.join(DATA_OUTPUT_ROOT, "snapshot/user_daily.parquet")
EXPORT_PATH_LIFECYCLE_7D = os.path.join(DATA_OUTPUT_ROOT, "lifecycle/user_7d.parquet")
EXPORT_PATH_REPORTS = os.path.join(DATA_OUTPUT_ROOT, "reports")

EXPORT_PATH_THRESHOLDS_BASELINE = os.path.join(os.path.dirname(__file__), '..', 'src', 'tools', 'thresholds_baseline.json')
