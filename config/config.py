"""
用户特征画像项目配置模块
包含数据库配置、导出路径等基础配置项
"""

# -------------------------- 1. 数据库配置 --------------------------
import os

# 数据库连接配置
DB_CONFIG = {
    "host": "rm-2zeeh8zj8e9m56a6fko.mysql.rds.aliyuncs.com",
    "user": "read_only_user",
    "password": os.getenv('DB_PASSWORD', 'Xll_179510@readOnly'),
    "database": "xianglilai",
    "port": 3306,
    "charset": "utf8mb4"
}

# -------------------------- 2. API 配置 --------------------------
# 高德地图API Key，用于地理位置相关功能
AMAP_API_KEY = os.getenv('AMAP_API_KEY', '231b5048493335b091de61840ce78489')

# -------------------------- 3. 导出文件配置 --------------------------
# 测试模式开关：True - 使用测试路径，False - 使用正式路径
TEST_MODE = True

# 基础导出文件夹配置，所有导出文件的根目录
if TEST_MODE:
    BASE_EXPORT_PATH = r"E:\test"
else:
    BASE_EXPORT_PATH = r"E:\OneDrive\Powerbi"

# ==================== 用户行为习惯文件夹 ====================
# 用户行为习惯相关文件路径
USER_BEHAVIOR_FOLDER = "用户行为习惯/用户特征画像"
EXPORT_PATH_USER_BEHAVIOR = USER_BEHAVIOR_FOLDER

# 用户生命周期管理文件
EXPORT_FILE_USER_LIFECYCLE_7D = "全量用户7天滚动生命周期档案.csv"
EXPORT_FILE_USER_DAILY_SNAPSHOT = "用户每日数据快照表.csv"
EXPORT_FILE_USER_ATTENDANCE = "用户月度出勤预估明细.csv"

# 每日用户数据汇总文件夹（parquet格式，原始数据不做处理，方便查看）
DAILY_USER_DATA_RAW_FOLDER = "每日用户数据汇总"
EXPORT_PATH_DAILY_USER_DATA_RAW = f"{USER_BEHAVIOR_FOLDER}/{DAILY_USER_DATA_RAW_FOLDER}"

# 每日用电数据文件夹（parquet格式，预处理数据）
DAILY_POWER_DATA_FOLDER = "每日用电数据"
EXPORT_PATH_DAILY_POWER_DATA = f"{USER_BEHAVIOR_FOLDER}/{DAILY_POWER_DATA_FOLDER}"

# 每日快照文件夹（按日期拆分保存，只存原始物理指标）
DAILY_SNAPSHOT_FOLDER = "每日快照文件"
EXPORT_PATH_DAILY_SNAPSHOT = f"{USER_BEHAVIOR_FOLDER}/{DAILY_SNAPSHOT_FOLDER}"

# 取值标准和用户形态报告
EXPORT_FILE_USER_PROFILE_REPORT = "取值标准和用户形态报告.csv"

# 24小时时序数据聚合文件
EXPORT_FILE_HOURLY_AGG = "小时级时序聚合表_曲线图专用.csv"
EXPORT_FILE_MINUTE_AGG = "分钟级时序聚合表_精细曲线图专用.csv"
EXPORT_FILE_TRACK_DETAIL = "骑行轨迹明细表_地图专用.csv"

# ==================== 基础数据文件夹 ====================
# 用户电池情况数据
BATTERY_STATUS_FOLDER = f"基础数据/用户电池情况"
EXPORT_PATH_BATTERY_STATUS_WITH_CONSUMPTION = BATTERY_STATUS_FOLDER
EXPORT_PATH_BATTERY_STATUS_30D = f"{BATTERY_STATUS_FOLDER}/前30天-前4天"
EXPORT_PATH_BATTERY_STATUS_3D = f"{BATTERY_STATUS_FOLDER}/前3天-昨天"

# 每日合约清单数据
DAILY_CONTRACT_LIST_FOLDER = "基础数据/每日合约清单"
EXPORT_PATH_DAILY_CONTRACT_LIST = DAILY_CONTRACT_LIST_FOLDER
EXPORT_FILE_CONTRACT_HISTORY = f"基础数据/每日合约清单集合.csv"

# 每日合约_辅助数据
DAILY_CONTRACT_HELPER_FOLDER = "基础数据/每日合约_辅助"
EXPORT_PATH_DAILY_CONTRACT_CHANGE_HELPER = DAILY_CONTRACT_HELPER_FOLDER
EXPORT_FILE_CONTRACT_ASSIST = f"基础数据/每日合约_辅助集合.csv"

# 合约表格数据
EXPORT_FILE_CONTRACT_TABLES_DATA = "基础数据/合约表格数据.csv"

# 省市区围栏数据
GEOJSON_FOLDER = "基础数据/省市区围栏"
EXPORT_FILE_PROVINCE_GEOJSON = f"{GEOJSON_FOLDER}/中国_省.geojson"
EXPORT_FILE_CITY_GEOJSON = f"{GEOJSON_FOLDER}/中国_市.geojson"
EXPORT_FILE_DISTRICT_GEOJSON = f"{GEOJSON_FOLDER}/中国_县.geojson"

# 生成完整的导出路径
def get_full_path(relative_path):
    """根据相对路径生成完整的导出路径
    
    Args:
        relative_path: 相对于BASE_EXPORT_PATH的路径
        
    Returns:
        完整的导出路径
    """
    return os.path.join(BASE_EXPORT_PATH, relative_path)

# 用户行为习惯相关完整路径
EXPORT_PATH_USER_BEHAVIOR = get_full_path(EXPORT_PATH_USER_BEHAVIOR)
EXPORT_PATH_DAILY_USER_DATA_RAW = get_full_path(EXPORT_PATH_DAILY_USER_DATA_RAW)
EXPORT_PATH_DAILY_POWER_DATA = get_full_path(EXPORT_PATH_DAILY_POWER_DATA)
EXPORT_PATH_DAILY_SNAPSHOT = get_full_path(EXPORT_PATH_DAILY_SNAPSHOT)
EXPORT_FILE_USER_LIFECYCLE_7D = get_full_path(os.path.join(EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_LIFECYCLE_7D))
EXPORT_FILE_USER_DAILY_SNAPSHOT = get_full_path(os.path.join(EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_DAILY_SNAPSHOT))
EXPORT_FILE_USER_ATTENDANCE = get_full_path(os.path.join(EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_ATTENDANCE))
EXPORT_FILE_USER_PROFILE_REPORT = get_full_path(os.path.join(EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_PROFILE_REPORT))

# 基础数据相关完整路径
EXPORT_PATH_BATTERY_STATUS_WITH_CONSUMPTION = get_full_path(EXPORT_PATH_BATTERY_STATUS_WITH_CONSUMPTION)
EXPORT_PATH_BATTERY_STATUS_30D = get_full_path(EXPORT_PATH_BATTERY_STATUS_30D)
EXPORT_PATH_BATTERY_STATUS_3D = get_full_path(EXPORT_PATH_BATTERY_STATUS_3D)
EXPORT_PATH_DAILY_CONTRACT_LIST = get_full_path(EXPORT_PATH_DAILY_CONTRACT_LIST)
EXPORT_FILE_CONTRACT_HISTORY = get_full_path(EXPORT_FILE_CONTRACT_HISTORY)
EXPORT_PATH_DAILY_CONTRACT_CHANGE_HELPER = get_full_path(EXPORT_PATH_DAILY_CONTRACT_CHANGE_HELPER)
EXPORT_FILE_CONTRACT_ASSIST = get_full_path(EXPORT_FILE_CONTRACT_ASSIST)
EXPORT_FILE_CONTRACT_TABLES_DATA = get_full_path(EXPORT_FILE_CONTRACT_TABLES_DATA)
EXPORT_FILE_PROVINCE_GEOJSON = get_full_path(EXPORT_FILE_PROVINCE_GEOJSON)
EXPORT_FILE_CITY_GEOJSON = get_full_path(EXPORT_FILE_CITY_GEOJSON)
EXPORT_FILE_DISTRICT_GEOJSON = get_full_path(EXPORT_FILE_DISTRICT_GEOJSON)

# ==================== 数据流水线输出路径 ====================
# L0 Raw Ingest 输出
EXPORT_PATH_RAW = get_full_path("data/raw")

# L1 Fact Layer 输出
EXPORT_PATH_FACT_DAILY = get_full_path("data/fact/daily")

# L2 Snapshot Layer 输出
EXPORT_PATH_SNAPSHOT = get_full_path("data/snapshot/user_daily.parquet")

# L3 Lifecycle Layer 输出
EXPORT_PATH_LIFECYCLE_7D = get_full_path("data/lifecycle/user_7d.parquet")

# L4 Report Layer 输出
EXPORT_PATH_REPORTS = get_full_path("data/reports")

# 动态阈值基准线存储路径
EXPORT_PATH_THRESHOLDS_BASELINE = os.path.join(os.path.dirname(__file__), '..', 'src', 'thresholds_baseline.json')
