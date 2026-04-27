"""
代理配置模块
包含数据库配置、导出路径等基础配置项
"""

# -------------------------- 1. 数据库配置 --------------------------
# 导入os模块，用于读取环境变量
import os

# 数据库连接配置
# 建议通过环境变量 DB_PASSWORD 设置数据库密码，提高安全性
DB_CONFIG = {
    "host": "rm-2zeeh8zj8e9m56a6fko.mysql.rds.aliyuncs.com",  # 数据库主机地址
    "user": "read_only_user",  # 数据库用户名
    "password": os.getenv('DB_PASSWORD', 'Xll_179510@readOnly'),  # 数据库密码
    "database": "xianglilai",  # 数据库名称
    "port": 3306,  # 数据库端口
    "charset": "utf8mb4"  # 字符集
}

# -------------------------- 2. API 配置 --------------------------
# 高德地图API Key，用于地理位置相关功能
# 建议通过环境变量 AMAP_API_KEY 设置，提高安全性
AMAP_API_KEY = os.getenv('AMAP_API_KEY', '231b5048493335b091de61840ce78489')

# -------------------------- 3. 导出文件配置 --------------------------
# 测试模式开关：True - 使用测试路径，False - 使用正式路径
TEST_MODE = True

# 基础导出文件夹配置，所有导出文件的根目录
# 输出路径配置
if TEST_MODE:
    BASE_EXPORT_PATH = r"E:\test"  # 测试路径
else:
    BASE_EXPORT_PATH = r"E:\OneDrive\Powerbi"  # 正式路径

# ==================== 基础数据文件夹 ====================
# 基础数据相关文件路径
BASE_DATA_FOLDER = "基础数据"

# 省市区围栏数据
GEOJSON_FOLDER = f"{BASE_DATA_FOLDER}/省市区围栏"
EXPORT_FILE_PROVINCE_GEOJSON = f"{GEOJSON_FOLDER}/中国_省.geojson"  # 中国省份地理数据
EXPORT_FILE_CITY_GEOJSON = f"{GEOJSON_FOLDER}/中国_市.geojson"  # 中国城市地理数据
EXPORT_FILE_DISTRICT_GEOJSON = f"{GEOJSON_FOLDER}/中国_县.geojson"  # 中国县地理数据

# 用户电池情况数据
BATTERY_STATUS_FOLDER = f"{BASE_DATA_FOLDER}/用户电池情况"
EXPORT_PATH_BATTERY_STATUS_WITH_CONSUMPTION = BATTERY_STATUS_FOLDER  # 带耗电量的用户电池情况
EXPORT_PATH_BATTERY_STATUS_30D = f"{BATTERY_STATUS_FOLDER}/前30天-前4天"  # 前30天到前4天的用户电池情况
EXPORT_PATH_BATTERY_STATUS_3D = f"{BATTERY_STATUS_FOLDER}/前3天-昨天"  # 前3天到昨天的用户电池情况

# 每日电池容量数据
BATTERY_CAPACITY_FOLDER = f"{BASE_DATA_FOLDER}/每日电池容量"
EXPORT_PATH_BATTERY_CAPACITY = BATTERY_CAPACITY_FOLDER  # 每日电池容量存储目录
EXPORT_FILE_BATTERY_CAPACITY_OUTPUT = "每日电池容量.csv"  # 每日电池容量输出文件

# 每日电池清单数据
DAILY_BATTERY_EXCHANGE_FOLDER = f"{BASE_DATA_FOLDER}/每日电池清单"
EXPORT_PATH_DAILY_BATTERY_EXCHANGE = DAILY_BATTERY_EXCHANGE_FOLDER  # 每日电池清单存储目录

# 每日合约清单数据
DAILY_CONTRACT_LIST_FOLDER = f"{BASE_DATA_FOLDER}/每日合约清单"
EXPORT_PATH_DAILY_CONTRACT_LIST = DAILY_CONTRACT_LIST_FOLDER  # 每日合约清单存储目录
EXPORT_FILE_CONTRACT_HISTORY = f"{BASE_DATA_FOLDER}/每日合约清单集合.csv"  # 每日合约清单集合

# 每日合约_辅助数据
DAILY_CONTRACT_HELPER_FOLDER = f"{BASE_DATA_FOLDER}/每日合约_辅助"
EXPORT_PATH_DAILY_CONTRACT_CHANGE_HELPER = DAILY_CONTRACT_HELPER_FOLDER  # 每日合约_辅助存储目录
EXPORT_FILE_CONTRACT_ASSIST = f"{BASE_DATA_FOLDER}/每日合约_辅助集合.csv"  # 每日合约_辅助集合

# 合约表格数据
EXPORT_FILE_CONTRACT_TABLES_DATA = f"{BASE_DATA_FOLDER}/合约表格数据.csv"  # 合约表格数据

# ==================== 合约信息文件夹 ====================
# 合约信息相关文件路径
CONTRACT_FOLDER = "合约信息"
EXPORT_PATH_CONTRACT = CONTRACT_FOLDER  # 合约相关文件存储目录

# 合约基础信息
EXPORT_FILE_NAME_USER_LIST = f"{CONTRACT_FOLDER}/9-1用户列表.csv"  # 用户列表文件
EXPORT_FILE_CONTRACT_SUMMARY = f"{CONTRACT_FOLDER}/4-1合约信息2025年.csv"  # 2025年合约信息汇总
EXPORT_FILE_CONTRACT_A2 = f"{CONTRACT_FOLDER}/A-2每日合约列表.csv"  # 每日合约列表
EXPORT_FILE_CONTRACT_A3 = f"{CONTRACT_FOLDER}/A-3每日新增合约.csv"  # 每日新增合约
EXPORT_FILE_CONTRACT_A4 = f"{CONTRACT_FOLDER}/A-4每日流失合约.csv"  # 每日流失合约
EXPORT_FILE_CONTRACT_A5 = f"{CONTRACT_FOLDER}/A-5暂停合约信息.csv"  # 暂停合约信息
EXPORT_FILE_CONTRACT_SNAPSHOT = f"{CONTRACT_FOLDER}/system_snapshot_latest.csv"  # 系统快照最新数据

# 特殊合约信息
EXPORT_FILE_ONLINE_USERS = f"{CONTRACT_FOLDER}/4-6在网用户信息-换电.csv"  # 在网用户信息（换电）
EXPORT_FILE_CONTRACT_SUE = f"{CONTRACT_FOLDER}/9-2起诉用户信息.csv"  # 起诉用户信息
EXPORT_FILE_CONTRACT_EARLY = f"{CONTRACT_FOLDER}/4-5用户最早合约时间.csv"  # 用户最早合约时间
EXPORT_FILE_CONTRACT_NEW = f"{CONTRACT_FOLDER}/9-3拉新用户列表.csv"  # 拉新用户列表
EXPORT_FILE_CONTRACT_NEW_DETAIL = f"{CONTRACT_FOLDER}/9-4拉新用户明细.csv"  # 拉新用户明细


# ==================== 电池信息文件夹 ====================
# 电池相关文件路径
BATTERY_FOLDER = "电池信息"
EXPORT_PATH_BATTERY = BATTERY_FOLDER  # 电池相关文件存储目录

# 电池基础信息
EXPORT_FILE_NAME_BATTERY_LIST = f"{BATTERY_FOLDER}/3-1电池列表.csv"  # 电池列表
EXPORT_FILE_NAME_BATTERY_EXPIRED = f"{BATTERY_FOLDER}/3-3过期电池清单.csv"  # 过期电池清单
EXPORT_FILE_NAME_BATTERY_SHIPMENT = f"{BATTERY_FOLDER}/单电池收发货记录.csv"  # 单电池收发货记录

# 电池历史信息
EXPORT_FILE_NAME_BATTERY_HISTORY = f"{BATTERY_FOLDER}/历史电池清单.csv"  # 历史电池清单

# ==================== 换电柜信息文件夹 ====================
# 换电柜相关文件路径
CABINET_FOLDER = "电池信息"  # 与电池信息共用文件夹

# 换电柜基础信息
EXPORT_FILE_NAME_CABINET_LIST = f"{CABINET_FOLDER}/3-2换电柜列表.csv"  # 换电柜列表

# 换电柜财务信息
EXPORT_FILE_NAME_CABINET_FINANCE = f"{CABINET_FOLDER}/3-7电柜财务实际差异.csv"  # 电柜财务实际差异
EXPORT_FILE_NAME_BATTERY_FINANCE = f"{CABINET_FOLDER}/3-8电池财务实际差异.csv"  # 电池财务实际差异

# 换电柜历史信息
EXPORT_FILE_NAME_CABINET_HISTORY = f"{CABINET_FOLDER}/历史电柜清单.csv"  # 历史电柜清单

# ==================== 换电柜用电参数文件夹 ====================
# 换电柜用电参数相关文件路径
CABINET_POWER_FOLDER = "换电柜用电参数"

# 用电参数文件
EXPORT_FILE_CABINET_TIME_STATS = f"{CABINET_POWER_FOLDER}/8-3电柜按时间统计.csv"  # 电柜按时间统计数据
EXPORT_FILE_CABINET_POINT_PARAMS = f"{CABINET_POWER_FOLDER}/换电柜时点参数.csv"  # 换电柜时点参数数据
EXPORT_FILE_CABINET_HOURLY_POWER = f"{CABINET_POWER_FOLDER}/换电柜小时用电.csv"  # 换电柜小时用电数据
EXPORT_FILE_CABINET_DAILY_POWER = f"{CABINET_POWER_FOLDER}/8-2-2电柜每日电量2026年.csv"  # 电柜每日电量数据

# ==================== 换电记录文件夹 ====================
# 换电记录相关文件路径
BATTERY_CHANGE_FOLDER = "换电记录"

# 2026年换电记录
BATTERY_CHANGE_2026_FOLDER = f"{BATTERY_CHANGE_FOLDER}/2026年"
EXPORT_PATH_BATTERY_CHANGE_2026 = BATTERY_CHANGE_2026_FOLDER  # 2026年换电记录存储目录
EXPORT_PATH_BATTERY_CHANGE_2026_INPUT = BATTERY_CHANGE_2026_FOLDER  # 2026年换电记录输入目录
EXPORT_FILE_BATTERY_CHANGE_2026_OUTPUT = f"{BATTERY_CHANGE_FOLDER}/2026年换电记录.csv"  # 2026年换电记录输出文件

# ==================== 收入情况文件夹 ====================
# 收入情况相关文件路径
INCOME_FOLDER = "5-收入情况"

# 收入情况文件
EXPORT_FILE_ONLINE_ORDERS = f"{INCOME_FOLDER}/5-1-2线上订单-2025年后.csv"  # 2025年后线上订单数据
EXPORT_FILE_REFUND_ORDERS = f"{INCOME_FOLDER}/5-2-2退款订单-2025年后.csv"  # 2025年后退款订单数据
EXPORT_FILE_TIKTOK_COUPONS = f"{INCOME_FOLDER}/5-5抖音优惠券.csv"  # 抖音优惠券数据
EXPORT_FILE_FRAME_ORDERS = f"{INCOME_FOLDER}/5-6含车架订单.csv"  # 含车架订单数据

# ==================== 套餐情况文件夹 ====================
# 套餐情况相关文件路径
PACKAGE_FOLDER = "6-套餐情况"
EXPORT_PATH_PACKAGE = PACKAGE_FOLDER  # 套餐情况存储目录

# 套餐情况文件
EXPORT_FILE_USER_PACKAGE = f"{PACKAGE_FOLDER}/6-1用户套餐.csv"  # 用户套餐数据
EXPORT_FILE_COUPONS = f"{PACKAGE_FOLDER}/6-2-1优惠券.csv"  # 优惠券数据
EXPORT_FILE_GROUP_PAYMENT_CHANGE = f"{PACKAGE_FOLDER}/6-4集团支付变更.csv"  # 集团支付变更数据

# ==================== 代理列表文件夹 ====================
# 代理相关文件路径
AGENT_FOLDER = "代理列表"
EXPORT_PATH_AGENT = AGENT_FOLDER  # 代理相关文件存储目录

# 代理文件
EXPORT_FILE_NAME_AGENT_LEVEL = "2-1代理级别.csv"  # 代理级别文件
EXPORT_FILE_NAME_COMPANY = "2-2所属公司.csv"  # 所属公司文件

# ==================== 用户行为习惯文件夹 ====================
# 用户行为习惯相关文件路径
USER_BEHAVIOR_FOLDER = "用户行为习惯/用户特征画像"
EXPORT_PATH_USER_BEHAVIOR = USER_BEHAVIOR_FOLDER  # 用户行为习惯存储目录

# 用户生命周期管理文件
EXPORT_FILE_USER_LIFECYCLE_7D = "全量用户7天滚动生命周期档案.csv"  # 全量用户7天滚动生命周期档案
EXPORT_FILE_USER_DAILY_SNAPSHOT = "用户每日数据快照表.csv"  # 用户每日数据快照表
EXPORT_FILE_USER_ATTENDANCE = "用户月度出勤预估明细.csv"  # 用户月度出勤预估明细

# 每日用户数据汇总文件夹（parquet格式，原始数据不做处理，方便查看）
DAILY_USER_DATA_RAW_FOLDER = "每日用户数据汇总"
EXPORT_PATH_DAILY_USER_DATA_RAW = f"{USER_BEHAVIOR_FOLDER}/{DAILY_USER_DATA_RAW_FOLDER}"  # 每日用户数据汇总存储目录

# 每日用电数据文件夹（parquet格式，预处理数据）
DAILY_POWER_DATA_FOLDER = "每日用电数据"
EXPORT_PATH_DAILY_POWER_DATA = f"{USER_BEHAVIOR_FOLDER}/{DAILY_POWER_DATA_FOLDER}"  # 每日用电数据存储目录

# 每日快照文件夹（按日期拆分保存，只存原始物理指标）
DAILY_SNAPSHOT_FOLDER = "每日快照文件"
EXPORT_PATH_DAILY_SNAPSHOT = f"{USER_BEHAVIOR_FOLDER}/{DAILY_SNAPSHOT_FOLDER}"  # 每日快照存储目录

# 取值标准和用户形态报告
EXPORT_FILE_USER_PROFILE_REPORT = "取值标准和用户形态报告.csv"  # 取值标准和用户形态报告

# 24小时时序数据聚合文件
EXPORT_FILE_HOURLY_AGG = "小时级时序聚合表_曲线图专用.csv"  # 小时级时序聚合表
EXPORT_FILE_MINUTE_AGG = "分钟级时序聚合表_精细曲线图专用.csv"  # 分钟级时序聚合表
EXPORT_FILE_TRACK_DETAIL = "骑行轨迹明细表_地图专用.csv"  # 骑行轨迹明细表

# ==================== 其他文件夹 ====================
# 用户换电数据相关文件路径
USER_CHANGE_31D_FOLDER = "7-2-1用户换电数据-31天"
EXPORT_PATH_USER_CHANGE_31D = USER_CHANGE_31D_FOLDER  # 31天用户换电数据存储目录

# 用户GPS数据相关文件路径
USER_GPS_FOLDER = "7-4用户GPS数据"
EXPORT_PATH_USER_GPS = USER_GPS_FOLDER  # 用户GPS数据存储目录

# 换电柜经纬度转地址相关文件路径
CABINET_ADDRESS_FOLDER = "O-换电柜经纬度转地址"
EXPORT_FILE_CABINET_ADDRESS = f"{CABINET_ADDRESS_FOLDER}/换电柜所属位置.xlsx"  # 换电柜所属位置数据

# 每日电池统计相关文件路径
DAILY_BATTERY_STATS_FOLDER = "每日资产情况"
EXPORT_PATH_DAILY_BATTERY_STATS = DAILY_BATTERY_STATS_FOLDER  # 每日资产情况存储目录

# 生成完整的导出路径

def get_full_path(relative_path):
    """根据相对路径生成完整的导出路径
    
    Args:
        relative_path: 相对于BASE_EXPORT_PATH的路径
        
    Returns:
        完整的导出路径
    """
    return os.path.join(BASE_EXPORT_PATH, relative_path)

# 重新定义所有导出路径为完整路径，将相对路径转换为绝对路径
EXPORT_PATH_AGENT = get_full_path(EXPORT_PATH_AGENT)
EXPORT_PATH_CONTRACT = get_full_path(EXPORT_PATH_CONTRACT)
EXPORT_PATH_BATTERY = get_full_path(EXPORT_PATH_BATTERY)
EXPORT_FILE_CONTRACT_SUMMARY = get_full_path(EXPORT_FILE_CONTRACT_SUMMARY)
EXPORT_FILE_CONTRACT_A2 = get_full_path(EXPORT_FILE_CONTRACT_A2)
EXPORT_FILE_CONTRACT_A3 = get_full_path(EXPORT_FILE_CONTRACT_A3)
EXPORT_FILE_CONTRACT_A4 = get_full_path(EXPORT_FILE_CONTRACT_A4)
EXPORT_FILE_CONTRACT_A5 = get_full_path(EXPORT_FILE_CONTRACT_A5)
EXPORT_FILE_CONTRACT_HISTORY = get_full_path(EXPORT_FILE_CONTRACT_HISTORY)
EXPORT_FILE_CONTRACT_ASSIST = get_full_path(EXPORT_FILE_CONTRACT_ASSIST)
EXPORT_FILE_CONTRACT_SNAPSHOT = get_full_path(EXPORT_FILE_CONTRACT_SNAPSHOT)
EXPORT_FILE_ONLINE_ORDERS = get_full_path(EXPORT_FILE_ONLINE_ORDERS)
EXPORT_FILE_REFUND_ORDERS = get_full_path(EXPORT_FILE_REFUND_ORDERS)
EXPORT_FILE_ONLINE_USERS = get_full_path(EXPORT_FILE_ONLINE_USERS)
EXPORT_PATH_BATTERY_CHANGE_2026 = get_full_path(EXPORT_PATH_BATTERY_CHANGE_2026)
EXPORT_PATH_BATTERY_CAPACITY = get_full_path(EXPORT_PATH_BATTERY_CAPACITY)
EXPORT_PATH_USER_CHANGE_31D = get_full_path(EXPORT_PATH_USER_CHANGE_31D)
EXPORT_PATH_USER_GPS = get_full_path(EXPORT_PATH_USER_GPS)
EXPORT_FILE_CABINET_TIME_STATS = get_full_path(EXPORT_FILE_CABINET_TIME_STATS)
EXPORT_FILE_CABINET_POINT_PARAMS = get_full_path(EXPORT_FILE_CABINET_POINT_PARAMS)
EXPORT_FILE_CABINET_HOURLY_POWER = get_full_path(EXPORT_FILE_CABINET_HOURLY_POWER)
EXPORT_PATH_DAILY_BATTERY_EXCHANGE = get_full_path(EXPORT_PATH_DAILY_BATTERY_EXCHANGE)
EXPORT_PATH_DAILY_CONTRACT_LIST = get_full_path(EXPORT_PATH_DAILY_CONTRACT_LIST)
EXPORT_FILE_CONTRACT_TABLES_DATA = get_full_path(EXPORT_FILE_CONTRACT_TABLES_DATA)
EXPORT_PATH_DAILY_CONTRACT_CHANGE_HELPER = get_full_path(EXPORT_PATH_DAILY_CONTRACT_CHANGE_HELPER)
EXPORT_PATH_DAILY_BATTERY_STATS = get_full_path(EXPORT_PATH_DAILY_BATTERY_STATS)
EXPORT_FILE_CABINET_ADDRESS = get_full_path(EXPORT_FILE_CABINET_ADDRESS)

# 电池相关文件路径转换为完整路径
EXPORT_FILE_NAME_BATTERY_LIST = get_full_path(EXPORT_FILE_NAME_BATTERY_LIST)
EXPORT_FILE_NAME_BATTERY_EXPIRED = get_full_path(EXPORT_FILE_NAME_BATTERY_EXPIRED)
EXPORT_FILE_NAME_BATTERY_SHIPMENT = get_full_path(EXPORT_FILE_NAME_BATTERY_SHIPMENT)
EXPORT_FILE_NAME_BATTERY_HISTORY = get_full_path(EXPORT_FILE_NAME_BATTERY_HISTORY)
EXPORT_FILE_NAME_BATTERY_FINANCE = get_full_path(EXPORT_FILE_NAME_BATTERY_FINANCE)

# 换电柜相关文件路径转换为完整路径
EXPORT_FILE_NAME_CABINET_LIST = get_full_path(EXPORT_FILE_NAME_CABINET_LIST)
EXPORT_FILE_NAME_CABINET_FINANCE = get_full_path(EXPORT_FILE_NAME_CABINET_FINANCE)
EXPORT_FILE_NAME_CABINET_HISTORY = get_full_path(EXPORT_FILE_NAME_CABINET_HISTORY)

# 套餐情况相关完整路径
EXPORT_PATH_PACKAGE = get_full_path(EXPORT_PATH_PACKAGE)
EXPORT_FILE_USER_PACKAGE = get_full_path(EXPORT_FILE_USER_PACKAGE)

# 新增配置的完整路径
EXPORT_PATH_BATTERY_CHANGE_2026_INPUT = get_full_path(EXPORT_PATH_BATTERY_CHANGE_2026_INPUT)
EXPORT_FILE_BATTERY_CHANGE_2026_OUTPUT = get_full_path(EXPORT_FILE_BATTERY_CHANGE_2026_OUTPUT)
EXPORT_PATH_BATTERY_STATUS_WITH_CONSUMPTION = get_full_path(EXPORT_PATH_BATTERY_STATUS_WITH_CONSUMPTION)
EXPORT_PATH_BATTERY_STATUS_30D = get_full_path(EXPORT_PATH_BATTERY_STATUS_30D)
EXPORT_PATH_BATTERY_STATUS_3D = get_full_path(EXPORT_PATH_BATTERY_STATUS_3D)
EXPORT_PATH_USER_BEHAVIOR = get_full_path(EXPORT_PATH_USER_BEHAVIOR)
EXPORT_PATH_DAILY_USER_DATA_RAW = get_full_path(EXPORT_PATH_DAILY_USER_DATA_RAW)
EXPORT_PATH_DAILY_POWER_DATA = get_full_path(EXPORT_PATH_DAILY_POWER_DATA)
EXPORT_PATH_DAILY_SNAPSHOT = get_full_path(EXPORT_PATH_DAILY_SNAPSHOT)
EXPORT_FILE_USER_LIFECYCLE_7D = get_full_path(os.path.join(EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_LIFECYCLE_7D))
EXPORT_FILE_USER_DAILY_SNAPSHOT = get_full_path(os.path.join(EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_DAILY_SNAPSHOT))
EXPORT_FILE_USER_ATTENDANCE = get_full_path(os.path.join(EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_ATTENDANCE))
EXPORT_FILE_USER_PROFILE_REPORT = get_full_path(os.path.join(EXPORT_PATH_USER_BEHAVIOR, EXPORT_FILE_USER_PROFILE_REPORT))
EXPORT_FILE_PROVINCE_GEOJSON = get_full_path(EXPORT_FILE_PROVINCE_GEOJSON)
EXPORT_FILE_CITY_GEOJSON = get_full_path(EXPORT_FILE_CITY_GEOJSON)
EXPORT_FILE_DISTRICT_GEOJSON = get_full_path(EXPORT_FILE_DISTRICT_GEOJSON)
EXPORT_FILE_BATTERY_CAPACITY_OUTPUT = get_full_path(EXPORT_FILE_BATTERY_CAPACITY_OUTPUT)


# -------------------------- 4. 脚本映射配置（脚本名称 -> 输出路径和文件名） --------------------------
# 用途：快速查看每个脚本对应的输出路径和文件名，方便管理和维护
# 键：脚本名称，值：包含导出路径和文件名的字典
SCRIPT_CONFIG = {
    # 代理级别脚本：导出代理分类信息到 Excel
    "2-1代理级别": {
        "export_path": EXPORT_PATH_AGENT,
        "file_name": EXPORT_FILE_NAME_AGENT_LEVEL
    },
    # 所属公司脚本：导出公司信息到 Excel
    "2-2所属公司": {
        "export_path": EXPORT_PATH_AGENT,
        "file_name": EXPORT_FILE_NAME_COMPANY
    },
    # 用户列表脚本：导出用户信息到 CSV
    "9-1用户列表": {
        "export_path": EXPORT_PATH_CONTRACT,
        "file_name": EXPORT_FILE_NAME_USER_LIST
    },
    # 单电池收发货记录脚本：导出电池收发货信息到 CSV
    "单电池收发货记录": {
        "export_path": EXPORT_PATH_BATTERY,
        "file_name": EXPORT_FILE_NAME_BATTERY_SHIPMENT
    },
    # 电池信息脚本：导出电池列表和过期电池清单到 CSV
    "电池信息": {
        "export_path": EXPORT_PATH_BATTERY,
        "file_names": [EXPORT_FILE_NAME_BATTERY_EXPIRED, EXPORT_FILE_NAME_BATTERY_LIST]
    },
    # 合约信息脚本：导出多个合约相关文件到 CSV
    "合约信息": {
        "export_path": EXPORT_PATH_CONTRACT,
        "file_names": [
            EXPORT_FILE_CONTRACT_SUMMARY,
            EXPORT_FILE_CONTRACT_A2,
            EXPORT_FILE_CONTRACT_A3,
            EXPORT_FILE_CONTRACT_A4,
            EXPORT_FILE_CONTRACT_A5
        ]
    },
    # 线上收款数据脚本：导出线上订单数据到 CSV
    "线上收款数据": {
        "export_path": get_full_path("5-收入情况"),
        "file_name": EXPORT_FILE_ONLINE_ORDERS
    },
    # 线上退款数据脚本：导出退款订单数据到 CSV
    "线上退款数据": {
        "export_path": get_full_path("5-收入情况"),
        "file_name": EXPORT_FILE_REFUND_ORDERS
    },
    # 在网用户信息脚本：导出在网用户信息到 CSV
    "在网用户信息": {
        "export_path": EXPORT_PATH_CONTRACT,
        "file_name": EXPORT_FILE_ONLINE_USERS
    },
    # 用户电流情况脚本：导出用户电池电流情况数据到 CSV
    "7-5用户电流情况": {
        "export_path": EXPORT_PATH_BATTERY_STATUS_WITH_CONSUMPTION,
        "file_name": "用户电池情况_{timestamp}.csv"
    },
    # 2026年换电记录统计脚本：导出2026年换电记录数据到 CSV
    "2026年换电记录统计": {
        "export_path": EXPORT_PATH_BATTERY_CHANGE_2026,
        "file_name": "换电记录_{date}.csv"
    },
    # 3-B电池剩余容量脚本：导出每日电池容量数据到 CSV
    "3-B电池剩余容量": {
        "export_path": EXPORT_PATH_BATTERY_CAPACITY,
        "file_name": "每日电池容量_{timestamp}.csv"
    },
    # 7-2-1用户换电数据-31天脚本：导出31天用户换电数据到 CSV
    "7-2-1用户换电数据-31天": {
        "export_path": EXPORT_PATH_USER_CHANGE_31D,
        "file_name": "用户换电数据-31天.csv"
    },
    # 7-2-2换电用户统计天数脚本：导出换电用户统计天数数据到 CSV
    "7-2-2换电用户统计天数": {
        "export_path": EXPORT_PATH_USER_CHANGE_31D,
        "file_name": "7-2-2换电用户统计天数.csv"
    },
    # 7-4用户GPS数据脚本：导出用户GPS数据到 CSV
    "7-4用户GPS数据": {
        "export_path": EXPORT_PATH_USER_GPS,
        "file_name": "用户GPS数据_带省市区县_高德坐标.csv"
    },
    # 8-3电柜按时间统计脚本：导出电柜按时间统计数据到 CSV
    "8-3电柜按时间统计": {
        "export_path": get_full_path("换电柜用电参数"),
        "file_name": "8-3电柜按时间统计.csv"
    },
    # 8-4换电柜时点参数脚本：导出电柜时点参数数据到 CSV
    "8-4换电柜时点参数": {
        "export_path": get_full_path("换电柜用电参数"),
        "file_name": "换电柜时点参数.csv"
    },
    # 8-5换电柜小时用电脚本：导出电柜小时用电数据到 CSV
    "8-5换电柜小时用电": {
        "export_path": get_full_path("换电柜用电参数"),
        "file_name": "换电柜小时用电.csv"
    },
    # A-1每日换电电池脚本：导出每日换电电池数据到 CSV
    "A-1每日换电电池": {
        "export_path": EXPORT_PATH_DAILY_BATTERY_EXCHANGE,
        "file_name": "换电电池统计_{日期}.csv"
    },
    # A-2每日合约列表脚本：导出每日合约列表数据到 CSV
    "A-2每日合约列表": {
        "export_path": EXPORT_PATH_DAILY_CONTRACT_LIST,
        "file_name": "用户换电合约_{日期}.csv"
    },
    # A-8统计代理表格时间脚本：导出合约表格数据到 CSV
    "A-8统计代理表格时间": {
        "export_path": get_full_path("基础数据"),
        "file_name": "合约表格数据.csv"
    },
    # 每日合约变化_辅助脚本：导出每日合约变化辅助数据到 CSV
    "每日合约变化_辅助": {
        "export_path": EXPORT_PATH_DAILY_CONTRACT_CHANGE_HELPER,
        "file_name": "每日合约变化_辅助_{日期}.csv"
    },
    # 每日电池统计脚本：导出每日电池统计数据到 CSV
    "每日电池统计": {
        "export_path": EXPORT_PATH_DAILY_BATTERY_STATS,
        "file_name": "换电电池统计_{日期}.csv"
    },
    # 用户id入网退网时间脚本：导出用户合约信息数据到 CSV
    "用户id入网退网时间": {
        "export_path": get_full_path("用户特征画像"),
        "file_name": "用户合约信息.csv"
    },
    # 6-1用户套餐脚本：导出用户套餐数据到 CSV
    "6-1用户套餐": {
        "export_path": EXPORT_PATH_PACKAGE,
        "file_name": "6-1用户套餐.csv"
    }
}
