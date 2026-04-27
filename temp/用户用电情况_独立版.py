import os
import sys
import time
import threading
from datetime import datetime, timedelta
from copy import deepcopy

import pymysql
import pandas as pd

# ==================== 内嵌配置 ====================
DB_CONFIG = {
    "host": "rm-2zeeh8zj8e9m56a6fko.mysql.rds.aliyuncs.com",
    "user": "read_only_user",
    "password": os.getenv('DB_PASSWORD', 'Xll_179510@readOnly'),
    "database": "xianglilai",
    "port": 3306,
    "charset": "utf8mb4"
}

TEST_MODE = True
if TEST_MODE:
    BASE_EXPORT_PATH = r"E:\test"
else:
    BASE_EXPORT_PATH = r"E:\OneDrive\Powerbi"

EXPORT_PATH_DAILY_USER_DATA_RAW = os.path.join(BASE_EXPORT_PATH, "用户行为习惯", "用户特征画像", "每日用户数据汇总")
# ================================================

SQL_QUERY = """
SELECT
    DATE(CURRENT_TIMESTAMP) AS `统计日期`,
    DATE_FORMAT(CURRENT_TIMESTAMP, '%H:%i:%s') AS `统计时间`,
    UNIX_TIMESTAMP(CURRENT_TIMESTAMP) AS `时间戳`,
    uec.`user_id` AS `用户id`,
    CASE
        WHEN uec.`agent_id` IN (1193, 1197, 1215, 1216, 1217, 1218, 1219, 1256, 1271, 1384, 1516, 1563, 1586, 1587, 1592, 1595, 1596, 1599, 1600, 1618, 1624, 1633, 1635, 1637, 1643, 1645, 1646, 1647, 1648, 1649, 1650, 1671, 1682, 1684, 1750, 1752, 1753, 1754, 1946, 1950, 1952, 1953, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1966, 1970, 1998, 1999, 2004, 2023, 2026, 2027, 2042, 2082, 2083, 2084, 2122)
        AND ac2.`model_name` LIKE '60%'
        THEN 10000
        ELSE uec.`agent_id`
    END AS `代理id`,
    uec.contract_id AS '合约id',
    bms.`device_id` AS `电池id`,
    bms.`totalvoltage` AS '电压',
    bms.`lastcapacity` AS '容量',
    bms.`rsoc` AS '电池SOC',
    - (
        CASE
            WHEN bms.`channel_id` IN ('GYDS','JXHC','SMKJ') THEN bms.`electric`/100
            ELSE ROUND(
                CASE
                    WHEN bms.`electric` >= 32768 THEN (bms.`electric` - 65536) / 100.0
                    ELSE bms.`electric` / 100.0
                END, 2)
        END
    ) AS `电流`,
    SUBSTRING_INDEX(
        CASE
            WHEN bms.`channel_id` IN ('GYDS','JXHC','SMKJ') THEN bms.`nntc_string`
            ELSE
                CONCAT_WS(
                ' ',
                CASE WHEN LENGTH(REPLACE(bms.`nntc_string`, ' ', '')) >= 4 THEN ROUND((CONV(SUBSTRING(REPLACE(bms.`nntc_string`, ' ', ''), 1, 4), 16, 10) - 2731) / 10, 1) ELSE NULL END,
                CASE WHEN LENGTH(REPLACE(bms.`nntc_string`, ' ', '')) >= 8 THEN ROUND((CONV(SUBSTRING(REPLACE(bms.`nntc_string`, ' ', ''), 5, 4), 16, 10) - 2731) / 10, 1) ELSE NULL END,
                CASE WHEN LENGTH(REPLACE(bms.`nntc_string`, ' ', '')) >= 12 THEN ROUND((CONV(SUBSTRING(REPLACE(bms.`nntc_string`, ' ', ''), 9, 4), 16, 10) - 2731) / 10, 1) ELSE NULL END,
                CASE WHEN LENGTH(REPLACE(bms.`nntc_string`, ' ', '')) >= 16 THEN ROUND((CONV(SUBSTRING(REPLACE(bms.`nntc_string`, ' ', ''), 13, 4), 16, 10) - 2731) / 10, 1) ELSE NULL END
            )
        END,
        ' ', 1
    ) AS `温度`,
    CASE WHEN UNIX_TIMESTAMP() - bms.`last_time` > 3600 THEN '离线' ELSE '在线' END AS `是否在线`,
    gps.`lat` AS `纬度`,
    gps.`lng` AS `经度`,
    gps.`speed` AS `速度`
FROM
    `xianglilai`.`xl_user_equipment_contract` uec
INNER JOIN `xianglilai`.`xl_device_bms` bms
    ON uec.`device_id` = bms.`device_id`
    AND bms.`cancellation` = 1
    AND bms.`channel_id` <> 'vr'
LEFT JOIN `xianglilai`.`xl_device_model` ac2
    ON uec.`model_id` = ac2.`id`
LEFT JOIN `xianglilai`.`xl_device_gps_now` gps
    ON uec.`device_id` = gps.`device_id`
WHERE
    uec.`deposit_status` = 1
    AND uec.`rent_status` = 1
    AND uec.`is_test` = 0
    AND uec.`begin_time` <= UNIX_TIMESTAMP()
    AND uec.`exp_time` >= UNIX_TIMESTAMP()
    AND uec.`contract_cg_mode` = 1
    AND uec.`ext_code_a` NOT IN ('a','b','t');
"""

VOLTAGE_THRESHOLD = 20
CAPACITY_THRESHOLD = 10
TEMP_THRESHOLD = 1

FORCE_SAVE_INTERVAL_SECONDS = 600

CACHE = {}
CACHE_LOCK = threading.Lock()

PENDING_DATA = []
PENDING_LOCK = threading.Lock()

CURRENT_DAY_PARQUET_PATH = None

TRACKED_FIELDS = [
    '电池id', '电压', '容量', '电池SOC', '电流', '温度', '是否在线', '经度', '纬度', '速度'
]


def wait_until_next_tick():
    now = datetime.now()
    if now.second < 30:
        target = now.replace(second=30, microsecond=0)
    else:
        target = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    wait_secs = (target - now).total_seconds()
    if wait_secs > 0:
        time.sleep(wait_secs)


def is_force_save_time(now):
    return now.minute % 10 == 0 and now.second == 0


def safe_numeric(val):
    try:
        return float(val)
    except:
        return None


def has_significant_change(old_record, new_record):
    old_voltage = safe_numeric(old_record.get('电压'))
    new_voltage = safe_numeric(new_record.get('电压'))
    if old_voltage is not None and new_voltage is not None:
        if abs(new_voltage - old_voltage) >= VOLTAGE_THRESHOLD:
            return True

    old_capacity = safe_numeric(old_record.get('容量'))
    new_capacity = safe_numeric(new_record.get('容量'))
    if old_capacity is not None and new_capacity is not None:
        if abs(new_capacity - old_capacity) >= CAPACITY_THRESHOLD:
            return True

    old_soc = safe_numeric(old_record.get('电池SOC'))
    new_soc = safe_numeric(new_record.get('电池SOC'))
    if old_soc is not None and new_soc is not None:
        if new_soc != old_soc:
            return True

    old_current = safe_numeric(old_record.get('电流'))
    new_current = safe_numeric(new_record.get('电流'))
    if new_current is not None and new_current != 0:
        return True

    old_temp = safe_numeric(old_record.get('温度'))
    new_temp = safe_numeric(new_record.get('温度'))
    if old_temp is not None and new_temp is not None:
        if abs(new_temp - old_temp) >= TEMP_THRESHOLD:
            return True

    if old_record.get('是否在线') != new_record.get('是否在线'):
        return True

    old_lat = safe_numeric(old_record.get('纬度'))
    new_lat = safe_numeric(new_record.get('纬度'))
    if old_lat is not None and new_lat is not None:
        if new_lat != old_lat:
            return True

    old_lng = safe_numeric(old_record.get('经度'))
    new_lng = safe_numeric(new_record.get('经度'))
    if old_lng is not None and new_lng is not None:
        if new_lng != old_lng:
            return True

    old_speed = safe_numeric(old_record.get('速度'))
    new_speed = safe_numeric(new_record.get('速度'))
    if new_speed is not None and new_speed != 0:
        return True

    if old_record.get('电池id') != new_record.get('电池id'):
        return True

    return False


def fetch_data():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        df = pd.read_sql(SQL_QUERY, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[{datetime.now()}] 数据库查询错误: {e}")
        return pd.DataFrame()


def update_cache_and_detect(df, force_save):
    global CACHE

    if df.empty:
        return []

    triggered_records = []
    now = datetime.now()

    with CACHE_LOCK:
        for _, row in df.iterrows():
            contract_id = row.get('合约id')
            if contract_id is None:
                continue

            record = row.to_dict()

            if contract_id not in CACHE:
                CACHE[contract_id] = {
                    'last_record': deepcopy(record),
                }
                triggered_records.append(record)
            else:
                cached = CACHE[contract_id]
                old_record = cached['last_record']

                if force_save or has_significant_change(old_record, record):
                    triggered_records.append(record)
                    cached['last_record'] = deepcopy(record)

    return triggered_records


def save_to_parquet(records):
    global CURRENT_DAY_PARQUET_PATH

    if not records:
        return

    today_str = datetime.now().strftime('%Y-%m-%d')
    today_parquet_path = os.path.join(EXPORT_PATH_DAILY_USER_DATA_RAW, f"battery_status_{today_str}.parquet")

    df_new = pd.DataFrame(records)

    with PENDING_LOCK:
        if CURRENT_DAY_PARQUET_PATH != today_parquet_path:
            CURRENT_DAY_PARQUET_PATH = today_parquet_path

        if os.path.exists(today_parquet_path):
            try:
                df_existing = pd.read_parquet(today_parquet_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except:
                df_combined = df_new
        else:
            df_combined = df_new

        os.makedirs(os.path.dirname(today_parquet_path), exist_ok=True)
        df_combined.to_parquet(today_parquet_path, engine='pyarrow', compression='snappy', index=False)

    print(f"[{datetime.now()}] 已保存 {len(records)} 条记录到 {today_parquet_path}")


def main_loop():
    global CURRENT_DAY_PARQUET_PATH

    print(f"[{datetime.now()}] 用户电池情况监控程序启动")
    print(f"轮询时刻: 每分钟 :30 秒 | 强制落库: 每10分钟整点 (00:00, 00:10, 00:20...)")
    print(f"输出路径: {EXPORT_PATH_DAILY_USER_DATA_RAW}")

    while True:
        try:
            wait_until_next_tick()

            now = datetime.now()
            if now.hour == 0 and now.minute == 0 and now.second < 5:
                CURRENT_DAY_PARQUET_PATH = None
                print(f"[{now}] 新的一天开始，重置parquet路径")

            force_save = is_force_save_time(now)
            print(f"[{now}] 开始查询 (强制落库={'是' if force_save else '否'})...")
            df = fetch_data()

            if df.empty:
                print(f"[{datetime.now()}] 本轮无数据，跳过")
                continue

            triggered = update_cache_and_detect(df, force_save)

            if triggered:
                save_to_parquet(triggered)
            else:
                print(f"[{datetime.now()}] 本轮无变化且未到强制落库时间，不入库")

        except Exception as e:
            print(f"[{datetime.now()}] 主循环异常: {e}")


if __name__ == "__main__":
    main_loop()
