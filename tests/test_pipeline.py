# -*- coding: utf-8 -*-
"""
端到端管线测试脚本
模拟包含改装车、正常电动车等多种用户类型的 mock 数据，验证全流程逻辑正确性。

用法: python tests/test_pipeline.py
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.fact_layer import preprocess_raw_data, calc_contract_metrics
from src.pipeline.snapshot_layer import _get_user_type, _get_vehicle_type

TEST_DATA_DIR = tempfile.mkdtemp(prefix='pipeline_test_')


def setup_directory():
    os.makedirs(os.path.join(TEST_DATA_DIR, 'battery_status'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, 'fact', 'daily'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, 'snapshot'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, 'lifecycle'), exist_ok=True)


def generate_gps_track(base_lat, base_lon, n_points, pattern='commute'):
    lats = np.zeros(n_points)
    lons = np.zeros(n_points)

    for i in range(n_points):
        if pattern == 'stationary':
            lats[i] = base_lat
            lons[i] = base_lon
        elif pattern == 'commute':
            progress = (i % 180) / 180.0
            if progress < 0.5:
                p = progress * 2
                lats[i] = base_lat + p * 0.02
            else:
                p = (1.0 - progress) * 2
                lats[i] = base_lat + p * 0.02
            lons[i] = base_lon + progress * 0.02
        elif pattern == 'loop':
            angle = 2 * np.pi * (i % 360) / 360.0
            radius = 0.01
            lats[i] = base_lat + radius * np.sin(angle)
            lons[i] = base_lon + radius * np.cos(angle)
        elif pattern == 'wide':
            progress = i / n_points
            angle = 2 * np.pi * progress
            radius = 0.03 * progress
            lats[i] = base_lat + radius * np.sin(angle)
            lons[i] = base_lon + radius * np.cos(angle)
        elif pattern == 'delivery_loop':
            base_progress = (i // 30) % 8
            offset_lat = [0, 0.005, 0.008, 0.003, -0.002, -0.005, -0.003, 0.001]
            offset_lon = [0, 0.003, 0.006, 0.010, 0.008, 0.005, 0.001, -0.002]
            lats[i] = base_lat + offset_lat[base_progress]
            lons[i] = base_lon + offset_lon[base_progress]

    return lats, lons


def create_mock_raw_data(n_days=7):
    base_date = datetime(2026, 5, 20)
    all_data = []

    users = [
        {
            'user_id': 'USER_MODIFY_01',
            'contract_id': 'CT_MODIFY_01',
            'pattern': 'commute',
            'base_lat': 31.25, 'base_lon': 121.48,
            'speed_kmh': 65, 'avg_current': 30, 'max_current': 65,
            'label': '改装/超速车（高速高电流，车辆形态=改装/超速车，客户形态=专送骑手）',
        },
        {
            'user_id': 'USER_EBIKE_02',
            'contract_id': 'CT_EBIKE_02',
            'pattern': 'commute',
            'base_lat': 31.20, 'base_lon': 121.55,
            'speed_kmh': 18, 'avg_current': 8, 'max_current': 15,
            'label': '电动自行车（≤25km/h，车辆=电动自行车，客户=标准骑手）',
        },
        {
            'user_id': 'USER_EMOPED_03',
            'contract_id': 'CT_EMOPED_03',
            'pattern': 'loop',
            'base_lat': 31.22, 'base_lon': 121.50,
            'speed_kmh': 35, 'avg_current': 15, 'max_current': 30,
            'label': '电动轻便摩托车（25-50km/h，车辆=电动轻便摩托车，客户=众包骑手）',
        },
        {
            'user_id': 'USER_EMOTO_04',
            'contract_id': 'CT_EMOTO_04',
            'pattern': 'wide',
            'base_lat': 31.18, 'base_lon': 121.60,
            'speed_kmh': 55, 'avg_current': 10, 'max_current': 20,
            'label': '电动摩托车（>50km/h但低电流，车辆=电动摩托车，客户=普通骑手）',
        },
        {
            'user_id': 'USER_STORAGE_05',
            'contract_id': 'CT_STORAGE_05',
            'pattern': 'stationary',
            'base_lat': 31.30, 'base_lon': 121.42,
            'speed_kmh': 0, 'avg_current': 12, 'max_current': 18,
            'label': '地摊/储能（GPS静止+持续放电，车辆=数据不足，客户=地摊/储能）',
        },
        {
            'user_id': 'USER_DELIVERY_06',
            'contract_id': 'CT_DELIVERY_06',
            'pattern': 'delivery_loop',
            'base_lat': 31.23, 'base_lon': 121.47,
            'speed_kmh': 20, 'avg_current': 10, 'max_current': 25,
            'label': '专送骑手（长时间骑行+高峰集中，车辆=电动自行车，客户=专送骑手）',
        },
        {
            'user_id': 'USER_CROWD_07',
            'contract_id': 'CT_CROWD_07',
            'pattern': 'wide',
            'base_lat': 31.26, 'base_lon': 121.52,
            'speed_kmh': 40, 'avg_current': 14, 'max_current': 30,
            'label': '众包骑手（分散骑行，车辆=电动轻便摩托车，客户=众包骑手）',
        },
    ]

    for day_offset in range(n_days):
        stat_date = base_date + timedelta(days=day_offset)
        stat_date_str = stat_date.strftime('%Y-%m-%d')

        for user in users:
            daily_contract = f"{user['contract_id']}_D{day_offset}"
            base_ts = int(stat_date.replace(hour=3, minute=0, second=0).timestamp())
            n_points = 1200

            lats, lons = generate_gps_track(
                user['base_lat'], user['base_lon'], n_points, user['pattern']
            )

            SOC_VALUES = np.linspace(95, 25, n_points)
            for i in range(0, n_points, 200):
                if i + 200 < n_points:
                    SOC_VALUES[i:i + 200] = np.linspace(95, 25, 200)

            for i in range(n_points):
                ts = base_ts + i * 60
                hour = datetime.fromtimestamp(ts).hour

                if user['user_id'] == 'USER_STORAGE_05':
                    speed_val = 0.0
                    soc = 30 + np.sin(2 * np.pi * i / n_points) * 15
                    current = 12 + np.random.normal(0, 1)
                    current = max(0, current)
                elif user['user_id'] == 'USER_MODIFY_01':
                    speed_val = user['speed_kmh'] + np.random.normal(0, 2)
                    speed_val = max(speed_val, 0)
                    if i % 15 == 0:
                        speed_val = max(speed_val, np.random.uniform(55, 75))
                    soc = SOC_VALUES[i % len(SOC_VALUES)]
                    current = abs(np.random.normal(user['avg_current'], 4))
                    if i % 15 == 0:
                        current = max(current, np.random.uniform(30, 70))
                elif user['user_id'] == 'USER_DELIVERY_06':
                    if hour in [11, 12, 13, 17, 18, 19]:
                        speed_val = user['speed_kmh'] + np.random.normal(0, 2)
                        speed_val = max(speed_val, 0)
                    elif hour < 6 or hour > 21:
                        speed_val = 0.0
                    else:
                        speed_val = user['speed_kmh'] * 0.4 + np.random.normal(0, 1)
                        speed_val = max(speed_val, 0)
                    current = abs(np.random.normal(user['avg_current'], 2)) if speed_val > 0 else 0.1
                    soc = SOC_VALUES[i % len(SOC_VALUES)]
                elif user['user_id'] == 'USER_CROWD_07':
                    if i % 7 == 0 and np.random.random() < 0.3:
                        speed_val = 0.0
                        current = 0.5
                    else:
                        speed_val = user['speed_kmh'] + np.random.normal(0, 3)
                        speed_val = max(speed_val, 0)
                        current = abs(np.random.normal(user['avg_current'], 3))
                    soc = SOC_VALUES[i % len(SOC_VALUES)]
                else:
                    speed_val = user['speed_kmh'] + np.random.normal(0, 1)
                    speed_val = max(speed_val, 0)
                    current = abs(np.random.normal(user['avg_current'], 2))
                    soc = SOC_VALUES[i % len(SOC_VALUES)]

                all_data.append({
                    '统计日期': stat_date_str,
                    '合约id': daily_contract,
                    '用户id': user['user_id'],
                    '时间戳': ts,
                    '纬度': lats[i],
                    '经度': lons[i],
                    '电流': float(current),
                    '速度': float(speed_val),
                    '温度': 25.0,
                    '电池SOC': float(max(0, min(100, soc))),
                    '电池id': 'BAT001',
                    '代理id': 'AGENT_001',
                    '是否在线': 1,
                })

    df = pd.DataFrame(all_data)
    return df, users


def run_fact_layer(df_raw, battery_voltage_map=None):
    """运行Fact Layer并返回结果"""
    print("\n" + "=" * 70)
    print("  L1 Fact Layer 测试")
    print("=" * 70)

    df_clean = preprocess_raw_data(df_raw)
    print(f"  清洗后: {len(df_clean):,} 行")

    df_fact = calc_contract_metrics(df_clean, battery_voltage_map or {})
    print(f"  合约指标: {len(df_fact):,} 条记录")

    return df_fact


def run_snapshot_layer(df_fact):
    """模拟Snapshot Layer聚合"""
    print("\n" + "=" * 70)
    print("  L2 Snapshot Layer 测试")
    print("=" * 70)

    df_fact = df_fact.copy()
    df_fact['用户id'] = df_fact['用户id'].astype(str).fillna('未知用户')
    df_fact['weighted_current'] = df_fact['骑行放电平均电流'] * df_fact['骑行总耗时(小时)']

    all_user_rows = []

    for date_str, df_day in df_fact.groupby('统计日期'):
        print(f"  处理日期: {date_str} ({len(df_day)} 合约)")

        g = df_day.groupby(['统计日期', '用户id'])

        df_agg = pd.DataFrame()
        df_agg['统计日期'] = g['统计日期'].first()
        df_agg['用户id'] = g['用户id'].first()
        df_agg['关联合约数'] = g['合约id'].nunique()
        df_agg['当日总行驶距离_km'] = g['行驶距离'].sum()
        df_agg['当日总骑行时长_h'] = g['骑行总耗时(小时)'].sum()
        df_agg['当日总放电时长_h'] = g['总放电时长(小时)'].sum()
        df_agg['当日总怠速放电时长_h'] = g['怠速放电时长(小时)'].sum()

        df_agg['单合约日均行驶里程_km'] = (df_agg['当日总行驶距离_km'] / df_agg['关联合约数'].clip(lower=1)).round(2)
        df_agg['单合约日均骑行时长_h'] = (df_agg['当日总骑行时长_h'] / df_agg['关联合约数'].clip(lower=1)).round(2)
        df_agg['单合约日均怠速放电_h'] = (df_agg['当日总怠速放电时长_h'] / df_agg['关联合约数'].clip(lower=1)).round(2)

        df_agg['当日总骑行次数'] = g['骑行次数'].sum()
        df_agg['当日是否出勤'] = (df_agg['当日总骑行次数'] > 0).astype(int)
        df_agg['当日总换电次数'] = g['换电次数'].sum()

        df_agg['当日总用电量_kWh'] = g['总用电量(kWh)'].sum()
        df_agg['单合约日均用电量_kWh'] = (df_agg['当日总用电量_kWh'] / df_agg['关联合约数'].clip(lower=1)).round(2)

        df_agg['午间高峰长时骑行总次数'] = g['午间高峰长时骑行次数'].sum()
        df_agg['晚间高峰长时骑行总次数'] = g['晚间高峰长时骑行次数'].sum()
        df_agg['平峰长时骑行总次数'] = g['平峰长时骑行次数'].sum()
        df_agg['夜间长时骑行总次数'] = g['夜间长时骑行次数'].sum()

        df_agg['午间高峰总里程_km'] = g['午间高峰骑行里程'].sum()
        df_agg['晚间高峰总里程_km'] = g['晚间高峰骑行里程'].sum()
        df_agg['平峰总里程_km'] = g['平峰骑行里程'].sum()
        df_agg['夜间总里程_km'] = g['夜间骑行里程'].sum()

        df_agg['当日高峰骑行占比'] = (
            (df_agg['午间高峰总里程_km'] + df_agg['晚间高峰总里程_km'])
            / df_agg['当日总行驶距离_km'].clip(lower=0.01)
        )

        df_agg['当日平峰换电次数'] = g['平峰换电次数'].sum()
        df_agg['当日深夜换电次数'] = g['深夜换电次数'].sum()
        df_agg['当日高峰换电次数'] = g['高峰换电次数'].sum()
        df_agg['当日高速骑行点数'] = g['高速骑行点数(>40kmh)'].sum()

        df_agg['当日平均骑行电流_A'] = g['骑行放电平均电流'].mean()
        df_agg['当日百公里电耗_kWh'] = (
            (df_agg['当日总用电量_kWh'] / df_agg['当日总行驶距离_km'].clip(lower=0.01)) * 100
        )

        df_agg['当日_R90活动半径_km'] = g['R90日常活动半径'].max()
        df_agg['当日凸包覆盖面积_km2'] = g['凸包覆盖面积'].max()

        if '上线时间熵值' in df_fact.columns:
            df_agg['当日上线时间熵值'] = g['上线时间熵值'].max()
        if '骑行时段集中度' in df_fact.columns:
            df_agg['当日骑行时段集中度'] = g['骑行时段集中度'].mean()
        if '路线曲折系数' in df_fact.columns:
            df_agg['当日路线曲折系数'] = g['路线曲折系数'].mean()
        if '速度变异系数' in df_fact.columns:
            df_agg['当日速度变异系数'] = g['速度变异系数'].mean()
        if '跨区域转移次数' in df_fact.columns:
            df_agg['当日跨区域转移次数'] = g['跨区域转移次数'].max()
        if '静止时长占比' in df_fact.columns:
            df_agg['当日静止时长占比'] = g['静止时长占比'].mean()

        df_custom = g.apply(lambda x: pd.Series({
            '车辆形态_综合': _get_vehicle_type(x),
            '客户形态_综合': _get_user_type(x),
        }), include_groups=False).reset_index(drop=True)

        df_agg = df_agg.reset_index(drop=True)
        df_agg = pd.concat([df_agg, df_custom], axis=1)

        all_user_rows.append(df_agg)

    df_result = pd.concat(all_user_rows, ignore_index=True)
    print(f"  Snapshot完成: {len(df_result)} 条用户日记录")
    return df_result


def run_lifecycle_layer(df_snapshot):
    """运行Lifecycle Layer 7天滚动计算"""
    print("\n" + "=" * 70)
    print("  L3 Lifecycle Layer - 7天滚动指标")
    print("=" * 70)

    from src.pipeline.lifecycle_layer import _calc_single_user_7d_rolling, ROLLING_WINDOW_DAYS

    df_snapshot = df_snapshot.copy()
    df_snapshot['统计日期'] = pd.to_datetime(df_snapshot['统计日期'])

    all_users = df_snapshot['用户id'].unique()
    results = []
    latest_date = df_snapshot['统计日期'].max()

    for user_id in all_users:
        user_data = df_snapshot[df_snapshot['用户id'] == user_id].sort_values('统计日期')
        if len(user_data) < 2:
            continue
        res = _calc_single_user_7d_rolling(user_data, user_id, latest_date)
        results.append(res)

    df_result = pd.DataFrame(results)
    print(f"  Lifecycle完成: {len(df_result)} 个用户")

    return df_result


def main():
    print("=" * 70)
    print("  用户特征画像管线 - 全流程测试")
    print("=" * 70)

    setup_directory()

    print("\n[1/5] 生成 Mock 数据（7天 × 7用户）...")
    df_raw, users_config = create_mock_raw_data(n_days=7)
    print(f"  生成 {len(df_raw):,} 行原始数据")
    for u in users_config:
        print(f"    {u['user_id']}: {u['label']}")

    print("\n[2/5] 运行 Fact Layer...")
    df_fact = run_fact_layer(df_raw)

    print("\n[3/5] 运行 Snapshot Layer...")
    df_snapshot = run_snapshot_layer(df_fact)

    print("\n[4/5] 运行 Lifecycle Layer...")
    df_lifecycle = run_lifecycle_layer(df_snapshot)

    print("\n" + "=" * 70)
    print("  [5/5] 结果验证")
    print("=" * 70)

    print("\n─── Fact Layer 核心指标 ───")
    for _, row in df_fact.iterrows():
        uid = row['用户id']
        if not uid.startswith('USER_'):
            continue
        vt = row.get('车辆形态', 'N/A')
        ct = row.get('客户形态', 'N/A')
        ms = row.get('最大速度', 0)
        ac = row.get('骑行放电平均电流', 0)
        pp = row.get('峰值功率_W', 0)
        td = row.get('行驶距离', 0)
        rh = row.get('骑行总耗时(小时)', 0)
        pr = row.get('高峰骑行占比', 0)
        dh = row.get('怠速放电时长(小时)', 0)

        ent = row.get('上线时间熵值', 'N/A')
        det = row.get('路线曲折系数', 'N/A')
        cv = row.get('速度变异系数', 'N/A')
        cr = row.get('跨区域转移次数', 'N/A')
        ir = row.get('静止时长占比', 'N/A')

        print(f"\n  {uid}:")
        print(f"    车辆形态={vt}, 客户形态={ct}")
        print(f"    最高速度={ms:.0f}km/h, 平均骑行电流={ac:.1f}A, 峰值功率={pp:.0f}W")
        print(f"    行驶距离={td:.1f}km, 骑行时长={rh:.1f}h, 高峰占比={pr:.1%}")
        print(f"    怠速放电={dh:.1f}h")
        print(f"    新特征: 时间熵={ent}, 曲折={det}, 速度CV={cv}, 跨区={cr}, 静止占比={ir}")

    if df_lifecycle is not None and len(df_lifecycle) > 0:
        print("\n─── Lifecycle Layer 7天汇总 ───")
        for _, row in df_lifecycle.iterrows():
            uid = row.get('用户id', 'N/A')
            vt7 = row.get('车辆形态_7d', 'N/A')
            ct7 = row.get('客户形态_综合_7d', 'N/A')
            ms7 = row.get('近7d最高速度_kmh', 0)
            ent7 = row.get('近7d平均上线时间熵值', 'N/A')
            print(f"  {uid}: 车辆={vt7}, 客户={ct7}, 最高速度={ms7:.0f}km/h, 熵={ent7}")

    print("\n─── 验证结论 ───")

    checks = []

    modify_rows = df_fact[df_fact['用户id'].str.contains('MODIFY')]
    if len(modify_rows) > 0:
        vt = modify_rows.iloc[0]['车辆形态']
        exp = '改装/超速车'
        ok = vt == exp
        checks.append((f'改装车车辆形态={vt}', ok, exp))

    ebike_rows = df_fact[df_fact['用户id'].str.contains('EBIKE')]
    if len(ebike_rows) > 0:
        vt = ebike_rows.iloc[0]['车辆形态']
        exp = '电动自行车'
        ok = vt == exp
        checks.append((f'电动自行车车辆形态={vt}', ok, exp))

    emoped_rows = df_fact[df_fact['用户id'].str.contains('EMOPED')]
    if len(emoped_rows) > 0:
        vt = emoped_rows.iloc[0]['车辆形态']
        exp = '电动轻便摩托车'
        ok = vt == exp
        checks.append((f'电动轻便摩托车车辆形态={vt}', ok, exp))

    emoto_rows = df_fact[df_fact['用户id'].str.contains('EMOTO')]
    if len(emoto_rows) > 0:
        vt = emoto_rows.iloc[0]['车辆形态']
        exp = '电动摩托车'
        ok = vt == exp
        checks.append((f'电动摩托车车辆形态={vt}', ok, exp))

    storage_rows = df_fact[df_fact['用户id'].str.contains('STORAGE')]
    if len(storage_rows) > 0:
        ct = storage_rows.iloc[0]['客户形态']
        exp = '地摊/储能'
        ok = ct == exp
        checks.append((f'地摊储能客户形态={ct}', ok, exp))

    delivery_rows = df_fact[df_fact['用户id'].str.contains('DELIVERY')]
    if len(delivery_rows) > 0:
        ct = delivery_rows.iloc[0]['客户形态']
        ok = ct in ('专送骑手', '众包骑手')
        checks.append((f'骑手客户形态={ct}', ok, '专送骑手或众包骑手'))

    for check, ok, expected in checks:
        status = '✅' if ok else '❌'
        print(f"  {status} {check} (期望: {expected})")

    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        print("\n  🎉 全部验证通过！")
    else:
        print("\n  ⚠️ 部分验证未通过，请检查日志")

    print(f"\n  测试数据保留在: {TEST_DATA_DIR}")
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)