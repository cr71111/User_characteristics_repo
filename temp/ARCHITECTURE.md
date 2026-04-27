# 两轮车换电用户分析系统 — 架构设计文档

**版本** v2.0 · **状态** 设计稿  
**适用范围** 全量用户7天滚动生命周期分析管道

---

## 目录

1. [设计目标](#1-设计目标)
2. [管道总览](#2-管道总览)
3. [层级职责与数据契约](#3-层级职责与数据契约)
4. [共享模块](#4-共享模块)
5. [目录结构](#5-目录结构)
6. [Parquet 分区策略](#6-parquet-分区策略)
7. [执行模式](#7-执行模式)
8. [关键设计原则](#8-关键设计原则)
9. [迁移步骤](#9-迁移步骤)

---

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| 单一职责 | 5个独立层，每层一个文件，职责边界清晰 |
| 幂等重算 | 每层以 Parquet 为边界，已存在则跳过，支持断点重跑 |
| 去重复代码 | 共享模块只保留一份，各层通过 import 引用 |
| 可追溯 | 每层输出 Parquet，出错可独立回放任意层 |
| 解耦计算与存储 | GPS围栏匹配、动态阈值等昂贵操作各自只执行一次 |

---

## 2. 管道总览

```
数据源
  │  IoT CSV 文件 / 合约信息表 / 三级 GeoJSON 围栏
  ▼
L0 raw_ingest.py          →  raw/{date}.parquet
  │  字段标准化 · 在线过滤 · GPS粗滤 · 去重
  ▼
L1 fact_layer.py          →  fact/daily/{date}.parquet
  │  骑行状态判定 · 物理指标计算 · GPS围栏匹配（仅此处做一次）
  ▼
L2 snapshot_layer.py      →  snapshot/user_daily.parquet（追加）
  │  合约→用户聚合 · 客户形态判定 · 日级评分
  ▼
L3 lifecycle_layer.py     →  lifecycle/user_7d.parquet
  │  7天滚动聚合 · 动态阈值 · 用户等级 · 风险标签
  ▼
L4 report_layer.py        →  reports/{date}/
     用户明细 CSV · 分布统计 · 风险告警列表
```

**单向依赖原则**：每一层只读上一层的 Parquet，不向下写，不回读自身历史输出。

---

## 3. 层级职责与数据契约

### L0 `raw_ingest.py`

**职责**：把原始 CSV 转成干净的 Parquet，不做任何业务推断。

**输入**
```
{EXPORT_PATH_BATTERY_STATUS_30D}/{date_folder}/*.csv
{EXPORT_PATH_BATTERY_STATUS_3D}/{date_folder}/*.csv
```

**处理步骤**

1. 读取 CSV，`low_memory=False`，指定 `dtype={'用户id': str, '合约id': str, '时间戳': int64}`
2. `normalize_column_names()` — 按 `COLUMN_NORMALIZE_MAP` 统一字段名
3. 检查 `REQUIRED_COLUMNS_RAW` 核心字段是否齐全
4. 过滤离线记录（`是否在线 not in ['在线','1','true','online','是']`）
5. 数值转换：`时间戳 / 纬度 / 经度 / 电流 / 温度 / 速度`
6. 去重：`subset=['合约id', '用户id', '时间戳']`
7. 粗过滤：中国境内经纬度范围 `lat∈[3,54], lon∈[73,136]`

**输出** `raw/{date}.parquet`

| 字段 | 类型 | 说明 |
|------|------|------|
| 时间戳 | int64 | Unix 秒 |
| 合约id | str | |
| 用户id | str | |
| 纬度 | float64 | |
| 经度 | float64 | |
| 电流 | float64 | 原始值，含充电负值 |
| 温度 | float64 | |
| 速度 | float64 | |
| 电池SOC | float64 | |
| 电池度数 | float64 | |
| 电池id | str | |
| 统计日期 | str | YYYY-MM-DD |

**幂等保证**：`raw/{date}.parquet` 已存在则整层跳过。

---

### L1 `fact_layer.py`

**职责**：从清洁原始行计算可观测的物理事实，不含任何业务推断。

**输入** `raw/{date}.parquet`

**处理步骤**

1. 电流分层清洗（骑行判定用 / 放电统计用 / 原始值）
2. 速度虚假值过滤
3. 骑行 / 放电状态判定（位移 + 电流）
4. 行程拆分（`MAX_TRIP_GAP_MIN` 间隔为断点）
5. 合约日级物理指标聚合
6. GPS 围栏匹配（`batch_gps_to_region`，**仅此层执行一次**）

**输出** `fact/daily/{date}.parquet`

| 字段组 | 代表字段 |
|--------|---------|
| 基础信息 | 统计日期, 合约id, 用户id |
| 地理 | 核心活动省份, 核心活动城市, 核心活动区县, 中心纬度, 中心经度 |
| 骑行 | 行驶距离, 骑行次数, 骑行总耗时_h, 平均骑行速度 |
| 电流 | 最大电流, 平均骑行电流, 电流标准差, 电流>60A次数, 电流>80A次数, 超100A连续次数 |
| SOC | 最低SOC, 平均骑行SOC, SOC低于20%时长占比, SOC低于10%时长占比 |
| 放电 | 总放电时长_h, 怠速放电时长_h |
| 电耗 | 总用电量_kWh, 百公里电耗_kWh, 换电次数 |
| 活动半径 | R90日常活动半径, R95核心活动半径, 最大活动半径_km, 凸包覆盖面积 |
| 时段分布 | 午间高峰骑行里程, 晚间高峰骑行里程, 平峰骑行里程, 夜间骑行里程 |

**幂等保证**：`fact/daily/{date}.parquet` 已存在则整层跳过。

---

### L2 `snapshot_layer.py`

**职责**：将合约日级事实聚合为用户日级快照，并做日级评分和客户形态判定。

**输入** `fact/daily/{date}.parquet`

**处理步骤**

1. 合约 → 用户聚合（多合约用户取 max / sum / 加权均值）
2. 客户形态判定（改装/超速车 → 地摊/储能 → 专送 / 众包 / 标准 / 普通骑手）
3. `_calc_monthly_score_v2()` 日级包月友好评分
4. 将新行追加写入 `snapshot/user_daily.parquet`，按 `[用户id, 统计日期]` 去重

**输出** `snapshot/user_daily.parquet`（追加写入，保留历史）

| 字段 | 聚合方式 |
|------|---------|
| 当日总行驶距离_km | sum |
| 当日总骑行时长_h | sum |
| 当日总放电时长_h | sum |
| 当日总怠速放电时长_h | sum |
| 当日最大电流_A | max |
| 当日最低SOC | min |
| 平均骑行电流_A | 骑行时长加权均值 |
| SOC低于20%时长占比 | 骑行时长加权均值 |
| 高峰骑行占比 | 骑行时长加权均值（非直接均值）|
| 客户形态_综合 | 优先级策略（改装 > 储能 > 按时长最多）|
| 包月友好评分 | `_calc_monthly_score_v2` |

---

### L3 `lifecycle_layer.py`

**职责**：7天滚动聚合 + 动态阈值计算 + 用户等级 / 风险标签。**唯一**调用 `dynamic_thresholds.py` 的层。

**输入** `snapshot/user_daily.parquet`（读取最近 30 天窗口）

**处理步骤**

1. 读取每个用户最近 7 天快照
2. 加权聚合 7d 滚动指标（近期权重高，见 `ROLLING_WEIGHTS`）
3. 月度用电预估（`calc_user_monthly_attendance`）
4. 调用 `DynamicBatteryAnalyzer.run()` 计算动态阈值
5. 调用 `score_and_classify()` 计算重评分数和用户等级
6. 合并合约信息（首次入网日期 → 新兵保护期判定）
7. 生命周期状态判定（新用户 / 活跃 / 轻度活跃 / 沉默 / 流失 / 已到期）

**用户等级分类优先级**

```
沉默用户  →  avg_cur=0 & max_cur=0 & monthly_energy=0 & ride_hours=0
暴力      →  over100≥2 | max_cur>P99 | monthly_energy>250，且非沉默
高风险上限 →  has_high_risk（超保护板/频繁超80A），最高到普通用户
中风险上限 →  has_medium_risk（月用电超标），最高到良好用户
高损耗强制 →  monthly_energy∈(150,250]，且非暴力/沉默/新兵
观察期    →  入网≤3天，且非暴力/沉默，result_level=''
正常分档  →  按 score 分档为优质 / 良好 / 普通 / 高损耗
```

**策略映射**

```python
STRATEGY_MAP = {
    '优质用户':   '留存激励',
    '良好用户':   '维持服务',
    '普通用户':   '引导升级',
    '高损耗用户': '限制预警',
    '暴力':       '清退处理',
    '观察期':     '新手引导',
    '沉默用户':   '激活唤醒',
}
```

**输出** `lifecycle/user_7d.parquet`

| 字段组 | 代表字段 |
|--------|---------|
| 7d 滚动指标 | 近7d平均骑行电流_A, 近7d最大电流_A, 近7d百公里电耗_kWh, 近7d最低SOC, … |
| 月度预估 | 月度预估工作天数, 单合约月度用电度数预估_kWh |
| 动态评级 | 重评分数, 用户等级_动态, 风险标签, 策略建议 |
| 等级说明 | 用户等级_动态说明 |
| 生命周期 | 用户生命周期状态_7d, 设备状态监控 |
| 对比 | 等级变化（与上期对比，可选）|

---

### L4 `report_layer.py`

**职责**：从生命周期 Parquet 生成各类面向业务的输出文件，不做任何计算。

**输入** `lifecycle/user_7d.parquet`

**输出**

```
reports/{date}/
├── user_detail.csv          # 用户明细（等级、风险标签、策略建议）
├── level_distribution.csv   # 用户等级分布统计
├── risk_alerts.csv          # 风险用户告警列表（暴力 + 高损耗）
├── shift_report.json        # 分布漂移检测报告
└── summary.txt              # 文本摘要，用于日报
```

---

## 4. 共享模块

### `dynamic_thresholds.py`

只在 L3 中被引用，其他层禁止导入。

| 类 | 职责 |
|----|------|
| `BaselineStore` | 读写 `thresholds_baseline.json`；`update_ema` 只更新内存，由 `save_current_data()` 批量落盘 |
| `AdaptiveThresholds` | 从 DataFrame 计算分位数阈值 |
| `DistributionShiftDetector` | 与基准线对比，输出漂移方向和建议 |
| `DynamicBatteryAnalyzer` | 整合入口，提供 `run()` + `score_and_classify()` |

### `utils/geo.py`

GPS 围栏加载和空间匹配逻辑，提供 `load_fence_data()` 和 `batch_gps_to_region()`。
从 L1 提取为独立模块，其他层禁止重复实现。

### `config/config.py`

所有路径和阈值常量的唯一来源，各层通过 import 使用，不在层内硬编码。

---

## 5. 目录结构

```
battery_repo/
├── src/
│   └── scheduled_tasks/
│       └── daily_2/
│           ├── raw_ingest.py           # L0
│           ├── fact_layer.py           # L1
│           ├── snapshot_layer.py       # L2
│           ├── lifecycle_layer.py      # L3
│           ├── report_layer.py         # L4
│           ├── dynamic_thresholds.py   # 共享模块（唯一）
│           ├── thresholds_baseline.json
│           └── run_pipeline.py         # 主入口
├── config/
│   └── config.py
├── utils/
│   └── geo.py
└── data/
    ├── raw/
    │   └── {date}.parquet
    ├── fact/
    │   └── daily/
    │       └── {date}.parquet
    ├── snapshot/
    │   └── user_daily.parquet
    ├── lifecycle/
    │   └── user_7d.parquet
    └── reports/
        └── {date}/
```

---

## 6. Parquet 分区策略

| 文件 | 写入模式 | 压缩 | 说明 |
|------|---------|------|------|
| `raw/{date}.parquet` | 覆盖写 | snappy | 每日独立文件 |
| `fact/daily/{date}.parquet` | 覆盖写 | snappy | 每日独立文件 |
| `snapshot/user_daily.parquet` | 追加写 | zstd | 按 `统计日期` 分区，历史累积 |
| `lifecycle/user_7d.parquet` | 覆盖写 | zstd | 全量快照，每次全量重写 |

**追加写入模式**（L2）

```python
import pyarrow as pa
import pyarrow.parquet as pq

def append_to_snapshot(df_new: pd.DataFrame, path: str) -> None:
    table_new = pa.Table.from_pandas(df_new)
    if os.path.exists(path):
        table_old = pq.read_table(path)
        df = pa.concat_tables([table_old, table_new]).to_pandas()
        df = df.drop_duplicates(subset=['用户id', '统计日期'], keep='last')
        table_new = pa.Table.from_pandas(df)
    pq.write_table(table_new, path, compression='zstd')
```

---

## 7. 执行模式

通过 `run_pipeline.py` 统一入口，`--mode` 参数控制执行范围：

| 模式 | 执行层 | 适用场景 |
|------|--------|---------|
| `full` | L0 → L4 | 首次运行，全量重建 |
| `incremental` | L0 → L4，仅处理新日期 | 日常定时任务 |
| `refresh` | L2 → L4 | 调整聚合逻辑后重跑 |
| `recalculate` | L3 → L4 | 调整评级规则后重跑 |
| `report_only` | L4 | 仅重新生成报告文件 |

```bash
python run_pipeline.py --mode incremental --date 2026-04-19
python run_pipeline.py --mode recalculate
python run_pipeline.py --mode report_only --date 2026-04-19
```

---

## 8. 关键设计原则

### 8.1 每层单一职责

- L0：原始清洗，不计算业务指标
- L1：物理事实，不推断客户意图
- L2：合约聚合和日级业务判断，不做跨日计算
- L3：时序聚合和动态评级，不生成最终报告
- L4：格式化输出，不做任何计算

### 8.2 昂贵操作各做一次

GPS 围栏空间匹配（`sjoin`）只在 L1 执行，结果写入 Parquet。动态阈值计算只在 L3 执行，结果写入 Parquet。L2 / L4 直接读字段，不重复计算。

### 8.3 共享模块唯一化

`dynamic_thresholds.py` 只保留一份，其他文件中的内联类定义全部删除，改为 import 引用。`utils/geo.py` 同理。

### 8.4 配置集中管理

所有文件路径从 `config.py` 读取，所有阈值在配置区集中定义，不在函数内散落硬编码值。

### 8.5 幂等性

每层在执行前检查输出 Parquet 是否已存在，存在则跳过，支持任意中断后重跑。

---

## 9. 迁移步骤

### Phase 1：提取共享模块

1. 新建 `utils/geo.py`，将各文件中的 `load_fence_data()` 和 `batch_gps_to_region()` 合并为一份
2. 确认 `dynamic_thresholds.py` 为唯一的阈值模块，删除其他文件中的内联类定义

### Phase 2：引入 Parquet 中间层

1. 在 `raw_ingest.py` 中，将清洁后的 DataFrame 写入 `raw/{date}.parquet`
2. `fact_layer.py` 读取 Parquet 输入，输出写到 `fact/daily/{date}.parquet`
3. 验证各层 Parquet 字段与原版 CSV 字段的一致性

### Phase 3：拆分业务层

1. 将合约聚合、客户形态、日级评分迁移到独立的 `snapshot_layer.py`
2. 将 7 天滚动、动态阈值、用户等级迁移到独立的 `lifecycle_layer.py`
3. 新建 `report_layer.py`，将现有报告输出逻辑迁移至此

### Phase 4：整合主入口

1. 新建 `run_pipeline.py`，实现 `--mode` 参数控制执行范围
2. 将 `main()` / `refresh_lifecycle_only()` / `recalculate_customer_profile()` 合并为单一入口
3. 端到端验证：新旧管道的用户等级分布差异应在可接受范围内

---

*文档生成时间：2026-04-20*
