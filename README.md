# 两轮车换电用户特征画像系统 - 项目技术文档 v2.2

> **最后更新:** 2026-05-11  
> **代码版本:** v2.2 (Pipeline层参数优化、OneDrive文件锁兼容、文档版本统一)  
> **项目路径:** `d:\PY代码\用户特征画像\User_characteristics_repo`

---

## 📋 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 核心模块详解](#3-核心模块详解)
  - [L1: Fact Layer](#l1-fact-layer-客观事实层)
  - [L2: Snapshot Layer](#l2-snapshot-layer-用户日级快照层)
  - [L3: Lifecycle Layer](#l3-lifecycle-layer-用户生命周期层)
  - [L4: Report Layer](#l4-report-layer-报告输出层)
- [4. 动态阈值系统](#4-动态阈值系统-dynamicthresholds)
- [5. Pipeline执行模式](#5-pipeline-执行模式)
- [6. 关键算法说明](#6-关键算法说明)
- [7. 项目文件结构](#7-项目文件结构)
- [8. 快速开始](#8-快速开始)
- [9. 常见问题排查](#9-常见问题排查)

---

## 1. 项目概述

### 1.1 系统定位
本系统是一个**两轮车换电用户行为分析与生命周期管理系统**，通过对电池使用数据的采集、处理、分析，实现：

| 核心能力 | 业务价值 |
|----------|----------|
| 用户行为特征画像构建 | 多维度刻画骑行习惯、用电规律、地理活动范围 |
| 生命周期状态识别与分类 | 支撑用户分层运营策略 |
| 动态风险预警与策略建议 | 及时发现异常行为，降低运营风险 |
| 运营决策数据支撑 | 为BI系统、CRM提供数据基础 |

### 1.2 技术栈
```
语言: Python 3.13+
数据处理: pandas, numpy, pyarrow (Parquet格式)
地理计算: shapely, geojson (GPS空间匹配)
科学计算: scipy (凸包计算)
文件格式: Parquet (高效压缩), CSV (报告输出), JSON (配置/基准线)
调度方式: 命令行 Pipeline (支持5种执行模式)
进度显示: tqdm (处理进度条)
```

### 1.3 版本历史
| 版本 | 日期 | 主要更新 |
|------|------|----------|
| v1.0 | 2026-04 | 初始版本，基础架构搭建 |
| v1.1 | 2026-04 | 添加动态阈值系统和用户评分 |
| v1.2 | 2026-04 | 实现梯形数值积分法计算用电量 |
| v1.3 | 2026-04 | 月度用电量预估改用出勤率方法 |
| v1.4 | 2026-04 | 启用EMA自动更新基准线 |
| **v1.5** | **2026-04-27** | **数据源迁移至Parquet、增量模式优化、清理废弃代码** |
| **v1.6** | **2026-05-07** | **代码质量重构 + 阈值常量统一 + 逻辑缺陷修复** |
| **v1.7** | **2026-05-08** | **用电量计算逻辑重构：SOC差法替代梯形积分 + 离线数据过滤** |
| **v2.0** | **2026-05-09** | **用电量计算优化：按电池段首尾SOC差值一次性计算，取消逐行累加** |
| **v2.1** | **2026-05-10** | **代码审查与BUG修复：评分逻辑统一、返回类型修复、默认模式修正、骑行时刻精度、出勤判定优化、OneDrive兼容** |
| **v2.2** | **2026-05-11** | **Pipeline层参数优化、OneDrive文件锁兼容、文档版本统一** |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据源层 (Data Source)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ battery_status_2026-04-24.parquet                          │   │
│  │ battery_status_2026-04-25.parquet   ← 每日原始数据         │   │
│  │ battery_status_2026-04-26.parquet     (Parquet格式)        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────┐  ┌───────────────────────────────────────┐   │
│  │ 4-5用户最早      │  │ 中国_省.geojson                       │   │
│  │ 合约时间.csv     │  │ 中国_市.geojson   ← 辅助数据          │   │
│  │                 │  │ 中国_县.geojson                       │   │
│  └─────────────────┘  └───────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 电池单体电压.csv  ← 电池规格信息 (v1.7+)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    数据处理流水线 (Pipeline)                          │
│              run_pipeline.py (统一调度入口)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐   ┌───────────┐   ┌────────────┐   ┌──────────┐    │
│   │  L1     │ → │   L2     │ → │    L3      │ → │   L4     │    │
│   │ Fact    │   │ Snapshot │   │ Lifecycle  │   │ Report   │    │
│   │ Layer   │   │  Layer   │   │   Layer    │   │  Layer   │    │
│   └─────────┘   └───────────┘   └────────────┘   └──────────┘    │
│      ↓              ↓               ↓               ↓            │
│   日级事实       7天滚动快照      评分+分类+画像    多格式报告      │
│   指标计算       聚合             生命周期判定      可视化就绪      │
│                                                                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        输出层 (Output)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  data/fact/daily/                ← L1 日级事实 (按日期分区)        │
│  data/snapshot/user_daily.parquet← L2 用户7天滚动快照               │
│  data/lifecycle/user_7d.parquet  ← L3 完整用户画像                 │
│  data/reports/                   ← L4 各类分析报告                  │
│  src/thresholds_baseline.json    ← 动态阈值基准线 (EMA自动更新)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流总览

```
原始数据采集 → L1日级指标计算 → L2 7天滚动聚合 → L3评分分类画像 → L4多格式报告
     ↓                ↓                  ↓                ↓              ↓
  parquet文件      合约级指标          用户级快照        97+字段        CSV/JSON/TXT
  (每日期1个)      (电压/电流/SOC)    (近7天统计)      (画像文本)     (业务就绪)
```

### 2.3 架构设计原则

| 原则 | 说明 | 实现位置 |
|------|------|----------|
| **分层解耦** | 每层职责单一，可独立重跑 | L1-L4各层独立函数 |
| **增量处理** | 仅处理新日期数据，避免重复计算 | incremental模式 |
| **容错设计** | 单日失败不影响整体，支持断点续跑 | try-except + 日志记录 |
| **可扩展性** | 新增指标只需修改对应层的计算函数 | 各层独立配置参数 |
| **性能优化** | Parquet列式存储 + 批量处理 + 并行计算 | fact_layer批量处理 |

---

## 3. 核心模块详解

### 3.1 L1: Fact Layer (客观事实层)

**📁 文件位置:** [src/fact_layer.py](src/fact_layer.py)  
**⚙️ 入口函数:** `process_fact_layer(target_date)`  
**🎯 职责:** 将每日原始数据转换为结构化的合约日级指标

#### 3.1.1 数据输入

**数据源:** 每日一个 Parquet 文件  
**命名规则:** `battery_status_YYYY-MM-DD.parquet`  
**存储位置:** `{BASE_EXPORT_PATH}/基础数据/用户电池情况/`

**原始字段清单:**
| 字段名 | 类型 | 说明 |
|--------|------|------|
| 统计日期 | date | 数据日期 |
| 统计时间 | time | 记录时间 |
| 时间戳 | datetime | 精确时间点 |
| 用户id | str | 用户唯一标识 |
| 代理id | str | 代理商ID |
| 合约id | str | 合约唯一标识 |
| 电池id | str | 电池唯一标识 |
| 电压(mV) | float | 实时电压 (毫伏) |
| 电流(A) | float | 实时电流 (安培) |
| 容量 | float | 电池容量 |
| 电池SOC(%) | float | 电量状态百分比 |
| 温度(°C) | float | 电池温度 |
| 是否在线 | bool | 在线状态 |
| 纬度 | float | GPS纬度 |
| 经度 | float | GPS经度 |
| 速度(km/h) | float | 实时速度 |

#### 3.1.2 处理流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 原始数据加载 │ →  │  数据清洗   │ →  │ GPS空间匹配 │ →  │ 用电量计算  │
│ read_parquet│    │ preprocess  │    │geo_to_region│    │trapezoidal  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                           │
                                                           ▼
                                                   ┌─────────────┐
                                                   │ 合约级聚合   │
                                                   │calc_contract │
                                                   └─────────────┘
                                                           │
                                                           ▼
                                                   ┌─────────────┐
                                                   │ 输出parquet  │
                                                   │YYYY-MM-DD    │
                                                   └─────────────┘
```

#### 3.1.3 核心函数说明

##### ① `preprocess_raw_data(df_raw)` - 数据预处理
```python
功能: 清洗原始数据，过滤无效记录

处理逻辑:
1. 排除电压 = 0 或电流 = 0 的记录 (无效采样)
2. 排除缺失关键字段 (用户id, 合约id, 时间戳) 的记录
3. 时间戳格式标准化 (转为datetime类型)
4. 按合约ID + 时间戳排序

输入:  DataFrame (原始数据)
输出: DataFrame (清洗后数据)
```

##### ② `batch_gps_to_region(df, lat_col, lon_col)` - GPS空间匹配
```python
功能: 将经纬度坐标转换为行政区划名称

算法流程:
1. 计算每个合约的几何中心点:
   center_lat = mean(所有记录的纬度)
   center_lon = mean(所有记录的经度)

2. 使用 GeoJSON 围栏数据进行 Point-in-Polygon 匹配:
   - 加载: 中国_省.geojson / 中国_市.geojson / 中国_县.geojson
   - 判断: shapely.Point(lon, lat).within(polygon)
   
3. 三级匹配顺序: 省 → 市 → 县 (逐级细化)

输出字段:
- 核心活动省份 (如: 广东省)
- 核心活动城市 (如: 深圳市)
- 核心活动区县 (如: 南山区)
- 中心经度, 中心纬度
```

##### ③ ⭐ SOC差法计算用电量 (核心算法，v1.8优化)
```python
应用位置: calc_contract_metrics() 函数内部

数学原理:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于电池SOC变化量计算实际消耗的电量

公式:
  E_kWh = Σ 标准电压_V × (容量_最早 / SOC_最早) / 1000 × (SOC_最早 - SOC_最晚) / 100

其中:
  标准电压_V: 从电池单体电压.csv查表获取 (未匹配则默认60V)
  容量_最早:  最早时间戳记录的容量字段
  SOC_最早:   最早时间戳记录的SOC百分比
  SOC_最晚:   最晚时间戳记录的SOC百分比

物理意义:
  满容量(Ah) = 最早容量 / 最早SOC
  满容量度数(kWh) = 标准电压 × 满容量 / 1000
  实际用电比例 = (最早SOC - 最晚SOC) / 100
  该段耗电量 = 满容量度数 × 实际用电比例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

代码实现要点:

def calculate_energy_soc_diff(df, battery_voltage_map):
    DEFAULT_VOLTAGE = 60.0
    total_energy_kwh = 0.0
    
    for bid, bgroup in df.groupby('电池id'):
        bgroup = bgroup.sort_values('时间戳')
        if len(bgroup) < 2:
            continue
        
        # 查表获取标准电压
        std_voltage = battery_voltage_map.get(bid, DEFAULT_VOLTAGE)
        
        # 取最早和最晚记录
        first_row = bgroup.iloc[0]
        last_row = bgroup.iloc[-1]
        
        first_cap = first_row['容量']
        first_soc = first_row['SOC']
        last_soc = last_row['SOC']
        
        if first_soc <= 0:
            continue
        
        # 计算满容量和耗电量
        full_capacity_ah = first_cap / first_soc
        soc_diff = first_soc - last_soc
        
        if soc_diff > 0:
            total_energy_kwh += std_voltage * full_capacity_ah * soc_diff / 100 / 1000
    
    return total_energy_kwh

计算步骤:
1. 按合约id分组
2. 按时间戳排序
3. 对每段电池id分组：
   a. 取最早时间戳的容量/SOC → 满容量(Ah)
   b. 匹配标准电压 → 满容量度数(kWh)
   c. 取最早和最晚SOC → 用电比例
   d. 计算该段耗电量
4. 所有电池段耗电量累加 → 当日总耗电量

优势对比:
┌──────────┬──────────────────────┬────────┬────────┐
│ 方法     │ 公式                 │ 精度   │ 稳定性 │
├──────────┼──────────────────────┼────────┼────────┤
│ 梯形法   │ (P_i+P_{i+1})/2×Δt  │ 中     │ 受噪声影响 │
│ SOC差法★ │ V×C/SOC×ΔSOC/100000 │ 高 ✓   │ 不受电流噪声影响 │
└──────────┴──────────────────────┴────────┴────────┘
```

##### ④ `calc_contract_metrics(df_clean, battery_voltage_map)` - 合约级指标聚合
```python
功能: 按合约ID分组，计算日级指标

新增参数 (v1.8):
- battery_voltage_map: dict, 电池id→标准电压(V)映射表

聚合维度:

【电量消耗指标】
- 日总用电量(kWh):     SOC差法计算结果 (核心输出, v1.8优化)
- 平均功率(kW):        日均功率 = 总用电 / 用电时长
- 最大功率(kW):        峰值功率
- 用电时长(h):         有效用电时间累计

【骑行相关指标】
- 骑行距离(km):        基于速度×时间积分估算
- 百公里耗电(kWh):     能效指标 = 用电量/(距离/100)
- 平均速度(km/h):      日均速度
- 最大速度(km/h):      峰值速度

【电池健康指标】
- 平均电压(V):        日均电压
- 最小电压(V):        最低电压 (反映负载情况)
- 平均电流(A):        日均电流
- 最大电流(A):        峰值电流 (风险识别关键)
- 平均SOC(%):         日均电量状态
- 最小SOC(%):         最低电量 (深度放电风险)
- SOC低于10%时长占比: 过度放电指标
- SOC低于20%时长占比: 低电量时长比例

【GPS空间指标】
- 中心经度/纬度:      活动区域中心
- 核心活动省市县:      行政区划匹配结果
- 活动半径(km):       凸包半径 (活动范围)

【保留原始字段】
- 统计日期, 用户id, 合约id, 电池id, 代理id
```

#### 3.1.4 数据输出

**输出格式:** Parquet (Snappy压缩)  
**输出路径:** `{BASE_EXPORT_PATH}/data/fact/daily/YYYY-MM-DD.parquet`  
**命名规则:** 按日期分区，每个日期一个文件

**示例:** `2026-04-24.parquet` 包含当日所有合约的日级指标

---

### 3.2 L2: Snapshot Layer (用户日级快照层)

**📁 文件位置:** [src/snapshot_layer.py](src/snapshot_layer.py)  
**⚙️ 入口函数:** `process_snapshot_layer(target_date)`  
**🎯 职责:** 将多日的Fact数据聚合成用户维度的7天滚动窗口快照

#### 3.2.1 数据输入

```
输入: data/fact/daily/*.parquet (所有日期的L1输出)
      加载全部历史日级数据进行聚合
```

#### 3.2.2 处理流程

```
┌─────────────────┐
│ 加载所有日级数据 │
│ fact/daily/*.pq │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 按用户+日期排序  │
│ 确保时间连续性   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 逐用户滚动窗口   │ ──→ │ 7天指标聚合     │
│ 取最近7天数据    │     │ sum/mean/max    │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
                         │ 客户类型判定     │
                         │ determine_       │
                         │ customer_type    │
                         └────────┬────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │ 输出user_daily   │
                          │ .parquet         │
                          └─────────────────┘
```

#### 3.2.3 7天滚动窗口计算逻辑

```python
对每个用户，取最近7天 (含当天) 的所有合约数据进行聚合:

【聚合规则】

数值型累加指标 (sum):
  近7d总用电量_kWh          = Σ 每日用电量
  近7d总骑行距离_km         = Σ 每日骑行距离
  近7d总换电次数             = count(distinct 每日换电)

数值型平均指标 (mean):
  近7d平均骑行电流_A         = mean(每日平均电流)
  近7d平均骑行速度_km_h      = mean(每日平均速度)
  近7d日均用电量_kWh         = mean(每日用电量)
  近7d单合约日均骑行时长_h   = mean(每日骑行时长)

极值型指标 (max):
  近7d最大电流_A             = max(每日最大电流)
  近7d最大速度_km_h          = max(每日最大速度)
  近7d最大功率_kW            = max(每日最大功率)

占比型指标 (mean of ratios):
  近7d_SOC低于10%时长占比    = mean(每日SOC<10%占比)
  近7d_SOC低于20%时长占比    = mean(每日SOC<20%占比)

计数指标:
  近7d有效记录天数           = count(有数据的日期数)
  近7d活跃合约数             = count(distinct 合约id)
```

#### 3.2.4 客户类型判定

```python
函数: determine_customer_type()

判定依据 (基于7天累积数据):

参数:
- total_riding_hours:     总骑行时长
- total_distance:         总骑行距离
- idle_discharge_hours:   空闲放电时长
- total_discharge_hours:  总放电时长

客户类型分类:
┌──────────┬──────────────────────────────────────────────┐
│ 类型     │ 判定条件                                     │
├──────────┼──────────────────────────────────────────────┤
│ 高频用户 │ 骑行频繁，使用强度高                           │
│ 中频用户 │ 正常使用频率                                  │
│ 低频用户 │ 使用较少                                     │
│ 极低频   │ 几乎不使用                                   │
│ 纯充电   │ 只充电不骑行 (idle_ratio接近100%)            │
└──────────┴──────────────────────────────────────────────┘
```

#### 3.2.5 数据输出

**输出格式:** Parquet  
**输出路径:** `{BASE_EXPORT_PATH}/data/snapshot/user_daily.parquet`  
**内容:** 全量用户的7天滚动快照 (每个用户一条记录)

---

### 3.3 L3: Lifecycle Layer (用户生命周期层)

**📁 文件位置:** [src/lifecycle_layer.py](src/lifecycle_layer.py)  
**⚙️ 入口函数:** `process_lifecycle_layer(target_date)`  
**🎯 职责:** 用户评分分类 + 风险标签生成 + 完整画像文本 + 合约信息整合

#### 3.3.1 数据输入

```
输入: snapshot/user_daily.parquet (L2输出的用户7天快照)
辅助: 4-5用户最早合约时间.csv (首次入网日期、合约到期时间)
配置: thresholds_baseline.json (动态阈值基准线)
```

#### 3.3.2 处理流程

```
┌─────────────────┐
│ 加载Snapshot数据 │
│ user_daily.parquet│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 加载合约信息     │ ──→ │ 整合首次入网日期 │
│ 4-5用户最早      │     │ 整结合约到期时间 │
│ 合约时间.csv     │     │ 计算用户年龄     │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
                         │ 动态阈值计算     │
                         │ DynamicBattery   │
                         │ Analyzer         │
                         └────────┬────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │ 用户等级判定     │
                          │ 风险标签生成     │
                          │ 策略建议匹配     │
                          └────────┬────────┘
                                   │
                                   ▼
                           ┌─────────────────┐
                           │ 月度用电量预估   │
                           │ 出勤率方法       │
                           │ 完全体画像文本   │
                           └────────┬────────┘
                                    │
                                    ▼
                            ┌─────────────────┐
                            │ 输出user_7d     │
                            │ .parquet         │
                            └─────────────────┘
```

#### 3.3.3 核心功能模块

##### ① 合约信息加载与整合
```python
函数: load_user_contract_info()

功能: 从"4-5用户最早合约时间.csv"加载用户的首次入网日期和合约到期时间

输出字段:
- 首次入网日期: YYYY-MM-DD 格式
- 合约到期时间: YYYY-MM-DD 格式
- 用户年龄(天): 从入网到今天的天数

数据来源: config.EXPORT_FILE_CONTRACT_EARLY
缓存机制: 全局变量 _user_contract_map，避免重复加载
```

##### ② 动态阈值计算 (DynamicBatteryAnalyzer)
```python
初始化参数:
baseline_path = 'src/thresholds_baseline.json'
use_ema_update = True    # ✅ 已启用EMA自动更新
ema_alpha = 0.3          # EMA平滑系数

功能:
1. 基于7天滚动窗口数据计算实时分位数
2. 与历史基准线对比检测分布漂移
3. 使用EMA平滑更新基准线 (当检测到漂移时)
4. 输出自适应阈值用于用户分级

详细说明见第4章《动态阈值系统》
```

##### ③ 用户等级分类体系
```python
等级列表 (按风险从低到高):
┌──────────┬───────┬────────────────────────────────────┐
│ 等级     │ 排名  │ 特征描述                            │
├──────────┼───────┼────────────────────────────────────┤
│ 优质用户 │  1    │ 用电规律、低损耗、高能效             │
│ 良好用户 │  2    │ 使用正常、偶有小异常                │
│ 观察期   │  3    │ 新用户或数据不足 (<7天)             │
│ 普通用户 │  3    │ 有一定异常但不严重                   │
│ 高损耗用户│  4   │ 用电量偏高、效率偏低                │
│ 沉默用户 │  5    │ 长期未使用 (>7天无数据)             │
│ 暴力     │  6    │ 严重超流、改装嫌疑、高风险           │
└──────────┴───────┴────────────────────────────────────┘

对应运营策略:
STRATEGY_MAP = {
    '优质用户': '留存激励',
    '良好用户': '维持服务',
    '普通用户': '引导升级',
    '高损耗用户': '限制预警',
    '暴力': '清退处理',
    '观察期': '新手引导',
    '沉默用户': '激活唤醒',
}
```

##### ④ ⭐ 月度用电量预估 (出勤率方法)
```python
函数: _calc_monthly_estimate()

算法原理:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
根据实际记录天数和出勤情况，分三档预估月度用电量:

【第一档】记录天数 < 7 天
  → 使用默认26个工作日估算
  → 月度预估 = 日均用电 × 26

【第二档】7 ≤ 记录天数 < 30 天
  → 基于实际出勤率推算
  → 出勤率 = 有数据天数 / 总观察天数
  → 月度预估 = 日均用电 × 出勤率 × 30

【第三档】记录天数 ≥ 30 天
  → 基于实际累积数据计算
  → 月度预估 = 累积总用电 / 月数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

优势:
✅ 更符合实际使用场景
✅ 避免新用户过度外推
✅ 考虑了用户出勤习惯的差异
```

##### ⑤ 完全体用户画像文本生成
```python
函数: generate_comprehensive_profile()

输出字段: 用户完全体画像 (长文本字符串)

内容结构:
[基本信息]
  用户ID: xxx
  用户等级: 优质用户
  客户类型: 高频用户
  注册天数: xxx天

[用电特征]
  近7天总用电: xx.x kWh
  日均用电: x.xx kWh
  百公里耗电: x.x kWh
  月度预估: xx.x kWh (基于出勤率xx%)

[骑行习惯]
  近7天骑行: xxx km
  平均速度: xx km/h
  活动范围: 广东省深圳市南山区
  出行时段: 习惯晚间高峰 (17-19点)

[电池健康]
  平均电压: xx.x V
  SOC均值: xx%
  低电量(<20%)时长占比: xx%

[风险评估]
  风险标签: 无明显风险
  运营建议: 留存激励
```

#### 3.3.4 数据输出

**输出格式:** Parquet  
**输出路径:** `{BASE_EXPORT_PATH}/data/lifecycle/user_7d.parquet`  
**内容:** 完整用户画像 (97+ 字段)，每个用户一条记录

---

### 3.4 L4: Report Layer (报告输出层)

**📁 文件位置:** [src/report_layer.py](src/report_layer.py)  
**⚙️ 入口函数:** `process_report_layer(target_date)`  
**🎯 职责:** 将Lifecycle层数据导出为多种业务友好的报告格式

#### 3.4.1 输出报告清单

| 报告文件名 | 格式 | 内容说明 | 目标用户 |
|------------|------|----------|----------|
| user_7d_full.csv | CSV | 完整用户画像 (97+字段) | 数据分析师/BI |
| 用户月度出勤预估明细.csv | CSV | 月度用电量预估明细 | 运营人员 |
| 用户完全体画像.txt | TXT | 结构化画像文本 | 客服/运营 |
| lifecycle_summary.json | JSON | 统计摘要 (各等级人数) | 系统集成 |

#### 3.4.2 数据输出

**输出路径:** `{BASE_EXPORT_PATH}/data/reports/`  
**编码格式:** UTF-8 with BOM (Excel兼容)

---

## 4. 动态阈值系统 (DynamicThresholds)

**📁 文件位置:** [src/dynamic_thresholds.py](src/dynamic_thresholds.py)  
**⚙️ 核心类:** `DynamicBatteryAnalyzer`

### 4.1 系统目标

解决**静态阈值无法适应业务变化**的问题：
- 新用户涌入导致分布偏移
- 季节因素影响用电模式
- 业务策略调整改变用户行为

### 4.2 工作流程

```
┌─────────────────┐
│  输入: 7天滚动   │
│  用户数据DataFrame│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Step 1: 计算     │ ──→ │ 实时分位数      │
│ 实时统计数据     │     │ Q1/Q2/Q3/IQR    │
└────────┬────────┘     └────────┬────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐     ┌─────────────────┐
│ Step 2: 加载     │ ──→ │ 历史基准线      │
│ 历史基准线       │     │ (上次计算的)    │
└────────┬────────┘     └────────┬────────┘
         │                      │
         ▼                      ▼
         └──────────┬───────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Step 3: 对比   │
            │ 检测分布漂移   │
            │ (KS检验)       │            │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ Step 4: 生成   │
            │ 自适应阈值     │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐     ┌───────────────┐
            │ Step 5: EMA    │ ──→ │ 更新基准文件   │
            │ 平滑更新       │     │ (如果检测到漂移)│
            └───────────────┘     └───────────────┘
```

### 4.3 EMA 自动更新机制 ⭐ (v1.5新增)

```python
启用条件:
  use_ema_update = True  # 在 lifecycle_layer.py 初始化时设置
  ema_alpha = 0.3        # 平滑系数 (0-1之间，越大越敏感)

触发时机:
  当检测到分布漂移 (has_shift=True) 时自动触发

更新公式:
  new_baseline = ema_alpha * live_quantiles + (1 - ema_alpha) * old_baseline

物理意义:
  - alpha=0.3: 新数据权重30%，历史经验权重70%
  - 既响应变化，又避免剧烈波动
  - 类似于金融领域的指数移动平均线

输出:
  - 更新文件: src/thresholds_baseline.json
  - 日志标记: [EMA] ✅ 基准线文件已更新

调试日志:
  [DynamicThresholds] [DEBUG] EMA配置状态 - use_ema: True
  [DynamicThresholds] [DEBUG] 分布漂移检测结果 - has_shift: True/False
  [DynamicThresholds] [EMA] ✅ 检测到分布漂移，执行EMA平滑更新...
  [DynamicThresholds] [EMA] ✅ 基准线文件已更新
```

### 4.4 阈值输出示例

```json
{
  "近7d总用电量_kWh": {"Q1": 2.5, "Q2": 5.0, "Q3": 8.0, "IQR": 5.5},
  "近7d平均骑行电流_A": {"Q1": 5.0, "Q2": 12.0, "Q3": 20.0, "IQR": 15.0},
  ...
}
```

---

## 5. Pipeline 执行模式

**📁 文件位置:** [src/run_pipeline.py](src/run_pipeline.py)  
**⚙️ 入口函数:** `main()` (命令行调用)

### 5.1 模式总览

| 模式 | 清除历史 | L1 Fact | L2 Snapshot | L3 Lifecycle | L4 Report | 适用场景 |
|------|----------|---------|-------------|--------------|-----------|----------|
| **full** | ✅ 全部清除 | ✅ 全量 | ✅ 全量 | ✅ 全量 | ✅ 全量 | 首次运行/重大变更后 |
| **incremental** | ❌ 保留 | ✅ 仅新日期 | ✅ 增量 | ✅ 增量 | ✅ 增量 | 日常定时任务 |
| **refresh** | ❌ 保留 | ❌ 跳过 | ✅ 重算 | ✅ 重算 | ✅ 重算 | 调整聚合逻辑后 |
| **recalculate** | ❌ 保留 | ❌ 跳过 | ❌ 跳过 | ✅ 重算 | ✅ 重算 | 调整评级规则后 |
| **report_only** | ❌ 保留 | ❌ 跳过 | ❌ 跳过 | ❌ 跳过 | ✅ 重算 | 仅需更新报告 |

### 5.2 详细说明

#### ① Full 模式 (全量重建)
```bash
命令: python run_pipeline.py --mode full

特点:
- 清除 data/fact/, data/snapshot/, data/lifecycle/, data/reports/ 所有历史数据
- 处理数据源中所有可用日期的原始文件
- 适用于: 首次运行、算法大改后、数据质量问题修复后

输出示例:
🚀 执行模式: FULL (全量重建)
   🗑️  已清除 fact/ 目录
   🗑️  已清除 snapshot/ 目录
   🗑️  已清除 lifecycle/ 目录
   🗑️  已清除 reports/ 目录
   ✅ 已清除 4 个历史数据目录，开始全量重算
```

#### ② Incremental 模式 (增量处理) ⭐ (v1.5优化)
```bash
命令: python run_pipeline.py --mode incremental
默认模式: 是 (不指定 --mode 时默认执行此模式)

特点:
- 保留所有历史数据
- 自动扫描数据源目录，检测缺失的日期
- 仅处理尚未生成 fact 的日期
- 自动排除当天数据 (避免处理中的不完整文件)
- 忽略 --date 参数 (自动检测模式)

智能检测逻辑:
  已存在文件: data/fact/daily/2026-04-24.parquet → 跳过
  已存在文件: data/fact/daily/2026-04-25.parquet → 跳过
  缺失日期:   2026-04-26 → ✅ 处理
  当天日期:   2026-04-27 → ⏭️ 跳过 (数据可能不完整)

适用场景:
- 每日定时任务 (cron/scheduler)
- 补录历史数据后的增量处理
- 日常运维自动化

输出示例:
🚀 执行模式: INCREMENTAL (增量处理)
   ⚠️ 注意: incremental模式将忽略--date参数，自动检测所有缺失日期
   📊 检测到 1 个缺失日期需要处理: ['2026-04-26']
   ⏭️ 跳过当天日期: 2026-04-27 (数据可能不完整)
```

#### ③ Refresh 模式 (刷新聚合)
```bash
命令: python run_pipeline.py --mode refresh [--date 2026-04-26]

特点:
- 保留 L1 (fact) 数据不变
- 重新执行 L2-L4 (聚合、评分、报告)
- 适用于: 修改了snapshot_layer聚合逻辑后

使用场景:
- 调整了7天滚动窗口的计算方式
- 修改了客户类型判定规则
- 变更了月度预计算因子
```

#### ④ Recalculate 模式 (重新评级)
```bash
命令: python run_pipeline.py --mode recalculate [--date 2026-04-26]

特点:
- 保留 L1-L2 数据不变
- 重新执行 L3-L4 (评分、分类、报告)
- 适用于: 修改了lifecycle_layer评级规则后

使用场景:
- 调整了用户等级划分阈值
- 修改了风险标签生成逻辑
- 变更了策略建议映射表
- 更新了月度用电量预估算法
```

#### ⑤ Report Only 模式 (仅报告)
```bash
命令: python run_pipeline.py --mode report_only [--date 2026-04-26]

特点:
- 保留 L1-L3 数据不变
- 仅重新执行 L4 (报告生成)
- 适用于: 仅需调整报告格式或补充输出

使用场景:
- 新增了报告输出字段
- 修改了CSV/JSON导出格式
- 需要重新生成可视化数据
```

### 5.3 命令行参数

```bash
用法: python run_pipeline.py [选项]

选项:
  --mode {full,incremental,refresh,recalculate,report_only}
                        执行模式 (默认: incremental)
  --date DATE           目标日期 YYYY-MM-DD (默认: 今天)
                        注: incremental模式会忽略此参数
  --layer {L1,L2,L3,L4} 从指定层开始重建，清除该层及下游数据后重算
                        L1=全量, L2=跳过Fact, L3=跳过Fact+Snapshot, L4=仅报告

示例:
  # 日常增量处理 (最常用)
  python run_pipeline.py
  
  # 全量重建
  python run_pipeline.py --mode full
  
  # 仅重跑评级和报告
  python run_pipeline.py --mode recalculate
  
  # 处理指定日期 (非incremental模式)
  python run_pipeline.py --mode refresh --date 2026-04-26
  
  # 修改Fact层逻辑后，从L1开始重建
  python run_pipeline.py --layer L1
  
  # 修改Snapshot层逻辑后，从L2开始重建
  python run_pipeline.py --layer L2
  
  # 修改评分/画像逻辑后，从L3开始重建
  python run_pipeline.py --layer L3
  
  # 仅修改报告模板，从L4开始重建
  python run_pipeline.py --layer L4
```

### 5.4 --layer 参数说明 ⭐ (v2.2新增)

当修改了某一层的核心逻辑时，不需要每次都从 L1 全量重算。`--layer` 参数允许你指定从哪一层开始重建，自动清除该层及下游数据：

| 参数值 | 清除范围 | 保留范围 | 适用场景 |
|--------|----------|----------|----------|
| `--layer L1` | fact/ snapshot/ lifecycle/ reports/ | 无 | 修改Fact层指标计算逻辑 |
| `--layer L2` | snapshot/ lifecycle/ reports/ | fact/ | 修改Snapshot层聚合逻辑 |
| `--layer L3` | lifecycle/ reports/ | fact/ snapshot/ | 修改评分/分类/画像逻辑 |
| `--layer L4` | reports/ | fact/ snapshot/ lifecycle/ | 仅修改报告模板 |

**效率对比：**
```
修改评分规则后:
  旧方式: python run_pipeline.py --mode full    ← 需重算全部4层，耗时数小时
  新方式: python run_pipeline.py --layer L3     ← 仅重算L3+L4，耗时几分钟
```

---

## 6. 关键算法说明

### 6.1 梯形数值积分法 (用电量计算)

**应用层级:** L1 Fact Layer  
**应用位置:** [fact_layer.py](src/fact_layer.py) - `calculate_energy_trapezoidal()` 函数

详见 3.1.3 节第③部分。

### 6.2 出勤率法 (月度用电量预估)

**应用层级:** L3 Lifecycle Layer  
**应用位置:** [lifecycle_layer.py](src/lifecycle_layer.py) - `_calc_monthly_estimate()` 函数

详见 3.3.3 节第④部分。

### 6.3 GPS空间匹配算法

**应用层级:** L1 Fact Layer  
**应用位置:** [utils/geo.py](utils/geo.py) - `batch_gps_to_region()` 函数

```python
算法步骤:
1. 计算合约所有采样点的几何中心 (mean lon, mean lat)
2. 创建 shapely.Point(center_lon, center_lat)
3. 加载 GeoJSON 行政区划边界数据 (省/市/县三级)
4. 依次判断 Point.within(Polygon):
   - 先匹配省份 (中国_省.geojson)
   - 再匹配城市 (中国_市.geojson)
   - 最后匹配区县 (中国_县.geojson)
5. 返回三级行政区划名称

性能优化:
- GeoJSON文件只加载一次 (全局缓存)
- 使用空间索引加速 (rtree/shapely prepared geometry)
- 批量处理减少IO开销
```

### 6.4 凸包算法 (活动半径计算)

**应用层级:** L1 Fact Layer  
**应用位置:** [fact_layer.py](src/fact_layer.py) - ConvexHull计算

```python
from scipy.spatial import ConvexHull

算法步骤:
1. 提取合约所有有效GPS点 [(lon1,lat1), (lon2,lat2), ...]
2. 构建ConvexHull对象
3. 计算凸包面积 (hull.area)
4. 转换为等效圆半径: radius = sqrt(area / π)
5. 输出: 活动半径(km)

物理意义:
- 反映用户日常活动的地理范围
- 半径越小 → 活动越集中 (如固定路线配送)
- 半径越大 → 活动越分散 (如随机出行)
```

### 6.5 EMA平滑算法 (基准线更新)

**应用层级:** L3 Lifecycle Layer → DynamicThresholds  
**应用位置:** [dynamic_thresholds.py](src/dynamic_thresholds.py) - `_ema_update_baseline()` 函数

详见 4.3 节。

### 6.6 时段划分与出行习惯识别

**应用层级:** L1 Fact Layer & L2 Snapshot Layer

```python
时段定义 (严格按照旧脚本):
午间高峰: 11:00-13:00  (NOON_PEAK_HOURS)
晚间高峰: 17:00-19:00  (EVENING_PEAK_HOURS)
夜间时段: 20:00-05:00   (NIGHT_HOURS)
平峰时段: 其余时间     (OFFPEAK_HOURS)

判定逻辑:
1. 在L1层按时段统计每次骑行的里程、时长、次数
2. 在L2层聚合最近7天的时段数据
3. 在L3层找出主导时段 (占比≥50%则标记为该习惯)

输出示例:
  出行时段: "习惯晚间高峰" 或 "全天" 或 "数据不足"
```

---

## 7. 项目文件结构

```
User_characteristics_repo/
├── config/
│   ├── __init__.py
│   └── config.py              # 路径配置 (数据源/输出/辅助数据)
│
├── src/
│   ├── __init__.py
│   ├── run_pipeline.py         # ⭐ 流水线主入口 (5种执行模式)
│   ├── fact_layer.py           # L1 日级事实计算 (梯形积分法)
│   ├── snapshot_layer.py       # L2 7天滚动快照聚合
│   ├── lifecycle_layer.py      # L3 评分分类+画像+合约信息
│   ├── report_layer.py         # L4 多格式报告导出
│   ├── score_layer.py          # 评分逻辑工具库 (被L3调用)
│   ├── dynamic_thresholds.py   # 动态阈值系统 (EMA自动更新)
│   └── thresholds_baseline.json # 阈值基准线文件 (自动更新)
│
├── utils/
│   ├── __init__.py
│   └── geo.py                  # GPS空间匹配工具 (省市县三级)
│
├── temp/                       # 已废弃的旧脚本 (已移至此处)
│   ├── ARCHITECTURE.md
│   ├── config_root.py
│   ├── raw_ingest.py
│   ├── score_layer_优化建议.md
│   ├── verify_fixes.py
│   ├── 改进.txt
│   ├── 用户完全体画像_新增方案.md
│   ├── 用户月度出勤预估明细.csv
│   ├── 用户生命周期管理 copy.py
│   └── 用户用电情况_独立版.py
│
├── PROJECT_LOGIC.md            # 📄 本技术文档
├── README.md                   # 项目说明文档
│
└── data/                       # (运行后生成)
    ├── fact/daily/             # L1输出: 按日期分区的parquet文件
    ├── snapshot/               # L2输出: user_daily.parquet
    ├── lifecycle/              # L3输出: user_7d.parquet
    └── reports/                # L4输出: CSV/TXT/JSON报告
```

---

## 8. 快速开始

### 8.1 环境要求

```bash
Python >= 3.13
操作系统: Windows 10/11 (推荐)

依赖库:
pandas>=2.0
numpy>=1.24
pyarrow>=12.0      # Parquet文件支持
shapely>=2.0       # GIS空间计算
scipy>=1.10        # 凸包算法
tqdm>=4.65         # 进度条显示
```

### 8.2 安装依赖

```bash
cd d:\PY代码\用户特征画像\User_characteristics_repo
pip install -r requirements.txt
```

### 8.3 配置数据源

编辑 [config/config.py](config/config.py):

```python
# 设置测试/正式环境
TEST_MODE = True  # 测试环境使用 E:\test
TEST_MODE = False # 正式环境使用 E:\OneDrive\Powerbi

# 确保以下数据源路径正确:
BATTERY_STATUS_FOLDER = "基础数据/用户电池情况"  # Parquet文件目录
CONTRACT_FOLDER = "合约信息"                       # 合约信息CSV
GEOJSON_FOLDER = "基础数据/省市区围栏"             # GeoJSON围栏文件
```

### 8.4 运行Pipeline

```bash
cd src

# 首次运行 (全量重建)
python run_pipeline.py --mode full

# 日常增量处理 (推荐)
python run_pipeline.py

# 仅重新生成报告
python run_pipeline.py --mode report_only

# 调整评级规则后重跑
python run_pipeline.py --mode recalculate
```

### 8.5 验证输出

检查输出文件是否存在且数据完整:

```bash
# L1输出 (应有多个日期文件)
dir ..\data\fact\daily\*.parquet

# L2输出 (单个文件)
dir ..\data\snapshot\user_daily.parquet

# L3输出 (核心结果)
dir ..\data\lifecycle\user_7d.parquet

# L4输出 (多个报告)
dir ..\data\reports\
```

---

## 9. 常见问题排查

### 9.1 数据问题

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 百公里耗电全部为零 | 骑行距离未计算或为零 | 检查速度字段是否有效 (>0) |
| 用电量异常偏大 | 存在电流尖峰 ( >100A ) | 检查数据质量，确认是否真实数据 |
| GPS匹配失败 | 经纬度为空或超出范围 | 检查原始数据GPS字段完整性 |
| 某些用户缺少合约信息 | 未在"4-5用户最早合约时间.csv"中 | 补充用户合约数据源 |

### 9.2 性能问题

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| L1处理非常慢 | 单日数据量过大 (>10万行) | 检查是否有异常大量数据 |
| 内存不足 | 一次性加载过多日期 | 使用 incremental 模式分批处理 |
| Snapshot聚合慢 | 历史日期过多 | 定期归档旧数据 |

### 9.3 阈值系统问题

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 基准线文件未更新 | EMA未启用或未检测到漂移 | 检查 lifecycle_layer.py 中 use_ema_update=True |
| 用户等级全部相同 | 阈值计算异常 | 检查 thresholds_baseline.json 是否有效 |
| 分布漂移误报 | 数据波动较大 | 调整 ema_alpha 降低敏感度 |

### 9.4 Pipeline模式问题

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| Incremental模式处理了当天数据 | 代码bug (已修复于v1.5) | 升级到最新代码 |
| Full模式未清除历史 | 目录权限问题 | 手动删除 data/ 下子文件夹 |
| Refresh模式报错 | 缺少L1数据 | 先运行 full 或 incremental 模式 |

### 9.5 调试技巧

```bash
# 1. 开启详细日志 (在代码中添加)
import logging
logging.basicConfig(level=logging.DEBUG)

# 2. 单独测试某一层
python -c "from fact_layer import process_fact_layer; process_fact_layer()"

# 3. 检查中间数据
import pandas as pd
df = pd.read_parquet('data/fact/daily/2026-04-26.parquet')
print(df.head())
print(df.columns.tolist())

# 4. 验证阈值系统
from dynamic_thresholds import DynamicBatteryAnalyzer
analyzer = DynamicBatteryAnalyzer(baseline_path='src/thresholds_baseline.json')
print(analyzer.store.load())  # 查看当前基准线
```

---

## 附录

### A. 字段血缘关系 (Field Lineage)

```
原始字段 → L1 Fact → L2 Snapshot → L3 Lifecycle → L4 Report

电压(mV), 电流(A), 时间戳
    ↓ (SOC差法)
日总用电量(kWh), 平均功率(kW), 最大功率(kW)
    ↓ (7天聚合 sum/mean)
近7d总用电量_kWh, 近7d日均用电量_kWh, 近7d最大功率_kW
    ↓ (动态阈值评分)
用电量评分, 月度用电量预估_kWh
    ↓ (报告导出)
user_7d_full.csv, 用户完全体画像.txt
```

### B. 版本更新日志

**v2.2 (2026-05-11)**
- ✅ Pipeline优化: 新增 `--layer` 参数，支持从指定层开始重建（L1/L2/L3/L4）
- ✅ OneDrive兼容: `_clear_directory()` 处理文件锁问题，逐文件删除被锁定的文件
- ✅ 文档统一: README.md 与 PROJECT_LOGIC.md 版本号同步

**v2.1 (2026-05-10)**
- ✅ BUG修复: `run_pipeline.py` --mode 默认值从 `full` 修正为 `incremental`，与 help 文本一致
- ✅ BUG修复: `fact_layer.py` `process_fact_layer()` 两处早期 `return {}` 改为 `return {}, set()`，避免调用方解包失败
- ✅ BUG修复: `lifecycle_layer.py` 7天滚动 MIN/MAX 被 -1 污染，导致骑行时间行大面积缺失
- ✅ 代码重构: `dynamic_thresholds.py` `score_and_classify()` 方法委托给 `score_layer.score_and_classify()`，消除 ~400 行重复代码
- ✅ 维护性提升: 评分逻辑统一由 `score_layer.py` 维护，避免两处逻辑不一致的风险
- ✅ 骑行时刻精度: `最早骑行时刻_h` 从整数小时改为带分钟的小数（如 7.38），画像显示 `HH:MM` 格式
- ✅ 出勤判定优化: 骑行>=0.5h 或 放电>=0.5h 即算出勤，覆盖地摊/储能等无骑行但有放电的场景
- ✅ 新增 `--layer` 参数: 支持从指定层开始重建（`--layer L1/L2/L3/L4`），避免每次 full 重算
- ✅ OneDrive 兼容: `_clear_directory()` 处理文件锁问题，逐文件删除被锁定的文件
- ✅ 移除 `base_path` 参数: 所有层函数不再接受 `base_path` 参数，统一使用 `config.py` 中的路径配置
- ✅ 增量模式优化: L2 Snapshot 层支持仅处理新增日期，避免全量重算

**v2.0 (2026-05-09)**
- ✅ 用电量计算优化: 按电池段首尾SOC差值一次性计算，取消逐行累加
- ✅ 计算逻辑简化: 每段电池id只取最早和最晚时间戳记录，直接计算该段耗电量
- ✅ 公式更新: `E_kWh = Σ 标准电压 × (容量_最早/SOC_最早) / 1000 × (SOC_最早 - SOC_最晚) / 100`

**v1.7 (2026-05-08)**
- ✅ 用电量计算重构: 从梯形积分法改为SOC差法，公式 `(标准电压 × 容量/SOC / 1000) × (SOC差 / 100)`
- ✅ 电池电压映射表: 新增 `load_battery_voltage_map()` 函数，从 `电池单体电压.csv` 加载标准电压
- ✅ 换电检测: SOC上升超过1%时自动重新计算初始容量，适配换电场景
- ✅ 离线数据过滤: 预处理阶段剔除 `是否在线=0/否/False` 的记录
- ✅ 配置路径扩展: `config.py` 新增 `BATTERY_INFO_FOLDER` 和 `EXPORT_FILE_BATTERY_CELL_VOLTAGE`

**v1.6 (2026-05-07)**
- ✅ 阈值常量统一管理: 所有阈值常量集中到 `score_common.py`，消除跨文件常量冲突
- ✅ 新用户保护期修复: `lifecycle_layer.py` 注入 `入网天数` 列，使 `score_layer.py` 保护期逻辑生效
- ✅ 客户形态聚合一致性: L3层客户形态聚合改为优先级方式（改装/超速车 > 地摊/储能 > 频率最高），与L2层对齐
- ✅ 暴力用户等级说明修复: `fact_layer.py` 中等级说明匹配名称从 `"暴力用户（超量放电/电池滥用）"` 改为 `"暴力"`
- ✅ 死代码清理: 删除 `fact_layer.py` 中永不可达的"外卖高强度车"分支
- ✅ 冗余导入清理: 删除 `snapshot_layer.py` 中未使用的 `calc_monthly_score_v2`、`determine_user_level_v2` 导入
- ✅ 重复导入清理: 删除 `fact_layer.py` 中 `preprocess_raw_data` 函数内重复的 `haversine` 导入
- ✅ 硬编码修复: `fact_layer.py` 中 `current_80a_count` 从硬编码 `0` 改为真实值 `res['电流>80A次数']`
- ✅ 注释补充: `dynamic_thresholds.py` 中 `score_and_classify` 方法添加用途说明注释

**v1.5 (2026-04-27)**
- ✅ 数据源迁移: 从原始CSV改为Parquet格式，移除L0层
- ✅ 增量模式优化: 自动检测缺失日期，排除当天不完整数据
- ✅ EMA自动更新: 启用基准线自动平滑更新机制
- ✅ 合约信息整合: 新增首次入网日期、合约到期时间字段
- ✅ 月度预估改进: 采用出勤率三档估算法
- ✅ 代码清理: 移除11个废弃脚本至temp目录

**v1.4 (2026-04-27)**
- ✅ 启用EMA动态更新阈值基准线

**v1.3 (2026-04-26)**
- ✅ 月度用电量预估改用出勤率方法
- ✅ 新增用户月度出勤预估明细报告

**v1.2 (2026-04-26)**
- ✅ 用电量计算统一采用梯形数值积分法
- ✅ 移除电池度数差值法的冲突实现

**v1.1 (2026-04-25)**
- ✅ 新增动态阈值系统 (DynamicThresholds)
- ✅ 新增用户评分和等级分类
- ✅ 新增风险标签和策略建议

**v1.0 (2026-04-24)**
- ✅ 初始架构搭建 (L1-L4四层流水线)
- ✅ 基础指标计算 (用电量、骑行、电池健康)
- ✅ GPS空间匹配 (省市县三级)
- ✅ 多格式报告输出

### C. 相关文档索引

| 文档 | 说明 | 路径 |
|------|------|------|
| **PROJECT_LOGIC.md** | 本技术文档 | `/PROJECT_LOGIC.md` |
| README.md | 项目介绍与快速入门 | `/README.md` |
| config.py | 路径与环境配置 | `/config/config.py` |
| thresholds_baseline.json | 动态阈值基准线 | `/src/thresholds_baseline.json` |

---

> **文档维护说明:**  
> 本文档随项目迭代同步更新，最后更新时间为 2026-05-10。  
> 如发现文档与代码不一致，请以代码实现为准，并及时更新本文档。
