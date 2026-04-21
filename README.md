# 用户特征画像项目

## 项目简介

两轮车换电用户特征画像分析系统，实现从原始数据到最终报告的完整数据流水线。

## 项目结构

```
用户特征画像/
├── config/
│   ├── __init__.py
│   └── config.py              # 项目配置文件
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py       # 数据流水线主程序
│   ├── 用户生命周期管理.py     # 用户生命周期分析模块
│   ├── dynamic_thresholds.py  # 动态阈值分析模块
│   ├── fact_layer.py          # 事实层模块
│   ├── score_layer.py         # 评分层模块
│   └── thresholds_baseline.json # 阈值基准文件
└── README.md                  # 项目文档
```

## 数据流水线

### 流程架构

```
原始数据 → 每日汇总 → 每日快照 → 生命周期档案 → 报告
  (CSV)    (Parquet)  (Parquet+CSV) (Parquet+CSV)  (Markdown)
```

### 阶段说明

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 阶段1 | 原始数据CSV | 每日汇总Parquet | 不更改任何数据，只是把每天的数据汇总起来，方便存档 |
| 阶段2 | 每日汇总Parquet | 每日快照Parquet+CSV | 从每日汇总数据中得到每日数据快照 |
| 阶段3 | 每日快照 | 生命周期档案Parquet+CSV | 调用用户生命周期管理逻辑 |
| 阶段4 | 生命周期档案 | 报告Markdown | 生成用户生命周期分析报告 |

## 使用方法

### 运行完整流水线

```bash
cd src
python data_pipeline.py
```

### 运行模式

在 `用户生命周期管理.py` 中配置运行模式：

- `"quick"` - 快速模式：跳过原始数据处理，仅利用现有快照重新生成
- `"incremental"` - 增量模式：只处理比快照表中最新日期更新的数据
- `"full"` - 完整模式：处理所有配置目录，重新计算所有数据
- `"recalculate"` - 重新计算模式：基于现有快照重新计算客户形态和用户等级

## 配置说明

在 `config/config.py` 中配置：

- `TEST_MODE` - 测试模式开关
- `BASE_EXPORT_PATH` - 基础导出路径
- 各数据文件夹路径配置

## 依赖

- pandas
- numpy
- pyarrow
- geopandas
- scipy
- tqdm

## 输出文件

### 每日汇总数据
- 路径：`{BASE_EXPORT_PATH}/用户行为习惯/用户特征画像/每日用户数据汇总/`
- 格式：Parquet（snappy压缩）
- 文件：`daily_summary_YYYY-MM-DD.parquet`

### 每日快照数据
- 路径：`{BASE_EXPORT_PATH}/用户行为习惯/用户特征画像/每日快照文件/`
- 格式：Parquet + CSV
- 文件：`daily_snapshot_YYYY-MM-DD.parquet/csv`

### 生命周期档案
- 路径：`{BASE_EXPORT_PATH}/用户行为习惯/用户特征画像/`
- 格式：Parquet + CSV
- 文件：`全量用户7天滚动生命周期档案.csv`、`user_lifecycle.parquet`

### 分析报告
- 路径：`{BASE_EXPORT_PATH}/用户行为习惯/用户特征画像/reports/`
- 格式：Markdown
- 文件：`user_lifecycle_report_YYYYMMDD_HHMMSS.md`
