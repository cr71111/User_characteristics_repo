# ML 模型使用说明

> **最后更新:** 2026-05-17
> **统一入口:** `src/ml/run_ml.py`

---

## 1. 前置条件

ML 依赖 pipeline 产出的 `user_7d.parquet`，路径由 `config.py` 自动读取。确保已安装依赖：

```bash
pip install scikit-learn xgboost pandas numpy
```

---

## 2. 统一入口

所有 ML 操作通过 `src/ml/run_ml.py` 统一调度，**自动查找最新文件，无需手动指定路径**。

```bash
python src/ml/run_ml.py {mode}
```

### 2.1 模式一览

| 模式 | 命令 | 说明 |
|------|------|------|
| `full` | `python src/ml/run_ml.py full` | 一键：异常检测 + 生成待标注 |
| `anomaly` | `python src/ml/run_ml.py anomaly` | 孤立森林 + DBSCAN 双模型检测 |
| `label generate` | `python src/ml/run_ml.py label generate` | 从异常报告中生成待标注CSV |
| `label save` | `python src/ml/run_ml.py label save` | 保存人工标注到标注库 |
| `label stats` | `python src/ml/run_ml.py label stats` | 查看标注进度和类别分布 |
| `train` | `python src/ml/run_ml.py train` | 训练 XGBoost 多分类器 |
| `predict` | `python src/ml/run_ml.py predict` | 对 lifecycle 数据预测异常类型 |

### 2.2 常用可选参数

| 参数 | 适用模式 | 默认值 | 说明 |
|------|----------|--------|------|
| `--contamination 0.03` | anomaly / full | 0.05 | 预期异常比例，越小越保守 |
| `--n-top 100` | label generate / full | 50 | 标注 Top-N 最异常用户 |
| `--test-size 0.3` | train | 0.2 | 测试集比例 |
| `--confidence 0.5` | predict | 0.6 | 低于此值标记为"低置信度" |

---

## 3. 完整操作流程

### 第一步：异常检测 + 生成待标注

```bash
python src/ml/run_ml.py full
```

**输出文件：**

| 文件 | 路径 | 说明 |
|------|------|------|
| 异常报告 | `ml_anomaly/anomaly_report_YYYYMMDD_HHMMSS.csv` | 所有用户异常分数 |
| 交叉验证 | `ml_anomaly/anomaly_report_xxx_cross_validation.csv` | 两模型一致/分歧统计 |
| 新发现 | `ml_anomaly/anomaly_report_xxx_new_findings.csv` | 模型判异常但规则判正常的用户 |
| 待标注 | `ml_anomaly/labels/to_label_YYYYMMDD_HHMMSS.csv` | 需要人工审核的用户列表 |

**异常报告关键列：**

| 列名 | 含义 | 值举例 |
|------|------|--------|
| `ml_异常分数_iforest` | 孤立森林分数（越低越异常） | -0.48 |
| `ml_异常标签_iforest` | -1=异常, 1=正常 | -1 |
| `ml_聚类标签_dbscan` | DBSCAN聚类标签，-1=噪声点 | -1 |
| `ml_模型判异常规则判正常` | 1=模型发现的新异常 | 1 |

---

### 第二步：人工标注

打开 `to_label_xxx.csv`，在 **"人工标注"** 列填写以下 7 个值之一：

| 标注值 | 判定依据 |
|--------|----------|
| **改装/超速** | 最高速度 > 70、最大电流 > 100、速度波动比 > 2 |
| **地摊/储能** | 怠速放电占比 > 50%、日行驶 < 2km、平均速度 < 5 |
| **电池老化** | 日换电 > 10 次、最低 SOC < 15%、百公里电耗 > 4 |
| **暴力驾驶** | 超 100A 连续 > 3 次、最高温度 > 70°C、高速占比高 |
| **其他异常** | 多个特征异常但不属于以上类别 |
| **正常** | 模型误判，各特征在合理范围 |
| **不确定** | 信息不足无法判断，先跳过 |

> **标注原则：** 宁可少标也不要标错。拿不准就填"不确定"，错误标注比不标注更伤害模型效果。

---

### 第三步：保存 + 训练 + 预测

```bash
python src/ml/run_ml.py label save
python src/ml/run_ml.py train
python src/ml/run_ml.py predict
```

**训练输出：**

| 文件 | 说明 |
|------|------|
| `ml_anomaly/models/xgb_model_YYYYMMDD_HHMMSS.json` | 模型权重 |
| `ml_anomaly/models/xgb_model_xxx_meta.json` | 训练指标（准确率、交叉验证、特征重要性） |
| `ml_anomaly/models/xgb_model_xxx_encoder.pkl` | 标签编码器（数字↔类别名映射） |

**预测输出新增列：**

| 列名 | 含义 |
|------|------|
| `ml_xgb_预测类别` | 模型判定的异常类型（中文） |
| `ml_xgb_置信度` | 预测把握程度 (0~1) |
| `ml_xgb_低置信度` | 1 = 建议人工复核，回流入标注流程 |
| `ml_xgb_概率_正常` | 判定为正常的概率 |
| `ml_xgb_概率_改装/超速` | 判定为各类异常的概率分布 |

---

## 4. 输出文件目录

```
E:\...\用户行为习惯\用户特征画像\
└── ml_anomaly/
    ├── anomaly_report_20260517_143000.csv        ← 异常检测报告
    ├── anomaly_report_xxx_cross_validation.csv   ← 双模型交叉验证
    ├── anomaly_report_xxx_new_findings.csv       ← 新发现的异常用户
    ├── labels/
    │   ├── to_label_20260517_143000.csv          ← 待标注（需人工填写）
    │   ├── labeled_data.csv                      ← 累积标注库（训练用）
    │   └── label_meta.json                       ← 标注元数据
    ├── models/
    │   ├── xgb_model_20260517_150000.json        ← XGBoost 模型
    │   ├── xgb_model_xxx_meta.json               ← 训练指标
    │   └── xgb_model_xxx_encoder.pkl             ← 标签编码器
    └── predictions/
        └── prediction_20260517_151000.csv         ← 预测结果
```

---

## 5. 模型技术细节

### 5.1 特征提取（features.py）

从 lifecycle 层 97+ 字段自动提取 5 大特征组：

| 特征组 | 包含特征 | 反映 |
|--------|----------|------|
| 骑行强度 | 日均时长、里程、次数、出勤率 | 活跃程度 |
| 速度特征 | 平均速度、P50/P90速度、夜间均速 | 速度水平 |
| 电流/功率 | 平均电流、最大电流、超100A次数、百公里电耗 | 功率异常 |
| 温度/SOC | 最高温度、SOC<20%占比、最低SOC、换电次数 | 电池健康 |
| 行为模式 | 怠速放电占比、高速点数、深夜换电比、速度波动 | 行为异常 |

### 5.2 孤立森林（IsolationForest）

| 参数 | 值 | 说明 |
|------|-----|------|
| contamination | 0.05 | 预期 5% 用户异常 |
| n_estimators | 200 | 树的数量 |
| 异常判定 | 分数 < 决策阈值 | 越低越异常 |

### 5.3 DBSCAN 聚类

| 参数 | 值 | 说明 |
|------|-----|------|
| eps | 特征标准差中位数 | 邻域半径自适应 |
| min_samples | 5 | 核心点最小邻居数 |
| 噪声点 | label = -1 | 不属于任何密集簇 |

### 5.4 双模型交叉验证逻辑

```
孤立森林判异常 + DBSCAN判噪声  →  ml_双模型异常 = 1（高置信异常）
孤立森林判异常 + DBSCAN判正常  →  ml_双模型异常 = 0（待审核）
孤立森林判正常 + DBSCAN判噪声  →  ml_双模型异常 = 0（待审核）
孤立森林判正常 + DBSCAN判正常  →  ml_双模型异常 = 0（双模型一致正常）
```

### 5.5 XGBoost 分类器

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 200 | 树的数量 |
| max_depth | 6 | 树最大深度 |
| learning_rate | 0.1 | 学习率 |
| 分类数 | 6 | 5种异常 + 正常 |
| 评估指标 | 准确率 + 5折交叉验证 | StratifiedKFold |

**6 分类标签：**

| ID | 类别 |
|----|------|
| 0 | 正常 |
| 1 | 改装/超速 |
| 2 | 地摊/储能 |
| 3 | 电池老化 |
| 4 | 暴力驾驶 |
| 5 | 其他异常 |

---

## 6. 闭环迭代流程

```
     ┌──────────────────────────────────────────────┐
     │                                              │
     ▼                                              │
┌──────────┐    ┌──────────┐    ┌──────────┐       │
│ anomaly  │───→│  label   │───→│  train   │       │
│ 异常检测 │    │ 人工标注 │    │ 模型训练 │       │
└──────────┘    └──────────┘    └──────────┘       │
                                    │               │
                                    ▼               │
                              ┌──────────┐          │
                              │ predict  │          │
                              │ 自动分类 │          │
                              └──────────┘          │
                                    │               │
                              ┌─────┴─────┐         │
                              │ 低置信度？ │───Yes──┘
                              └─────┬─────┘  回标注
                                    │
                                    No
                                    ▼
                              结果直接使用
```

**核心逻辑：** 标注越多 → 模型越准 → 低置信度越少 → 需要人工标注的越少。

---

## 7. 最佳实践

| 建议 | 说明 |
|------|------|
| 标注 ≥ 50 条再训练 | 标注太少模型学不到有效规律，此时的准确率不可信 |
| 宁缺毋滥 | 拿不准就填"不确定"，错误标注的伤害远大于遗漏 |
| 定期运行 anomaly | 即是没有新标注，也能持续发现可疑用户 |
| 低置信度入队 | predict 结果中 `ml_xgb_低置信度=1` 的用户应回标，形成闭环 |
| 关注模型衰退 | 如果预测准确率持续下降，说明用户行为模式变化，需要补充新标注 |
| 类别平衡 | 尽量每种异常类型都有 ≥ 10 条标注，避免模型偏向多数类 |