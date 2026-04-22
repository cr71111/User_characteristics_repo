# Score Layer 优化建议

> 基于 fact_layer → snapshot_layer → lifecycle_layer → score_layer 全链路分析
> 问题核心：`score_and_classify()` 与日级 `_calc_monthly_score_v2()` 评分维度严重错位，6 个维度在 7d 聚合后断链。

---

## 一、问题总览

| 优先级 | 字段 / 维度 | 问题描述 | 影响 |
|--------|------------|---------|------|
| 🔴 高 | `近7d最高温度_℃` | 日级扣 0-8 分，7d 层完全丢弃 | 高温用户评分虚高 |
| 🔴 高 | `近7d最高速度_kmh` | 日级扣 0-10 分，7d 层完全丢弃 | 改装/超速用户漏判 |
| 🔴 高 | `近7d电流超60A总次数` | 60-80A 区间无风险标签 | 中高电流用户漏标 |
| 🔴 高 | `近7d超100A累计时长_h` | 日层判暴力用此字段，7d 层只看次数 | 暴力判定逻辑不一致 |
| 🔴 高 | `近7d_SOC低于10%时长占比` | 已生成但未用，只检查最低 SOC 绝对值 | 连续亏电程度判断缺失 |
| 🔴 高 | `近7d总换电次数` | 日层超 3 次扣分，7d 层完全丢弃 | 高频换电用户评分偏高 |
| 🟡 中 | `近7d平均骑行SOC` | SOC 30-80% 黄金区间奖励，7d 层无 | 优质 SOC 管理用户无正向激励 |
| 🟡 中 | `近7d高峰骑行占比` | 骑手类型已判定，策略无差异化 | 专送骑手与普通用户策略相同 |
| 🟡 中 | `近7d工作特点` | 预计算标签未接入策略建议 | 运营干预精度低 |
| 🟠 低 | `近7d出勤率` | 活跃度未参与评分加权 | 活跃优质用户与低频用户无区分 |
| 🟠 低 | `近7d单合约日均行驶里程_km` | 高里程 × 高电耗放大效应未建模 | 重度使用场景风险低估 |

---

## 二、具体优化方案

### 2.1 补充温度维度（解决最大逻辑断层）

`score_and_classify()` 的 `total` 计算中增加温度扣分，与日级评分对齐：

```python
# 在 score_and_classify() 中，读取温度字段
max_temp_7d = df['近7d最高温度_℃'].fillna(0).values

# 温度扣分（与 _calc_monthly_score_v2 保持一致）
temp_penalty = np.select(
    [max_temp_7d >= 70, max_temp_7d >= 55],
    [-8.0, -4.0],
    default=0.0
)

# 加入 total 计算
total = dim_avg_cur + dim_max_cur + dim_energy + dim_soc_low + dim_monthly + over80_score
total = total + idle_penalty + temp_penalty  # 新增
```

同步更新风险标签，增加高温告警：

```python
temp_risk = np.select(
    [max_temp_7d >= 70, max_temp_7d >= 55],
    ['电池高温告警(>70°C)', '电池温度偏高(>55°C)'],
    default=''
)

df['风险标签'] = pd.Series([
    ' / '.join(filter(None, [r, s, e, t])) if any([r, s, e, t]) else '正常'
    for r, s, e, t in zip(risk_tags, soc_risk, energy_risk, temp_risk)
], index=df.index)
```

---

### 2.2 补充速度维度

```python
max_speed_7d = df['近7d最高速度_kmh'].fillna(0).values

# 速度扣分
speed_penalty = np.select(
    [max_speed_7d >= 80, max_speed_7d >= 60, max_speed_7d >= 50],
    [-10.0, -6.0, -3.0],
    default=0.0
)

total = total + speed_penalty
```

同时强化静态储能判定，引入速度作为额外门槛：

```python
# 原有条件基础上，增加：近7d最高速度 < 5 km/h 作为辅助条件
avg_speed_7d = df['近7d平均骑行速度_kmh'].fillna(999).values

is_static_storage = (
    (static_discharge_ratio > 0.8) & 
    (max_radius < 2.0) & 
    (discharge_hours > 1.0) &
    (avg_speed_7d < 5.0)   # 新增：均速极低才认定为静态储能
)
```

---

### 2.3 补充 60A-80A 中高电流风险标签

当前 `cond_protect` 只捕获了超保护板电流（>60A），但 60-80A 区间没有独立标签：

```python
over60 = df['近7d电流超60A总次数'].fillna(0).values

# 在现有 cond_over80 之后，增加 60A 区间条件
cond_over60 = (
    (over60 > 10) &          # 7天内超 10 次
    ~cond_violent & ~cond_extreme & ~cond_protect & ~cond_over80
)

risk_tags = np.select(
    [cond_violent, cond_extreme, cond_protect, cond_over80, cond_high_avg, cond_over60],
    ['暴力放电', '极端电流', '超保护板电流', '频繁超80A', '持续高耗流', '中高电流频繁'],
    default=''
)
```

---

### 2.4 修复超100A 判定逻辑一致性

日级 `_determine_user_level_v2` 里判暴力用户的条件是 `over100a_hours >= 0.1`，但 7d `score_and_classify` 里只看次数：

```python
over100_hours = df['近7d超100A累计时长_h'].fillna(0).values

# 修复 is_violent 判定，与日级逻辑对齐
is_violent = (
    (over100 >= 2) | 
    (over100_hours >= 0.5) |   # 新增：7d 累计超 0.5 小时也判暴力
    (max_cur > max_cur_P99) | 
    (monthly_energy_val > MONTHLY_ENERGY_VIOLENT_THRESHOLD)
)
```

---

### 2.5 补充换电次数扣分

```python
total_swaps = df['近7d总换电次数'].fillna(0).values
daily_avg_swaps = total_swaps / 7.0

# 换电扣分（参考日级逻辑：超3次/天开始扣分）
swap_penalty = np.where(
    daily_avg_swaps > 5, -12.0,
    np.where(daily_avg_swaps > 3, -6.0,
    np.where(daily_avg_swaps > 2, -3.0, 0.0))
)

total = total + swap_penalty
```

---

### 2.6 补充 SOC 低于10% 时长占比

```python
soc_below_10_ratio_7d = df['近7d_SOC低于10%时长占比'].fillna(0).values

# 深度亏电风险标签细化
soc_risk = np.select(
    [
        soc_below_10_ratio_7d > 0.05,     # 超5%时长处于深度亏电
        min_soc < 10,                      # 最低SOC绝对值
        soc_low_ratio > 0
    ],
    ['持续深度亏电', '深度亏电', '低SOC告警'],
    default=''
)
```

---

### 2.7 补充 SOC 黄金区间奖励

```python
avg_soc_7d = df['近7d平均骑行SOC'].fillna(0).values

SOC_OPTIMAL_LOWER = 30
SOC_OPTIMAL_UPPER = 80

soc_bonus = np.where(
    (avg_soc_7d >= SOC_OPTIMAL_LOWER) & 
    (avg_soc_7d <= SOC_OPTIMAL_UPPER) & 
    (soc_low_ratio == 0),
    3.0,   # 与日级评分保持一致
    0.0
)

total = total + soc_bonus
```

---

### 2.8 策略建议差异化（接入骑手类型）

当前策略 map 只依赖用户等级，对所有"普通用户"输出"引导升级"。改为结合客户形态：

```python
def _get_strategy(level, customer_type, work_pattern):
    """根据用户等级 + 客户形态 + 工作特点生成差异化策略"""
    base_strategy = {
        '优质用户': '留存激励',
        '良好用户': '维持服务',
        '普通用户': '引导升级',
        '高损耗用户': '限制预警',
        '暴力': '清退处理',
        '观察期': '新手引导',
        '沉默用户': '激活唤醒',
    }.get(level, '维持服务')

    # 骑手类型覆盖
    if customer_type == '专送骑手' and level in ('普通用户', '良好用户'):
        return '专送骑手关怀'
    if customer_type == '地摊/储能':
        return '非正常用电核查'
    if customer_type == '改装/超速车':
        return '风险用户核查'
    if customer_type == '众包骑手' and level == '普通用户':
        return '众包骑手引导'
    
    # 工作时段覆盖
    if work_pattern in ('习惯晚上',) and level in ('高损耗用户',):
        return '夜间高损耗预警'

    return base_strategy

# 应用
df['策略建议'] = [
    _get_strategy(lv, ct, wp)
    for lv, ct, wp in zip(
        df['用户等级_动态'].values,
        df.get('客户形态_综合_7d', pd.Series('未知', index=df.index)).fillna('未知').values,
        df.get('近7d工作特点', pd.Series('', index=df.index)).fillna('').values
    )
]
```

---

## 三、常量补充

在 `score_layer.py` 顶部新增：

```python
# 温度阈值（与 fact_layer 对齐）
TEMP_HIGH_THRESHOLD = 55.0
TEMP_EXTREME_THRESHOLD = 70.0

# 速度阈值
SPEED_HIGH_THRESHOLD = 50.0
SPEED_EXTREME_THRESHOLD = 60.0
SPEED_VIOLENT_THRESHOLD = 80.0

# SOC 黄金区间
SOC_OPTIMAL_LOWER = 30
SOC_OPTIMAL_UPPER = 80
SOC_BONUS_SCORE = 3.0

# 换电阈值
NORMAL_DAILY_SWAPS = 3
HIGH_DAILY_SWAPS = 5

# 60A 电流告警次数门槛
OVER60A_WARNING_COUNT = 10

# 超100A 累计时长门槛（7天）
OVER100A_HOURS_THRESHOLD_7D = 0.5
```

---

## 四、修改后维度对照

| 评分维度 | 日级 `_calc_monthly_score_v2` | 7d `score_and_classify`（修改后） |
|---------|-------------------------------|----------------------------------|
| 平均电流 | ✅ | ✅ |
| 最大电流 | ✅ | ✅ |
| 超80A 次数 | ✅ | ✅ |
| 超100A 次数/时长 | ✅ | ✅（补充时长） |
| 超60A 次数 | ✅ | ✅（新增） |
| 百公里电耗 | ✅ | ✅ |
| SOC低于20%占比 | ✅ | ✅ |
| SOC低于10%占比 | ✅ | ✅（新增） |
| SOC 平均值/奖励 | ✅ | ✅（新增） |
| 温度 | ✅ | ✅（新增） |
| 速度 | ✅ | ✅（新增） |
| 换电次数 | ✅ | ✅（新增） |
| 包月友好分 | ✅ | ✅ |
| 怠速放电比 | — | ✅ |
| 静态储能判定 | ✅ | ✅（补充速度条件） |
| 策略差异化 | — | ✅（新增骑手类型） |

---

## 五、注意事项

**改动顺序建议**：先补高优先级的温度、换电、SOC低10%（独立字段，不影响现有逻辑），再做超100A累计时长修复（涉及 `is_violent` 判定逻辑），最后做策略差异化（涉及多字段联动）。

**阈值回退**：新增字段均加 `.fillna(0)` 或合理默认值，避免历史数据缺字段时崩溃。

**基线更新**：`thresholds_baseline.json` 中目前没有速度和温度的分位数，若要做动态阈值对齐，需在 `DynamicBatteryAnalyzer.run()` 中补充这两个维度的分位数计算。
