# 数学建模竞赛工具箱

> **不是教程。是**`import` **就用的 2700 行实战工具箱 + 拿奖工作流。**

覆盖国赛/美赛六大题型。 60+ 可复用函数，全部 type-hinted，零除安全，生产级质量。GitHub 上的数学建模仓库很多，但**这是少数几个真能在比赛现场复用的**。这里没有"读完 50 页文档才能开始"的废话。

## 一句话上手

```python
from math_modeling_toolkit import gm11, sensitivity_analysis, plot_comparison

# 预测
result = gm11([58.3, 63.8, 68.9, 74.6, 82.1], forecast_steps=3)
print("预测值:", result["pred"])

# 灵敏度分析（国奖论文分水岭）
sensitivity_analysis(my_model, {"alpha": 0.5}, "alpha", range_pct=0.2)

# 论文级图表（Matplotlib 默认风？不存在的）
plot_comparison([y_true, y_pred], labels=["真实值", "预测值"], style="paper")
```

## 安装

```bash
pip install git+https://github.com/zack59309-maker/math-modeling-toolkit.git
```

或者传统方式：

```bash
git clone https://github.com/zack59309-maker/math-modeling-toolkit.git
cd math-modeling-toolkit
pip install -r requirements.txt
```

## 项目结构

```
math-modeling-toolkit/
├── scripts/
│   ├── mm_optim.py     优化模型（线性/整数/非线性/遗传/退火/粒子群/NSGA-II）
│   ├── mm_predict.py   预测模型（灰色 GM(1,1)/Lasso/Ridge/SVR/BP/LSTM）
│   ├── mm_eval.py      评价模型（AHP/熵权/TOPSIS/模糊/灰色关联/组合赋权）
│   ├── mm_stats.py     统计模型（KMeans/DBSCAN/层次聚类/PCA/t-SNE）
│   ├── mm_visual.py    论文级图表（线图/柱图/热力/3D/灵敏度/对比图）
│   └── mm_utils.py     通用工具（数据清洗/灵敏度分析/误差指标/描述统计）
├── examples/
│   ├── basic_usage.py       — 全部模型快速演示
│   ├── sensitivity_demo.py  — 灵敏度分析完整案例
│   └── gm11_demo.py         — 灰色预测完整案例
├── requirements.txt
├── pyproject.toml
└── LICENSE (MIT)
```

## 拿奖工作流（不是课表，是作战手册）

大部分队伍的问题不是没时间，而是**前 36 小时在犹豫，最后 12 小时赶论文**。

### 发题后 6 小时内：定题

用排除法，不要用选优法：

```
你擅长什么？
├─ 物理/微分方程 → 国赛 A / 美赛 A
├─ 优化/离散 → 国赛 B / 美赛 B/D
├─ 数据/统计 → 国赛 C / 美赛 C
├─ 评价/政策 → 美赛 E/F
└─ 没人拿手 → 淘汰这题，不要比赛时现学
```

**30 分钟定题，队长拍板，禁止回头。**

### 第 6-24 小时：跑出第一版结果（不管多丑）

**铁律：Day 2 晚上前必须有数字结果。** 没有结果就没法写论文。

```python
# 先用最简单的算法跑基线
from math_modeling_toolkit import gm11, topsis, linprog_safe, descriptive_stats

# 预测题 → 灰色预测先跑
pred = gm11(data, forecast_steps=5)

# 评价题 → TOPSIS 先无权重跑
scores = topsis(matrix, weights=None)

# 优化题 → 线性规划先跑松弛解
result = linprog_safe(c, A_ub=b, b_ub=b)

# 数据 → 描述统计
stats = descriptive_stats(data)
```

> 第一版不需要准，它只是给你一个**基线**。后面所有改进都用来回答"比这个基线好多少"。

### 第 24-36 小时：模型迭代——至少 2-3 个方法对比

国赛评委的逻辑：**不是你创新不创新，而是你有没有做对比分析。**

| 如果你用了 | 再对比 | 证明什么 |
|-----------|--------|---------|
| GM(1,1) | 残差修正 GM(1,1) | 改进有效 |
| 线性回归 | Lasso / Ridge | 正则化有效 |
| AHP | AHP + 熵权 | 主客观结合更好 |
| 单目标 | 多目标 Pareto | 权衡更好 |
| TOPSIS | 熵权 TOPSIS | 客观权重提升区分度 |

### 第 36-48 小时：灵敏度分析（必做）+ 可视化

**这是国奖和普通论文的分水岭。没有灵敏度分析的论文直接扣分。**

```python
from math_modeling_toolkit import sensitivity_analysis, plot_sensitivity_curve

result = sensitivity_analysis(
    model_func=my_model,
    base_params={"alpha": 0.5, "beta": 0.3},
    param_to_vary="alpha",
    range_pct=0.3,  # ±30%
)
plot_sensitivity_curve(result, title="参数 α 的灵敏度", style="paper")
```

论文里不能只说"结果稳定"——要有数字：
> **"当 α 在 ±20% 范围内变化时，目标值波动 ≤ 4.7%"**

### 第 48-96 小时：论文

**摘要占 40% 分。** 评委先看摘要，不行直接降档。

六句话模板（100 字内）：

```
[背景]针对XXX问题，建立了XXX模型。
[方法]模型以XXX为目标，采用XXX算法，理由是XXX。
[求解]求解得到XXX结果（必须有数字）。
[验证]灵敏度分析表明，XXX变化±20%时，结果波动<5%。
[对比]与传统XXX方法相比，精度提升XXX%。
[结论]该模型适用于XXX场景。
```

## 为什么这个工具箱比你自己写的好

| 问题 | 这个工具箱怎么解决的 |
|------|-------------------|
| 比赛时慌，从零写代码 | 全函数 `import` 即用，不需要写一行重复代码 |
| 边界条件没处理（除零/空数组） | 每个除法都有 epsilon 保护，边缘情况全覆盖 |
| 图表像课堂作业 | `style='paper'` 一键切换论文级配色和字体 |
| 参数一改又要重跑 | `sensitivity_analysis` 自动扫参数范围 |
| 不知道怎么选模型 | 决策树直接告诉你"什么情况用什么" |
| 摘要不会写 | 六句话模板，填空即可 |

## License

MIT — 随便用，比赛加油 🚀
