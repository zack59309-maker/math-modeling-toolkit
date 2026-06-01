# 数学建模竞赛工具箱

> 2700 行 Python 工具箱 + 比赛全流程速查。覆盖国赛/美赛六大题型，60+ 可复用函数。

```python
from math_modeling_toolkit import gm11, sensitivity_analysis, plot_comparison

# 预测
result = gm11([58.3, 63.8, 68.9, 74.6, 82.1], forecast_steps=3)

# 灵敏度分析
sensitivity_analysis(my_model, {"alpha": 0.5}, "alpha", range_pct=0.2)

# 论文级图表
plot_comparison([y_true, y_pred], labels=["真实值", "预测值"], style="paper")
```

## 安装

```bash
pip install git+https://github.com/zack59309-maker/math-modeling-toolkit.git
```

## 模块

| 模块 | 核心函数 |
|------|---------|
| `mm_optim` | `linprog_safe`, `genetic_algorithm`, `simulated_annealing`, `particle_swarm`, `nsga2_mo` |
| `mm_predict` | `gm11`, `arima_model`, `lasso_regression`, `ridge_regression`, `svr_model`, `bpnn_predict`, `lstm_predict` |
| `mm_eval` | `ahp`, `entropy_weight`, `topsis`, `fuzzy_eval`, `grey_relation`, `combined_weight` |
| `mm_stats` | `hierarchical_cluster`, `dbscan_cluster`, `kmeans_cluster`, `pca_transform`, `tsne_transform` |
| `mm_visual` | `plot_line`, `plot_bar`, `plot_heatmap`, `plot_3d_surface`, `plot_sensitivity_curve`, `plot_comparison` |
| `mm_utils` | `fill_missing`, `normalize`, `sensitivity_analysis`, `two_factor_sensitivity`, `rmse`, `mae`, `mape` |

## 比赛流程

### 1. 定题
排除法：淘汰做不了的 → 淘汰数据拿不到的 → 剩下选你最熟的。锁定了不准换。

### 2. 跑第一版
不管结果多丑，先跑通。第一版是基线，不是答案。

| 题型 | 先跑这个 |
|------|---------|
| 优化 | `scipy.optimize.linprog` 线性松弛 |
| 预测 | `gm11()` 灰色预测 |
| 评价 | `topsis()` 不加权 |

### 3. 对比
至少 2-3 个模型。国赛不看创新，看你有没有对比。
GM(1,1) → 残差修正 | 线性回归 → Lasso/Ridge | AHP → AHP+熵权 | TOPSIS → 熵权 TOPSIS

### 4. 灵敏度 + 画图
灵敏度是国奖分水岭。论文里写"α 变化 ±20%，目标波动 ≤4.7%"，别只写"结果稳定"。
图表用 `style='paper'`，Matplotlib 默认配色放论文 = 扣分。

### 5. 写论文
摘要占 40 分。六句话：
`[背景]针对 XXX 问题建立 XXX 模型。[方法]采用 XXX 算法（说理由），得到 XXX 结果（有数字）。[验证]灵敏度：XXX 变化 ±20%，波动 <5%。[对比]与传统 XXX 相比提升 XXX%。[结论]适用 XXX 场景。`

## 选模型速查

| 题型 | 怎么做 |
|------|--------|
| 优化 | 连续→线性规划 / 整数→整数规划 / NP 难→遗传/退火 / 多目标→NSGA-II |
| 预测 | 平稳→回归 / 趋势→GM(1,1) / 数据多→随机森林 / 非线性→SVR/LSTM |
| 评价 | 主观→AHP / 客观→熵权 TOPSIS / 模糊→模糊综合评价 |
| 微分方程 | 传染病→SIR / 人口→Logistic / 参数未知→最小二乘 |
| 图论 | 最短路径→Dijkstra / 最小成本→MST / 分配→匈牙利 |
| 统计 | 分类→SVM/随机森林 / 聚类→K-Means/DBSCAN / 降维→PCA |

## 项目结构

```
scripts/      6 个模块，直接 import
examples/     3 个可运行示例
requirements.txt + pyproject.toml
```

## License

MIT
