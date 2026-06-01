# 数学建模竞赛工具箱

> **Math Modeling Toolkit** — 国赛/美赛/Python 数学建模工具箱，覆盖六大题型，80+ 可复用函数，即拿即用。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 快速上手

```python
# 例子：灰色预测 + 灵敏度分析
from scripts.mm_predict import gm11
from scripts.mm_utils import error_report
import numpy as np

# 原始数据（如 GDP 历年数据）
data = np.array([58.3, 63.8, 68.9, 74.6, 82.1])

# 预测未来 3 年
result = gm11(data, forecast_steps=3)
print("预测值:", result["pred"])
print("MAPE:", f"{result['mape']:.2f}%")
```

## 工具一览

| 模块 | 功能 | 核心函数 |
|------|------|---------|
| `scripts/mm_utils.py` | 数据预处理 + 灵敏度分析 + 误差指标 | `fill_missing`, `normalize`, `sensitivity_analysis`, `error_report` |
| `scripts/mm_optim.py` | 优化模型 | `solve_lp`, `solve_nlp`, `genetic_algorithm`, `simulated_annealing`, `particle_swarm`, `nsga2_mo`, `knapsack_01` |
| `scripts/mm_predict.py` | 预测模型 | `gm11`, `gm11_improved`, `linear_regression`, `polynomial_regression`, `HoltWinters`, `SimpleBP` |
| `scripts/mm_eval.py` | 评价/决策模型 | `ahp_weight`, `entropy_weight`, `topsis`, `fuzzy_eval`, `grey_relation_degree`, `critic_weight` |
| `scripts/mm_stats.py` | 统计与数据挖掘 | `kmeans_clustering`, `dbscan_clustering`, `pca_analysis`, `normality_test`, `anova_one_way` |
| `scripts/mm_visual.py` | 论文级图表 | `plot_line`, `plot_bar`, `plot_heatmap`, `plot_3d_surface`, `plot_comparison`, `plot_pareto_front` |

## 安装

```bash
pip install -r requirements.txt
# 或
pip install numpy scipy pandas matplotlib scikit-learn statsmodels
```

> **可选依赖**（特定场景需要）
> - `pulp` — 整数规划（CBC 求解器）
> - `cvxpy` — 凸优化
> - `ortools` — Google OR-Tools（路径规划等）
> - `prophet` — Facebook Prophet 时序预测

## 六大题型速查

| 题型 | 适用问题 | 推荐模型 |
|------|---------|---------|
| **优化规划** | 路径/排班/生产/选址/资源分配 | 线性/整数/非线性规划、遗传算法、模拟退火 |
| **预测** | GDP/人口/股票/销量/温度 | 灰色预测、回归、指数平滑、BP神经网络 |
| **评价决策** | 综合排名/方案比选/风险评估 | AHP、熵权法、TOPSIS、模糊评价、CRITIC |
| **微分方程** | 传染病/生态/物理过程/SIR模型 | 参数辨识、数值求解（需 scipy.integrate） |
| **图论网络** | 最短路径/最小生成树/最大流 | Dijkstra、Kruskal、Floyd（需 networkx） |
| **统计挖掘** | 分类/聚类/降维/异常检测 | K-Means、DBSCAN、PCA、PCA+t-SNE |

## 比赛工作流

```
Day 1 上午 ── 选题读题，拆解问题
Day 1 下午 ── 数据准备，EDA 探索性分析(plot_scatter, descriptive_stats)
Day 2 全天 ── 建模求解，跑出第一版结果（铁律: Day2 晚前必须有结果）
Day 3 上午 ── 灵敏度分析(sensitivity_analysis)，模型验证
Day 3 下午 ── 结果可视化(plot_comparison, plot_heatmap)，插图
Day 4 全天 ── 论文写作，摘要打磨，提交
```

## 灵敏度分析为什么重要

国赛/美赛评分中，**没有灵敏度分析的论文直接扣 5-10 分**。本工具箱提供三种：

```python
from scripts.mm_utils import sensitivity_analysis, two_factor_sensitivity, monte_carlo_sensitivity

# 1. 单参数灵敏度（必做）
result = sensitivity_analysis(my_model, base_params, "alpha", range_pct=0.2, steps=15)

# 2. 双参数热力图（加分项）
X, Y, Z = two_factor_sensitivity(my_model, base_params, ["alpha", "beta"])

# 3. 蒙特卡洛灵敏度（美赛加分）
df = monte_carlo_sensitivity(my_model, {"alpha": ("uniform", (0, 1))})
```

## 示例脚本

`examples/` 目录下有完整可运行的示例：

- `examples/basic_usage.py` — 所有模型的快速演示
- `examples/sensitivity_demo.py` — 灵敏度分析完整案例
- `examples/gm11_demo.py` — 灰色预测完整案例

## 论文写作

摘要占 **40%** 分数。格式：

```
[背景] 针对XXX问题，建立了XXX模型。
[方法] 以XXX为目标，在XXX约束下求解。
[求解] 采用XXX算法，得到XXX结果。
[验证] 灵敏度分析表明，当XXX变化±20%时，结果波动<5%。
[对比] 与传统方法相比，精度提升XXX%。
[结论] 模型适用于XXX场景。

全文不超过300字。
```

## 项目结构

```
math-modeling-toolkit/
├── scripts/               ← 核心算法库（6个模块，80+函数）
│   ├── mm_utils.py        — 数据预处理、灵敏度分析、误差指标
│   ├── mm_optim.py        — 优化模型（LP/ILP/NLP/GA/SA/PSO/NSGA-II）
│   ├── mm_predict.py      — 预测模型（灰色/回归/平滑/BP）
│   ├── mm_eval.py         — 评价模型（AHP/熵权/TOPSIS/模糊/CRITIC）
│   ├── mm_stats.py        — 统计模型（聚类/PCA/检验/ANOVA）
│   └── mm_visual.py       — 论文级可视化（配色/字体预设）
├── examples/              ← 可运行示例
│   ├── basic_usage.py     — 全部模型快速演示
│   ├── sensitivity_demo.py— 灵敏度分析完整案例
│   └── gm11_demo.py       — 灰色预测完整案例
├── figures/               — 示例输出图表
├── requirements.txt       — 依赖清单
├── pyproject.toml         — 项目配置
├── LICENSE                — MIT 许可证
└── README.md              ← 你现在在这里
```

## License

MIT — 随便用，比赛加油。
