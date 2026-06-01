# 数学建模竞赛工具箱

> **Math Modeling Toolkit** — 开箱即用的 Python 数学建模工具箱，覆盖国赛/美赛六大题型，70+ 可复用函数，即拿即用。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 一句话上手

```python
from scripts.mm_predict import gm11
result = gm11(np.array([58.3, 63.8, 68.9, 74.6, 82.1]), forecast_steps=3)
print("预测值:", result["pred"])
```

## 工具一览

| 模块 | 功能 | 亮点函数 |
|------|------|---------|
| `mm_utils.py` | 数据预处理 + 灵敏度分析 + 误差指标 | `sensitivity_analysis`, `two_factor_sensitivity`, `error_report` |
| `mm_optim.py` | 优化模型 | `solve_lp`, `genetic_algorithm`, `simulated_annealing`, `particle_swarm`, `nsga2_mo` |
| `mm_predict.py` | 预测模型 | `gm11`, `gm11_improved`, `linear_regression`, `holt_winters`, `SimpleBP` |
| `mm_eval.py` | 评价/决策模型 | `ahp_weight`, `entropy_weight`, `topsis`, `fuzzy_eval`, `critic_weight` |
| `mm_stats.py` | 统计与数据挖掘 | `kmeans_clustering`, `dbscan_clustering`, `pca_analysis`, `normality_test` |
| `mm_visual.py` | 论文级图表 | `plot_line`, `plot_bar`, `plot_heatmap`, `plot_3d_surface`, `plot_comparison` |

## 安装

```bash
pip install -r requirements.txt
```

## 比赛工作流

```
Day 1 上午 ── 选题读题，拆解问题
Day 1 下午 ── 数据准备，EDA (descriptive_stats, plot_scatter)
Day 2 全部 ── 建模求解，跑出第一版结果（铁律: Day2晚前必须有结果）
Day 3 上午 ── 灵敏度分析 (sensitivity_analysis)，模型验证
Day 3 下午 ── 结果可视化 (plot_comparison, plot_heatmap)
Day 4 全部 ── 论文写作，摘要打磨（摘要占40%分！）
```

## 灵敏度分析为什么重要

国赛/美赛评分中，**没有灵敏度分析的论文直接扣 5-10 分**。

```python
# 单参数灵敏度（必做）
sensitivity_analysis(my_model, base_params, "alpha", range_pct=0.2, steps=15)

# 双参数热力图（加分项）
two_factor_sensitivity(my_model, base_params, ["alpha", "beta"])

# 蒙特卡洛灵敏度（美赛加分）
monte_carlo_sensitivity(my_model, {"alpha": ("uniform", (0, 1))})
```

## 项目结构

```
math-modeling-toolkit/
├── scripts/           ← 6个模块，70+函数，直接 import 用
│   ├── mm_utils.py     数据预处理 + 灵敏度分析 + 误差指标
│   ├── mm_optim.py     优化模型 (LP/ILP/GA/SA/PSO/NSGA-II)
│   ├── mm_predict.py   预测模型 (灰色/回归/平滑/BP/GP)
│   ├── mm_eval.py      评价模型 (AHP/熵权/TOPSIS/模糊/CRITIC)
│   ├── mm_stats.py     统计模型 (聚类/PCA/检验/ANOVA)
│   └── mm_visual.py    论文级可视化 (配色/字体预设)
├── examples/          ← 3个可运行示例
│   ├── basic_usage.py      — 全部模型快速演示
│   ├── sensitivity_demo.py — 灵敏度分析完整案例
│   └── gm11_demo.py        — 灰色预测完整案例
├── requirements.txt   — 依赖清单
├── pyproject.toml     — pip install 一键安装
└── LICENSE            — MIT 许可证
```

## License

MIT — 随便用，比赛加油 🚀
