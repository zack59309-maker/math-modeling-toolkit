# 数学建模竞赛工具箱 — 基本使用示例
# 运行: python examples/basic_usage.py

import sys
sys.path.insert(0, '..')

import numpy as np

print("=" * 60)
print("数学建模竞赛工具箱 — 快速演示")
print("=" * 60)

# ── 1. 数据预处理 ──
print("\n[1/6] 数据预处理")
from scripts.mm_utils import fill_missing, normalize, detect_outliers_iqr

data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 100.0, 7.0])
print(f"  原始:        {data}")
print(f"  填充缺失值: {fill_missing(data, 'linear')}")
print(f"  归一化:      {normalize(np.array([[1,2,3],[4,5,6]]), 'minmax')}")
print(f"  异常值检测:  {detect_outliers_iqr(data)}")

# ── 2. 灰色预测 ──
print("\n[2/6] 灰色预测 GM(1,1)")
from scripts.mm_predict import gm11

X = np.array([58.3, 63.8, 68.9, 74.6, 82.1, 90.2, 99.1])
result = gm11(X, forecast_steps=3)
print(f"  原始数据:  {X}")
print(f"  拟合值:    {np.round(result['fitted'], 2)}")
print(f"  预测值:    {np.round(result['pred'], 2)}")
print(f"  MAPE:      {result['mape']:.2f}%")
print(f"  发展系数 a:{result['a']:.4f}, 灰色作用量 b:{result['b']:.4f}")

# ── 3. 线性规划 ──
print("\n[3/6] 线性规划")
from scripts.mm_optim import solve_lp

result = solve_lp(
    c=[-3, -2],  # max 3x+2y → min -3x-2y
    A_ub=[[2, 1], [1, 1]],
    b_ub=[18, 8],
    bounds=[(0, None), (0, None)]
)
print(f"  最优解: x={result['x'][0]:.2f}, y={result['x'][1]:.2f}")
print(f"  最大值: {-result['fval']:.2f}")

# ── 4. 评价模型 ──
print("\n[4/6] 评价模型")
from scripts.mm_eval import topsis, entropy_weight

# 4个方案 × 3个指标
data = np.array([
    [85, 90, 70],
    [78, 85, 92],
    [92, 72, 80],
    [70, 88, 85],
])
weights = entropy_weight(data)
print(f"  熵权法权重: {np.round(weights, 4)}")

eval_result = topsis(data, weights=weights)
print(f"  TOPSIS得分: {eval_result['score']}")
print(f"  排名:       {eval_result['rank']}")

# ── 5. 聚类分析 ──
print("\n[5/6] 聚类分析")
from scripts.mm_stats import kmeans_clustering, elbow_method

np.random.seed(42)
cluster_data = np.vstack([
    np.random.randn(20, 2) + [0, 0],
    np.random.randn(20, 2) + [5, 5],
    np.random.randn(20, 2) + [0, 5],
])

elbow = elbow_method(cluster_data, max_k=6)
print(f"  建议 K 值: {elbow['best_k_by_silhouette']}")

clusters = kmeans_clustering(cluster_data, n_clusters=3)
print(f"  轮廓系数:  {clusters['silhouette_score']}")

# ── 6. 可视化 ──
print("\n[6/6] 可视化")
from scripts.mm_visual import plot_line, plot_bar

x = np.arange(10)
y1 = np.sin(x)
y2 = np.cos(x)

fig = plot_line(
    x, [y1, y2],
    labels=["sin(x)", "cos(x)"],
    title="sin 和 cos 曲线",
    xlabel="x", ylabel="y",
    save_path="../figures/demo_line.png"
)

fig = plot_bar(
    ["A", "B", "C", "D"],
    [85, 92, 78, 88],
    labels=["得分"],
    title="方案得分对比",
    save_path="../figures/demo_bar.png"
)
print("  图表已保存: figures/demo_line.png, figures/demo_bar.png")

print("\n" + "=" * 60)
print("✅ 全部演示完成！")
print("=" * 60)
