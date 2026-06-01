"""
灵敏度分析完整案例 — 展示单参数、双参数、蒙特卡洛三种方法
运行: python examples/sensitivity_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("=" * 60)
print("灵敏度分析完整案例")
print("=" * 60)

# ── 案例背景 ──
# 假设一个经济增长模型: GDP = alpha * K^beta * L^(1-beta)
# 其中 K 是资本投入，L 是劳动力投入，alpha 是全要素生产率

def cobb_douglas(params):
    """柯布-道格拉斯生产函数"""
    alpha = params["alpha"]
    beta = params["beta"]
    K = params["K"]
    L = params["L"]
    return alpha * (K ** beta) * (L ** (1 - beta))

base_params = {"alpha": 1.2, "beta": 0.4, "K": 100, "L": 50}
base_output = cobb_douglas(base_params)
print(f"\n基准参数: {base_params}")
print(f"基准 GDP: {base_output:.2f}")

# ── 1. 单参数灵敏度 ──
print("\n[1/3] 单参数灵敏度分析")
from scripts.mm_utils import sensitivity_analysis

result_alpha = sensitivity_analysis(
    cobb_douglas, base_params, "alpha",
    range_pct=0.3, steps=15
)

values = [r["param_value"] for r in result_alpha]
outputs = [r["output"] for r in result_alpha]

print(f"  alpha 从 {values[0]:.3f} 到 {values[-1]:.3f}")
print(f"  GDP 从 {outputs[0]:.2f} 到 {outputs[-1]:.2f}")
print(f"  弹性: {(outputs[-1]-outputs[0])/outputs[0]*100:.1f}%")

# ── 2. 双参数灵敏度（热力图）──
print("\n[2/3] 双参数灵敏度分析（热力图）")
from scripts.mm_utils import two_factor_sensitivity

X, Y, Z = two_factor_sensitivity(
    cobb_douglas, base_params, ["alpha", "beta"],
    range_pct=0.2, steps=10
)

print(f"  X 范围: {X[0,0]:.3f} ~ {X[0,-1]:.3f}")
print(f"  Y 范围: {Y[0,0]:.3f} ~ {Y[-1,0]:.3f}")
print(f"  Z 范围: {Z.min():.2f} ~ {Z.max():.2f}")

# 画热力图和 3D 图
from scripts.mm_visual import plot_heatmap, plot_3d_surface

fig1 = plot_heatmap(
    Z,
    xticklabels=[f"{x:.2f}" for x in X[0]],
    yticklabels=[f"{y:.2f}" for y in Y[:, 0]],
    title="双参数灵敏度热力图 (α × β)",
    save_path="figures/sensitivity_heatmap.png"
)
print("  → 已保存: figures/sensitivity_heatmap.png")

fig2 = plot_3d_surface(
    X, Y, Z,
    title="双参数灵敏度 3D 曲面",
    xlabel="α (全要素生产率)", ylabel="β (资本弹性)", zlabel="GDP",
    save_path="figures/sensitivity_3d.png"
)
print("  → 已保存: figures/sensitivity_3d.png")

# ── 3. 蒙特卡洛灵敏度 ──
print("\n[3/3] 蒙特卡洛灵敏度分析")
from scripts.mm_utils import monte_carlo_sensitivity

df = monte_carlo_sensitivity(
    cobb_douglas,
    param_distributions={
        "alpha": ("uniform", (0.8, 1.6)),
        "beta": ("uniform", (0.3, 0.5)),
        "K": ("normal", (100, 10)),
        "L": ("normal", (50, 5)),
    },
    n_samples=2000
)

# 相关性分析
corr_matrix = df.corr()
param_cols = [c for c in df.columns if c != "output"]
print("\n  参数与输出的相关性:")
for col in param_cols:
    c = corr_matrix.loc[col, "output"]
    print(f"    {col}: r={c:.4f}")

# 输出统计
print(f"\n  蒙特卡洛输出统计:")
print(f"    Mean GDP: {df['output'].mean():.2f}")
print(f"    Std GDP:  {df['output'].std():.2f}")
print(f"    P5-P95:   {df['output'].quantile(0.05):.2f} ~ {df['output'].quantile(0.95):.2f}")

print("\n" + "=" * 60)
print("✅ 灵敏度分析演示完成！")
print("=" * 60)
