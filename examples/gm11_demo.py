"""
灰色预测 GM(1,1) 完整案例
运行: python examples/gm11_demo.py
"""

import sys
sys.path.insert(0, '..')

import numpy as np

print("=" * 60)
print("灰色预测 GM(1,1) 完整案例")
print("=" * 60)

# ── 中国 GDP 数据（亿元，1980-2000）──
gdp = np.array([
    4587.6, 4935.8, 5373.4, 6020.9, 7278.5,
    9098.9, 10376.2, 12174.6, 15180.4, 17179.7,
    18872.9, 22005.6, 27194.5, 35673.2, 48637.5,
    61339.9, 71813.6, 79715.0, 85195.5, 90564.4,
    100280.1
])

years = np.arange(1980, 2001)

# ── 用前 15 年 (1980-1994) 预测后 6 年 (1995-2000) ──
print(f"\n训练数据: 1980-1994 ({len(gdp[:15])} 个样本)")
print(f"待预测:   1995-2000 ({len(gdp[15:])} 个样本)")

from scripts.mm_predict import gm11, gm11_improved, gm11_test_accuracy
from scripts.mm_utils import error_report

# 基础 GM(1,1)
result = gm11(gdp[:15], forecast_steps=6)

# 残差修正 GM(1,1)
result_improved = gm11_improved(gdp[:15], forecast_steps=6)

# ── 精度检验 ──
print("\n── 精度检验 ──")
acc = gm11_test_accuracy(result)
print(f"  后验差比 C: {acc['C']:.4f}")
print(f"  小误差概率 p: {acc['p']:.4f}")
print(f"  整体等级: {acc['grade']}")

# ── 拟合效果 ──
print("\n── 拟合效果（前15年）──")
print(f"  {'年份':<6} {'实际':<10} {'拟合':<10} {'残差':<10} {'相对误差%':<10}")
for i in range(15):
    print(f"  {years[i]:<6} {gdp[i]:<10.1f} {result['fitted'][i]:<10.1f} "
          f"{result['residual'][i]:<10.1f} {result['relative_error'][i]:<10.2f}")

# ── 预测效果 ──
print("\n── 预测效果（后6年）──")
print(f"  {'年份':<6} {'实际':<10} {'GM(1,1)':<10} {'残差修正':<10}")

for i in range(6):
    print(f"  {years[15+i]:<6} {gdp[15+i]:<10.1f} {result['pred'][i]:<10.1f} "
          f"{result_improved['pred'][i]:<10.1f}")

# ── 误差对比 ──
print("\n── 误差对比 ──")
report_base = error_report(gdp[15:], result["pred"])
report_impr = error_report(gdp[15:], result_improved["pred"])

print(f"  {'指标':<12} {'GM(1,1)':<12} {'残差修正':<12}")
for key in report_base:
    print(f"  {key:<12} {str(report_base[key]):<12} {str(report_impr[key]):<12}")

# ── 未来预测 ──
print("\n── 使用全部 21 年数据预测 2001-2005 ──")
result_full = gm11(gdp, forecast_steps=5)
print(f"  {'年份':<8} {'预测值':<12}")
for i, y in enumerate(range(2001, 2006)):
    print(f"  {y:<8} {result_full['pred'][i]:<12.1f}")

print(f"\n  发展系数 a = {result_full['a']:.4f}")
print(f"  灰色作用量 b = {result_full['b']:.4f}")

# ── 画图 ──
from scripts.mm_visual import plot_comparison

pred_years = np.arange(1995, 2001)
all_years = np.arange(1980, 2001)

# 合成对比数据
observed = gdp[15:]
pred_basic = result["pred"]
pred_impr = result_improved["pred"]

fig = plot_comparison(
    [observed, pred_basic, pred_impr],
    labels=["实际值", "GM(1,1)", "残差修正 GM(1,1)"],
    title="灰色预测模型对比 (1980-2000 GDP)",
    ylabel="GDP（亿元）",
    save_path="../figures/gm11_comparison.png"
)
print("\n  → 已保存: figures/gm11_comparison.png")

print("\n" + "=" * 60)
print("✅ GM(1,1) 演示完成！")
print("=" * 60)
