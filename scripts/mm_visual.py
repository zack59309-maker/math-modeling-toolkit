"""
mm_visual.py — 数学建模竞赛风格可视化

论文级图表：配色、字体、布局专为竞赛论文优化。
所有函数返回 matplotlib figure 对象，可进一步自定义。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 激活 3D 投影
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ── 配色方案（色盲友好、打印友好）──

COLORS: Dict[str, str] = {
    "blue": "#2E86AB",
    "red": "#A23B72",
    "green": "#3C887E",
    "orange": "#F18F01",
    "purple": "#6B2D82",
    "gray": "#6C757D",
    "light_blue": "#8DB8D2",
    "light_red": "#C97B9E",
    "light_green": "#7BB5A4",
}

COLOR_LIST: List[str] = [
    "#2E86AB", "#A23B72", "#3C887E", "#F18F01",
    "#6B2D82", "#C97B9E", "#8DB8D2", "#7BB5A4",
    "#E6A176", "#B88BB4",
]


# ── 字体设置 ──

def set_chinese_font() -> Optional[str]:
    """
    设置中文字体（解决 matplotlib 中文乱码）。
    会自动检测系统可用中文字体。
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for font in [
            "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
            "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
            "STSong", "FangSong", "KaiTi",
        ]:
            try:
                matplotlib.font_manager.findfont(font, fallback_to_default=False)
                plt.rcParams["font.sans-serif"] = [font] + plt.rcParams["font.sans-serif"]
                plt.rcParams["axes.unicode_minus"] = False
                return font
            except Exception:
                continue
        plt.rcParams["axes.unicode_minus"] = False
        return None


def set_paper_style() -> None:
    """设置论文风格样式"""
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "grid.alpha": 0.3,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ═══════════════════════════════════════════════
# 基本图表
# ═══════════════════════════════════════════════

def plot_line(
    x: np.ndarray,
    y: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[Union[str, List[str]]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    style: str = "paper",
    save_path: Optional[str] = None,
    show_legend: bool = True,
) -> plt.Figure:
    """
    折线图（论文风格）。

    Parameters
    ----------
    y : array or list of arrays
    style : str
        "paper" — 无网格，简洁
        "academic" — 有网格，更正式

    Returns
    -------
    plt.Figure
    """
    if style == "paper":
        set_paper_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(y, np.ndarray):
        y_list = [y]
    else:
        y_list = y
    if isinstance(labels, str):
        labels_list = [labels]
    else:
        labels_list = labels

    for i, yi in enumerate(y_list):
        ax.plot(
            x, yi,
            color=COLOR_LIST[i % len(COLOR_LIST)],
            linewidth=2, marker="o", markersize=4,
            label=labels_list[i] if (labels_list and i < len(labels_list)) else None,
        )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_legend and labels_list and any(labels_list):
        ax.legend(frameon=False)

    if style == "academic":
        ax.grid(True, alpha=0.3, linestyle="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_bar(
    categories: Sequence[str],
    values: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[Union[str, List[str]]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    horizontal: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    柱状图（分组柱状图）。
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(values, np.ndarray):
        values_list = [values]
    else:
        values_list = values
    if isinstance(labels, str):
        labels_list = [labels]
    else:
        labels_list = labels

    n_groups = len(values_list)
    n_cats = len(categories)
    width = 0.8 / n_groups
    x = np.arange(n_cats)

    for i, v in enumerate(values_list):
        offset = (i - n_groups / 2 + 0.5) * width
        label = labels_list[i] if (labels_list and i < len(labels_list)) else None
        kwargs = dict(width=width, label=label, color=COLOR_LIST[i % len(COLOR_LIST)])
        if horizontal:
            ax.barh(x + offset, v, **kwargs)
        else:
            ax.bar(x + offset, v, **kwargs)

    ax.set_title(title, fontweight="bold", pad=12)
    if horizontal:
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right")

    if labels_list and any(labels_list):
        ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    c: Optional[np.ndarray] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """散点图。"""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(x, y, c=c, cmap="viridis", s=30, alpha=0.7, edgecolors="none")
    if c is not None:
        plt.colorbar(scatter, ax=ax)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════
# 热力图
# ═══════════════════════════════════════════════

def plot_heatmap(
    data: np.ndarray,
    xticklabels: Optional[Sequence[str]] = None,
    yticklabels: Optional[Sequence[str]] = None,
    title: str = "",
    cmap: str = "RdBu_r",
    annotate: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    热力图（灵敏度分析、相关性矩阵等）。
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)

    if annotate:
        vmax = float(np.nanmax(np.abs(data)))
        threshold = vmax * 0.5 if vmax > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                color = "white" if abs(val) > threshold else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    if xticklabels:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    if yticklabels:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    ax.set_title(title, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════
# 3D 曲面图
# ═══════════════════════════════════════════════

def plot_3d_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """3D 曲面图。"""
    set_paper_style()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════
# 灵敏度分析曲线
# ═══════════════════════════════════════════════

def plot_sensitivity_curve(
    sensitivity_results: List[Dict[str, float]],
    title: str = "灵敏度分析",
    xlabel: str = "参数变化",
    ylabel: str = "模型输出",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    灵敏度分析曲线图。

    Parameters
    ----------
    sensitivity_results : list of dict
        每个元素来自 mm_utils.sensitivity_analysis 的返回
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    param_values = [r["param_value"] for r in sensitivity_results]
    outputs = [r["output"] for r in sensitivity_results]

    base_output = outputs[len(outputs) // 2]

    ax.plot(param_values, outputs, "o-", color=COLORS["blue"], linewidth=2, markersize=6)
    ax.axhline(y=base_output, color=COLORS["red"], linestyle="--", alpha=0.5, label=f"基准值={base_output:.4f}")
    ax.fill_between(param_values, base_output, outputs, alpha=0.1, color=COLORS["blue"])

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════
# 对比图
# ═══════════════════════════════════════════════

def plot_comparison(
    data: List[np.ndarray],
    labels: List[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
    style: str = "paper",
) -> plt.Figure:
    """
    多条曲线对比图（模型对比、预测 vs 真实等）。
    """
    if style == "paper":
        set_paper_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(data[0]))

    for i, (d, label) in enumerate(zip(data, labels)):
        ax.plot(x, d, color=COLOR_LIST[i % len(COLOR_LIST)], linewidth=2, label=label)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if style == "academic":
        ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════
# Pareto 前沿图
# ═══════════════════════════════════════════════

def plot_pareto_front(
    pareto_front: List[Tuple[float, ...]],
    all_solutions: Optional[List[Tuple[float, ...]]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Pareto 前沿",
    xlabel: str = "目标 1",
    ylabel: str = "目标 2",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """双目标 Pareto 前沿图。"""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    front = np.array(pareto_front)

    if all_solutions:
        all_sol = np.array(all_solutions)
        ax.scatter(all_sol[:, 0], all_sol[:, 1], c=COLORS["gray"], alpha=0.3, s=20, label="所有解")

    sorted_idx = np.argsort(front[:, 0])
    front_sorted = front[sorted_idx]

    ax.plot(front_sorted[:, 0], front_sorted[:, 1], color=COLORS["red"], linewidth=2, label="Pareto 前沿")
    ax.scatter(front[:, 0], front[:, 1], c=COLORS["red"], s=50, zorder=5)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════
# 子图模板
# ═══════════════════════════════════════════════

def subplots_grid(
    n_plots: int,
    n_cols: int = 2,
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    创建一致风格的子图网格。

    Returns
    -------
    fig, axes : list of Axes
    """
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for ax in axes_flat[:n_plots]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig, axes_flat[:n_plots]


# ═══════════════════════════════════════════════
# LaTeX 插图助手
# ═══════════════════════════════════════════════

def figure_caption(caption: str, label: str = "") -> str:
    """
    生成 LaTeX 图表标题代码。
    """
    tex = r"\begin{figure}[htbp]" + "\n"
    tex += r"  \centering" + "\n"
    tex += r"  \includegraphics[width=0.8\textwidth]{figures/comparison.png}" + "\n"
    tex += f"  \\caption{{{caption}}}\n"
    if label:
        tex += f"  \\label{{{label}}}\n"
    tex += r"\end{figure}" + "\n"
    return tex
