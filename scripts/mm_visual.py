"""
mm_visual.py — 数学建模竞赛风格可视化

论文级图表：配色、字体、布局专为竞赛论文优化。
所有函数返回 matplotlib figure 对象，可进一步自定义。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.gridspec import GridSpec


# ── 配色方案 ──

# 竞赛论文推荐配色（色盲友好、打印友好）
COLORS = {
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

COLOR_LIST = [
    "#2E86AB", "#A23B72", "#3C887E", "#F18F01",
    "#6B2D82", "#C97B9E", "#8DB8D2", "#7BB5A4",
    "#E6A176", "#B88BB4",
]


# ── 字体设置 ──

def set_chinese_font():
    """
    设置中文字体（解决 matplotlib 中文乱码）。
    会自动检测系统可用中文字体。
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 尝试常见中文字体
        for font in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
                     "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                     "STSong", "FangSong", "KaiTi"]:
            try:
                matplotlib.font_manager.findfont(font, fallback_to_default=False)
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                return font
            except Exception:
                continue
        # 最后一个备选
        plt.rcParams['axes.unicode_minus'] = False


def set_paper_style():
    """设置论文风格样式"""
    plt.rcParams.update({
        'figure.dpi': 150,
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'grid.alpha': 0.3,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
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
        单条或多条折线
    style : str
        "paper" — 无网格，简洁
        "academic" — 有网格，更正式
    """
    if style == "paper":
        set_paper_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if isinstance(y, np.ndarray):
        y = [y]
    if isinstance(labels, str):
        labels = [labels]
    
    for i, yi in enumerate(y):
        label = labels[i] if (labels and i < len(labels)) else None
        ax.plot(x, yi, color=COLOR_LIST[i % len(COLOR_LIST)],
                linewidth=2, marker='o', markersize=4, label=label)
    
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if show_legend and (labels and any(labels)):
        ax.legend(frameon=False)
    
    if style == "academic":
        ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


def plot_bar(
    categories: List[str],
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if isinstance(values, np.ndarray):
        values = [values]
    if isinstance(labels, str):
        labels = [labels]
    
    n_groups = len(values)
    n_cats = len(categories)
    width = 0.8 / n_groups
    
    x = np.arange(n_cats)
    
    for i, v in enumerate(values):
        label = labels[i] if (labels and i < len(labels)) else None
        offset = (i - n_groups/2 + 0.5) * width
        if horizontal:
            ax.barh(x + offset, v, width, label=label, color=COLOR_LIST[i % len(COLOR_LIST)])
        else:
            ax.bar(x + offset, v, width, label=label, color=COLOR_LIST[i % len(COLOR_LIST)])
    
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel if not horizontal else ylabel)
    ax.set_ylabel(ylabel if not horizontal else xlabel)
    
    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha='right')
    
    if labels and any(labels):
        ax.legend(frameon=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
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
    """
    散点图。
    """
    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter = ax.scatter(x, y, c=c, cmap='viridis', s=30,
                         alpha=0.7, edgecolors='none')
    
    if c is not None:
        plt.colorbar(scatter, ax=ax)
    
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


# ═══════════════════════════════════════════════
# 热力图
# ═══════════════════════════════════════════════

def plot_heatmap(
    data: np.ndarray,
    xticklabels: Optional[List[str]] = None,
    yticklabels: Optional[List[str]] = None,
    title: str = "",
    cmap: str = "RdBu_r",
    annotate: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    热力图（灵敏度分析、相关性矩阵等）。
    """
    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax)
    
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}",
                        ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(data[i, j]) > data.max() * 0.5 else "black")
    
    if xticklabels:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    if yticklabels:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    
    ax.set_title(title, fontweight='bold', pad=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
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
    """
    3D 曲面图（双参数灵敏度分析等）。
    """
    set_paper_style()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


# ═══════════════════════════════════════════════
# 灵敏度分析曲线
# ═══════════════════════════════════════════════

def plot_sensitivity_curve(
    sensitivity_results: List[Dict],
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    param_values = [r["param_value"] for r in sensitivity_results]
    outputs = [r["output"] for r in sensitivity_results]
    
    # 转换为百分比变化
    base_output = outputs[len(outputs)//2]
    
    ax.plot(param_values, outputs, 'o-', color=COLORS["blue"],
            linewidth=2, markersize=6)
    ax.axhline(y=base_output, color=COLORS["red"], linestyle='--',
               alpha=0.5, label=f'基准值={base_output:.4f}')
    
    ax.fill_between(param_values, base_output, outputs,
                     alpha=0.1, color=COLORS["blue"])
    
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
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
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(data[0]))
    
    for i, (d, label) in enumerate(zip(data, labels)):
        ax.plot(x, d, color=COLOR_LIST[i % len(COLOR_LIST)],
                linewidth=2, linestyle='-', label=label)
    
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if style == "academic":
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


# ═══════════════════════════════════════════════
# 帕累托前沿图
# ═══════════════════════════════════════════════

def plot_pareto_front(
    pareto_front: List[Tuple],
    all_solutions: Optional[List[Tuple]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Pareto 前沿",
    xlabel: str = "目标 1",
    ylabel: str = "目标 2",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    双目标 Pareto 前沿图。
    """
    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    front = np.array(pareto_front)
    
    # 所有解
    if all_solutions:
        all_sol = np.array(all_solutions)
        ax.scatter(all_sol[:, 0], all_sol[:, 1], c=COLORS["gray"],
                   alpha=0.3, s=20, label='所有解')
    
    # Pareto 前沿
    sorted_idx = np.argsort(front[:, 0])
    front_sorted = front[sorted_idx]
    
    ax.plot(front_sorted[:, 0], front_sorted[:, 1],
            color=COLORS["red"], linewidth=2, label='Pareto 前沿')
    ax.scatter(front[:, 0], front[:, 1], c=COLORS["red"], s=50, zorder=5)
    
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


# ═══════════════════════════════════════════════
# 子图模板
# ═══════════════════════════════════════════════

def subplots_grid(
    n_plots: int,
    n_cols: int = 2,
    figsize: Tuple = (14, 10),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    创建子图网格。
    
    Returns
    -------
    fig, axes
    """
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    return fig, axes


# ═══════════════════════════════════════════════
# 论文插图助手
# ═══════════════════════════════════════════════

def figure_caption(caption: str, label: str = "") -> str:
    """
    生成 LaTeX 图表标题。
    
    用法:
        # 在论文中使用
        print(figure_caption("模型A与模型B的预测结果对比", "fig:comparison"))
        → \\begin{figure}[htbp]
          \\centering
          \\includegraphics[width=0.8\\textwidth]{figures/comparison.png}
          \\caption{模型A与模型B的预测结果对比}
          \\label{fig:comparison}
          \\end{figure}
    """
    tex = r"\begin{figure}[htbp]" + "\n"
    tex += r"  \centering" + "\n"
    tex += r"  \includegraphics[width=0.8\textwidth]{figures/comparison.png}" + "\n"
    tex += f"  \\caption{{{caption}}}" + "\n"
    if label:
        tex += f"  \\label{{{label}}}" + "\n"
    tex += r"\end{figure}" + "\n"
    return tex
