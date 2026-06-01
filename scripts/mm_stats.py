"""
mm_stats.py — 统计/数据挖掘模型工具集

聚类、分类、降维、假设检验、相关性分析。
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from scipy import stats as sp_stats
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ═══════════════════════════════════════════════
# 描述统计
# ═══════════════════════════════════════════════

def descriptive_stats(data: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    描述性统计报告。

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features) or (n_samples,)

    Returns
    -------
    dict
        每列的均值、中位数、标准差、偏度、峰度、四分位数
    """
    data = np.array(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    result: Dict[str, Dict[str, float]] = {}
    for i in range(data.shape[1]):
        col = data[~np.isnan(data[:, i]), i]
        if len(col) == 0:
            continue
        result[f"feature_{i}"] = {
            "mean": round(float(np.mean(col)), 4),
            "median": round(float(np.median(col)), 4),
            "std": round(float(np.std(col)), 4),
            "min": round(float(np.min(col)), 4),
            "max": round(float(np.max(col)), 4),
            "skew": round(float(sp_stats.skew(col)), 4),
            "kurtosis": round(float(sp_stats.kurtosis(col)), 4),
            "q25": round(float(np.percentile(col, 25)), 4),
            "q75": round(float(np.percentile(col, 75)), 4),
            "n": len(col),
        }
    return result


# ═══════════════════════════════════════════════
# 相关性分析
# ═══════════════════════════════════════════════

def pearson_corr(X: np.ndarray) -> Dict[str, Any]:
    """
    Pearson 相关系数矩阵（线性相关）。
    """
    if X.shape[1] == 2:
        r, p = sp_stats.pearsonr(X[:, 0], X[:, 1])
        return {
            "corr": r,
            "p_value": p,
            "interpretation": f"r={r:.4f}, p={p:.4f} — {'显著' if p < 0.05 else '不显著'}",
        }
    else:
        corr_matrix = np.corrcoef(X.T)
        n = X.shape[1]
        p_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                _, p = sp_stats.pearsonr(X[:, i], X[:, j])
                p_matrix[i, j] = p_matrix[j, i] = p
        return {
            "corr_matrix": np.round(corr_matrix, 4),
            "p_matrix": np.round(p_matrix, 4),
        }


def spearman_corr(X: np.ndarray) -> Dict[str, Any]:
    """
    Spearman 秩相关系数（单调相关，不要求正态分布）。
    """
    corr, p_vals = sp_stats.spearmanr(X)
    return {
        "corr_matrix": np.round(corr, 4),
        "p_matrix": np.round(p_vals, 4),
    }


# ═══════════════════════════════════════════════
# 假设检验
# ═══════════════════════════════════════════════

def t_test_one_sample(data: np.ndarray, mu: float = 0.0) -> Dict[str, Any]:
    """单样本 t 检验"""
    t_stat, p_val = sp_stats.ttest_1samp(data, mu)
    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
        "conclusion": "拒绝原假设" if p_val < 0.05 else "不能拒绝原假设",
    }


def t_test_two_sample(
    data1: np.ndarray,
    data2: np.ndarray,
    equal_var: bool = True,
) -> Dict[str, Any]:
    """独立样本 t 检验"""
    t_stat, p_val = sp_stats.ttest_ind(data1, data2, equal_var=equal_var)
    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
        "conclusion": "两组均值有显著差异" if p_val < 0.05 else "两组均值无显著差异",
    }


def paired_t_test(data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
    """配对 t 检验"""
    t_stat, p_val = sp_stats.ttest_rel(data1, data2)
    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
    }


def chi_square_test(observed: np.ndarray) -> Dict[str, Any]:
    """
    卡方检验（独立性检验）。
    """
    chi2, p_val, dof, expected = sp_stats.chi2_contingency(observed)
    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_val), 4),
        "dof": int(dof),
        "significant": bool(p_val < 0.05),
    }


def normality_test(data: np.ndarray) -> Dict[str, Any]:
    """
    正态性检验（Shapiro-Wilk / D'Agostino-Pearson 综合判断）。
    """
    data = data.flatten()
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return {"is_normal": False, "note": "样本量不足"}

    if len(data) < 5000:
        stat, p = sp_stats.shapiro(data)
    else:
        stat, p = sp_stats.normaltest(data)

    return {
        "stat": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "is_normal": bool(p > 0.05),
        "conclusion": "数据服从正态分布" if p > 0.05 else "数据不服从正态分布",
    }


# ═══════════════════════════════════════════════
# 聚类分析
# ═══════════════════════════════════════════════

def kmeans_clustering(
    data: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    K-Means 聚类。

    Returns
    -------
    dict
        {"labels", "centers", "inertia", "silhouette_score"}
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto", **kwargs)
    labels = model.fit_predict(data)
    sil = silhouette_score(data, labels) if n_clusters > 1 else 0.0
    return {
        "labels": labels,
        "centers": model.cluster_centers_,
        "inertia": float(model.inertia_),
        "silhouette_score": round(float(sil), 4),
        "model": model,
    }


def elbow_method(data: np.ndarray, max_k: int = 10) -> Dict[str, Any]:
    """
    肘部法则确定最佳 k 值。

    Returns
    -------
    dict
        {"k_values", "inertias", "silhouettes", "best_k_by_silhouette"}
    """
    inertias: List[float] = []
    silhouettes: List[float] = []
    k_values = list(range(2, min(max_k + 1, len(data))))

    for k in k_values:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(data)
        inertias.append(float(model.inertia_))
        sil = silhouette_score(data, labels) if k > 1 else 0.0
        silhouettes.append(float(sil))

    best_k = int(k_values[np.argmax(silhouettes)]) if silhouettes else 2
    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k_by_silhouette": best_k,
    }


def dbscan_clustering(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[str, Any]:
    """
    DBSCAN 密度聚类（不需要预先指定簇数，能识别噪声点）。
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int(np.sum(labels == -1))

    sil = 0.0
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            sil = float(silhouette_score(data[mask], labels[mask]))

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette_score": round(sil, 4),
        "model": model,
    }


def hierarchical_clustering(
    data: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "ward",
) -> Dict[str, Any]:
    """
    层次聚类。

    Parameters
    ----------
    linkage : str
        "ward" — 离差平方和（默认）
        "complete" — 最长距离法
        "average" — 平均距离法
        "single" — 最短距离法
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(data)
    sil = silhouette_score(data, labels) if n_clusters > 1 else 0.0
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "silhouette_score": round(float(sil), 4),
        "model": model,
    }


# ═══════════════════════════════════════════════
# 降维
# ═══════════════════════════════════════════════

def pca_analysis(
    data: np.ndarray,
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """
    主成分分析 (PCA)。

    Returns
    -------
    dict
        {"transformed", "explained_variance_ratio", "cumulative_ratio",
         "loadings", "n_components", "suggested_components"}
    """
    if n_components is None:
        n_components = min(data.shape)

    model = PCA(n_components=n_components)
    transformed = model.fit_transform(data)

    explained = model.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    suggested = int(np.argmax(cumulative > 0.85) + 1) if np.any(cumulative > 0.85) else n_components

    return {
        "transformed": transformed,
        "explained_variance_ratio": np.round(explained, 4),
        "cumulative_ratio": np.round(cumulative, 4),
        "loadings": np.round(model.components_.T, 4),
        "n_components": n_components,
        "suggested_components": suggested,
        "model": model,
    }


# ═══════════════════════════════════════════════
# 方差分析
# ═══════════════════════════════════════════════

def anova_one_way(groups: List[np.ndarray]) -> Dict[str, Any]:
    """
    单因素方差分析。

    Parameters
    ----------
    groups : list of arrays
        多组数据

    Returns
    -------
    dict
        {"F_stat", "p_value", "significant", "conclusion"}
    """
    F, p_val = sp_stats.f_oneway(*groups)
    return {
        "F_stat": round(float(F), 4),
        "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
        "conclusion": "组间存在显著差异" if p_val < 0.05 else "组间无显著差异",
    }
