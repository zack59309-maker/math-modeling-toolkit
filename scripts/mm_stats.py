"""
mm_stats.py — 统计/数据挖掘模型工具集

聚类、分类、降维、假设检验、相关性分析。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ═══════════════════════════════════════════════
# 描述统计
# ═══════════════════════════════════════════════

def descriptive_stats(data: np.ndarray) -> Dict:
    """
    描述性统计报告。
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features) or (n_samples,)
    
    Returns
    -------
    dict
        均值、中位数、标准差、偏度、峰度、四分位数
    """
    data = np.array(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    result = {}
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
            "skew": round(float(stats.skew(col)), 4),
            "kurtosis": round(float(stats.kurtosis(col)), 4),
            "q25": round(float(np.percentile(col, 25)), 4),
            "q75": round(float(np.percentile(col, 75)), 4),
            "n": len(col),
        }
    return result


# ═══════════════════════════════════════════════
# 相关性分析
# ═══════════════════════════════════════════════

def pearson_corr(X: np.ndarray) -> Dict:
    """
    Pearson 相关系数矩阵（线性相关）。
    
    Returns
    -------
    dict
        {"corr_matrix": ..., "p_matrix": ...}
    """
    corr, p_vals = stats.pearsonr(X.T) if X.shape[1] == 2 else (
        None, None
    )
    if X.shape[1] == 2:
        return {
            "corr": corr,
            "p_value": p_vals,
            "interpretation": (
                f"r={corr:.4f}, p={p_vals:.4f} — "
                f"{'显著' if p_vals < 0.05 else '不显著'}"
            )
        }
    else:
        corr_matrix = np.corrcoef(X.T)
        n = X.shape[1]
        p_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                _, p = stats.pearsonr(X[:, i], X[:, j])
                p_matrix[i, j] = p_matrix[j, i] = p
        return {"corr_matrix": np.round(corr_matrix, 4),
                "p_matrix": np.round(p_matrix, 4)}


def spearman_corr(X: np.ndarray) -> Dict:
    """
    Spearman 秩相关系数（单调相关，不要求正态分布）。
    """
    corr, p_vals = stats.spearmanr(X)
    return {"corr_matrix": np.round(corr, 4),
            "p_matrix": np.round(p_vals, 4)}


# ═══════════════════════════════════════════════
# 假设检验
# ═══════════════════════════════════════════════

def t_test_one_sample(data: np.ndarray, mu: float = 0) -> Dict:
    """单样本 t 检验"""
    t_stat, p_val = stats.ttest_1samp(data, mu)
    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "significant": p_val < 0.05,
        "conclusion": "拒绝原假设" if p_val < 0.05 else "不能拒绝原假设",
    }


def t_test_two_sample(data1: np.ndarray, data2: np.ndarray,
                      equal_var: bool = True) -> Dict:
    """独立样本 t 检验"""
    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=equal_var)
    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "significant": p_val < 0.05,
        "conclusion": "两组均值有显著差异" if p_val < 0.05 else "两组均值无显著差异",
    }


def paired_t_test(data1: np.ndarray, data2: np.ndarray) -> Dict:
    """配对 t 检验"""
    t_stat, p_val = stats.ttest_rel(data1, data2)
    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "significant": p_val < 0.05,
    }


def chi_square_test(observed: np.ndarray) -> Dict:
    """
    卡方检验（独立性检验）。
    
    Parameters
    ----------
    observed : np.ndarray
        观测频数矩阵
    """
    chi2, p_val, dof, expected = stats.chi2_contingency(observed)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_val, 4),
        "dof": dof,
        "significant": p_val < 0.05,
    }


def normality_test(data: np.ndarray) -> Dict:
    """
    正态性检验（Shapiro-Wilk + D'Agostino-Pearson 综合判断）。
    """
    data = data.flatten()
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return {"is_normal": False, "note": "样本量不足"}
    
    # Shapiro-Wilk (n < 5000 时准确)
    if len(data) < 5000:
        stat, p = stats.shapiro(data)
    else:
        stat, p = stats.normaltest(data)  # D'Agostino-Pearson
    
    return {
        "stat": round(stat, 4),
        "p_value": round(p, 4),
        "is_normal": p > 0.05,
        "conclusion": "数据服从正态分布" if p > 0.05 else "数据不服从正态分布",
    }


# ═══════════════════════════════════════════════
# 聚类分析
# ═══════════════════════════════════════════════

def kmeans_clustering(
    data: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    **kwargs,
) -> Dict:
    """
    K-Means 聚类。
    
    Returns
    -------
    dict
        {"labels": 聚类标签, "centers": 聚类中心,
         "inertia": 总内平方和, "silhouette": 轮廓系数}
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state,
                   n_init='auto', **kwargs)
    labels = model.fit_predict(data)
    
    silhouette = silhouette_score(data, labels) if n_clusters > 1 else 0
    
    return {
        "labels": labels,
        "centers": model.cluster_centers_,
        "inertia": model.inertia_,
        "silhouette_score": round(silhouette, 4),
        "model": model,
    }


def elbow_method(data: np.ndarray, max_k: int = 10) -> Dict:
    """
    肘部法则确定最佳 k 值。
    
    Returns
    -------
    dict
        {"k_values": [...], "inertias": [...], "silhouettes": [...]}
    """
    inertias = []
    silhouettes = []
    k_values = list(range(2, min(max_k + 1, len(data))))
    
    for k in k_values:
        model = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = model.fit_predict(data)
        inertias.append(model.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(data, labels))
        else:
            silhouettes.append(0)
    
    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k_by_silhouette": k_values[np.argmax(silhouettes)],
    }


def dbscan_clustering(
    data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict:
    """
    DBSCAN 密度聚类（不需要预先指定簇数，能识别噪声点）。
    
    适用：形状不规则的簇、有噪声数据。
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    silhouette = silhouette_score(
        data[labels != -1], labels[labels != -1]
    ) if n_clusters > 1 else 0
    
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette_score": round(silhouette, 4),
        "model": model,
    }


def hierarchical_clustering(
    data: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "ward",
) -> Dict:
    """
    层次聚类。
    
    Parameters
    ----------
    linkage : str
        "ward" — 离差平方和（默认）
        "complete" — 最长距离
        "average" — 平均距离
        "single" — 最短距离
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage
    )
    labels = model.fit_predict(data)
    
    silhouette = silhouette_score(data, labels) if n_clusters > 1 else 0
    
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "silhouette_score": round(silhouette, 4),
        "model": model,
    }


# ═══════════════════════════════════════════════
# 降维
# ═══════════════════════════════════════════════

def pca_analysis(data: np.ndarray, n_components: Optional[int] = None) -> Dict:
    """
    主成分分析 (PCA)。
    
    Returns
    -------
    dict
        {"transformed": 降维后数据, "explained_variance_ratio": 方差贡献率,
         "cumulative_ratio": 累计方差贡献率, "loadings": 载荷矩阵,
         "n_components": 选择的主成分数}
    """
    if n_components is None:
        n_components = min(data.shape[0], data.shape[1])
    
    model = PCA(n_components=n_components)
    transformed = model.fit_transform(data)
    
    explained = model.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    
    # 建议保留的主成分数（累计方差 > 85%）
    suggested = np.argmax(cumulative > 0.85) + 1 if np.any(cumulative > 0.85) else n_components
    
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

def anova_one_way(groups: List[np.ndarray]) -> Dict:
    """
    单因素方差分析。
    
    Parameters
    ----------
    groups : list of arrays
        多组数据
    
    Returns
    -------
    dict
        {"F_stat": ..., "p_value": ..., "significant": bool}
    """
    F, p_val = stats.f_oneway(*groups)
    return {
        "F_stat": round(F, 4),
        "p_value": round(p_val, 4),
        "significant": p_val < 0.05,
        "conclusion": "组间存在显著差异" if p_val < 0.05 else "组间无显著差异",
    }
