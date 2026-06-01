"""
mm_eval.py — 评价/决策模型工具集

AHP层次分析、熵权法、TOPSIS、模糊综合评价、灰色关联度分析、CRITIC法
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════
# AHP 层次分析法
# ═══════════════════════════════════════════════

# 随机一致性指标 RI
_RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
       8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54, 13: 1.56,
       14: 1.58, 15: 1.59}


def ahp_consistency(matrix: np.ndarray) -> Dict:
    """
    判断矩阵一致性检验。
    
    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        判断矩阵
    
    Returns
    -------
    dict
        {"lambda_max": 最大特征值, "CI": 一致性指标,
         "RI": 随机一致性指标, "CR": 一致性比率,
         "is_consistent": 是否通过检验(CR<0.1),
         "weight": 归一化特征向量}
    """
    n = matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(matrix)
    lambda_max = np.max(np.real(eigvals))
    
    # 特征向量归一化
    weight = np.real(eigvecs[:, np.argmax(eigvals)])
    weight = weight / np.sum(weight)
    
    CI = (lambda_max - n) / (n - 1)
    RI = _RI.get(n, 1.5)
    CR = CI / RI if RI > 0 else 0
    
    return {
        "lambda_max": round(lambda_max, 4),
        "CI": round(CI, 4),
        "RI": RI,
        "CR": round(CR, 4),
        "is_consistent": CR < 0.10,
        "weight": np.round(weight, 4),
    }


def ahp_weight(matrix: np.ndarray) -> np.ndarray:
    """
    计算 AHP 权重（含一致性检验，若不通过给出警告）。
    
    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        判断矩阵（1-9标度）
    
    Returns
    -------
    np.ndarray
        归一化权重向量
    """
    result = ahp_consistency(matrix)
    if not result["is_consistent"]:
        import warnings
        warnings.warn(f"一致性检验未通过 (CR={result['CR']:.4f} ≥ 0.1)")
    return result["weight"]


def ahp_scale_matrix(comparisons: List[Tuple[int, int, float]], n: int) -> np.ndarray:
    """
    根据部分比较结果构造完整判断矩阵。
    
    Parameters
    ----------
    comparisons : list of (i, j, value)
        i 比 j 的重要程度（1-9标度）
        如 [(0, 1, 3)] 表示 指标0比指标1稍微重要(3)
    n : int
        指标数量
    
    Returns
    -------
    np.ndarray
        完整判断矩阵
    """
    matrix = np.ones((n, n))
    for i, j, v in comparisons:
        matrix[i, j] = v
        matrix[j, i] = 1 / v
    return matrix


def ahp_consistency_auto_fix(matrix: np.ndarray, max_iter: int = 100) -> Dict:
    """
    自动修正不满足一致性的判断矩阵。
    通过最小扰动使 CR < 0.1。
    """
    result = ahp_consistency(matrix)
    if result["is_consistent"]:
        return result
    
    # 逐步调整
    n = matrix.shape[0]
    best_matrix = matrix.copy()
    best_cr = result["CR"]
    best_weight = result["weight"]
    
    for _ in range(max_iter):
        # 随机扰动
        perturbed = matrix.copy()
        for i in range(n):
            for j in range(i+1, n):
                if np.random.rand() < 0.3:  # 30%概率扰动
                    factor = np.random.uniform(0.5, 2.0)
                    new_val = matrix[i, j] * factor
                    new_val = max(1/9, min(9, new_val))  # 保持1-9标度
                    perturbed[i, j] = new_val
                    perturbed[j, i] = 1 / new_val
        
        result = ahp_consistency(perturbed)
        if result["CR"] < best_cr:
            best_cr = result["CR"]
            best_matrix = perturbed.copy()
            best_weight = result["weight"]
            if result["is_consistent"]:
                break
    
    return {
        "lambda_max": None,
        "CI": None,
        "RI": _RI.get(n, 1.5),
        "CR": round(best_cr, 4),
        "is_consistent": best_cr < 0.10,
        "weight": best_weight,
        "note": "auto_fixed" if best_cr < 0.10 else f"best_effort (CR={best_cr:.4f})",
    }


# ═══════════════════════════════════════════════
# 熵权法
# ═══════════════════════════════════════════════

def entropy_weight(data: np.ndarray) -> np.ndarray:
    """
    熵权法计算客观权重。
    
    原理：数据变异程度越大 → 熵值越小 → 权重越大。
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        每个指标已经正向化且非负。
    
    Returns
    -------
    np.ndarray, shape (n_features,)
        权重向量，和为1
    """
    # 归一化
    data = np.array(data, dtype=float)
    data_sum = data.sum(axis=0)
    data_sum[data_sum == 0] = 1
    p = data / data_sum
    
    # 熵值
    n = data.shape[0]
    k = 1 / np.log(n) if n > 1 else 1
    e = -k * np.sum(p * np.log(p + 1e-10), axis=0)
    
    # 权重
    d = 1 - e
    return d / np.sum(d)


# ═══════════════════════════════════════════════
# TOPSIS
# ═══════════════════════════════════════════════

def topsis(
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    positive: Optional[List[int]] = None,
) -> Dict:
    """
    TOPSIS 综合评价。
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        原始数据（未归一化，各指标方向可不同）
    weights : np.ndarray, optional
        各指标权重，默认等权
    positive : list of int
        正向指标列索引，未列出则为负向指标
    
    Returns
    -------
    dict
        {"score": 综合得分, "rank": 排名, "d_best": 到正理想解距离,
         "d_worst": 到负理想解距离, "best": 正理想解, "worst": 负理想解}
    """
    data = np.array(data, dtype=float)
    n, m = data.shape
    
    if weights is None:
        weights = np.ones(m) / m
    
    if positive is None:
        positive = list(range(m))
    
    # 1. 正向化（负向指标取倒数）
    data_norm = data.copy()
    for j in range(m):
        if j not in positive:
            data_norm[:, j] = 1 / (data[:, j] + 1e-10)
    
    # 2. 归一化（向量归一化）
    norm = np.sqrt((data_norm ** 2).sum(axis=0))
    norm[norm == 0] = 1
    data_norm = data_norm / norm
    
    # 3. 加权
    weighted = data_norm * weights
    
    # 4. 理想解
    best = weighted.max(axis=0)
    worst = weighted.min(axis=0)
    
    # 5. 距离
    d_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))
    
    # 6. 得分
    score = d_worst / (d_best + d_worst + 1e-10)
    rank = np.argsort(-score)
    
    # 归一化到 [0, 1] 方便比较
    score_norm = (score - score.min()) / (score.max() - score.min() + 1e-10)
    
    return {
        "score": np.round(score_norm, 4),
        "rank": rank + 1,
        "d_best": np.round(d_best, 4),
        "d_worst": np.round(d_worst, 4),
        "best": best,
        "worst": worst,
    }


# ═══════════════════════════════════════════════
# 模糊综合评价
# ═══════════════════════════════════════════════

def fuzzy_eval(
    weights: np.ndarray,
    relation_matrix: np.ndarray,
    method: str = "maxmin",
) -> np.ndarray:
    """
    模糊综合评价。
    
    Parameters
    ----------
    weights : np.ndarray, shape (n_factors,)
        各因素的权重
    relation_matrix : np.ndarray, shape (n_factors, n_grades)
        隶属度矩阵，每行表示一个因素对各评价等级的隶属度
    method : str
        "maxmin" — 主因素突出型 M(∧,∨)
        "maxmul" — 主因素突出型 M(·,∨)  
        "weighted" — 加权平均型 M(·,⊕)
    
    Returns
    -------
    np.ndarray, shape (n_grades,)
        综合评价向量
    """
    w = np.array(weights)
    R = np.array(relation_matrix)
    
    if method == "maxmin":
        # M(∧,∨): 先取小再取大
        B = np.max(np.minimum(w[:, None], R), axis=0)
    elif method == "maxmul":
        # M(·,∨): 先乘再取大
        B = np.max(w[:, None] * R, axis=0)
    elif method == "weighted":
        # M(·,⊕): 加权平均
        B = w @ R
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 归一化
    B_sum = B.sum()
    B = B / B_sum if B_sum > 0 else B
    
    return B


def fuzzy_membership_triangular(
    x: np.ndarray,
    a: float, b: float, c: float,
) -> np.ndarray:
    """
    三角形隶属度函数。
    
    适用于"大约为 b"的模糊概念。
    """
    result = np.zeros_like(x, dtype=float)
    mask_left = (a <= x) & (x < b)
    mask_right = (b <= x) & (x <= c)
    result[mask_left] = (x[mask_left] - a) / (b - a)
    result[mask_right] = (c - x[mask_right]) / (c - b)
    return np.clip(result, 0, 1)


def fuzzy_membership_trapezoid(
    x: np.ndarray,
    a: float, b: float, c: float, d: float,
) -> np.ndarray:
    """
    梯形隶属度函数。
    
    适用于"在 [b, c] 之间"的模糊概念。
    """
    result = np.zeros_like(x, dtype=float)
    result[(b <= x) & (x <= c)] = 1
    mask_left = (a <= x) & (x < b)
    mask_right = (c < x) & (x <= d)
    result[mask_left] = (x[mask_left] - a) / (b - a)
    result[mask_right] = (d - x[mask_right]) / (d - c)
    return np.clip(result, 0, 1)


# ═══════════════════════════════════════════════
# 灰色关联度分析
# ═══════════════════════════════════════════════

def grey_relation_degree(
    reference: np.ndarray,
    compare: np.ndarray,
    rho: float = 0.5,
) -> np.ndarray:
    """
    灰色关联度分析。
    
    计算多个比较序列与参考序列的关联度。
    
    Parameters
    ----------
    reference : np.ndarray, shape (n_points,)
        参考序列（母序列）
    compare : np.ndarray, shape (n_series, n_points)
        比较序列（子序列）
    rho : float
        分辨系数，默认 0.5
    
    Returns
    -------
    np.ndarray, shape (n_series,)
        每个比较序列的关联度
    """
    compare = np.array(compare, dtype=float)
    reference = np.array(reference, dtype=float)
    
    # 归一化（均值法）
    ref_mean = reference.mean()
    ref_norm = reference / ref_mean if ref_mean != 0 else reference
    comp_mean = compare.mean(axis=1, keepdims=True)
    comp_norm = compare / (comp_mean + 1e-10)
    
    # 计算差序列
    diff = np.abs(comp_norm - ref_norm)
    
    # 关联系数
    delta_min = diff.min()
    delta_max = diff.max()
    gamma = (delta_min + rho * delta_max) / (diff + rho * delta_max)
    
    # 关联度
    degree = gamma.mean(axis=1)
    
    return np.round(degree, 4)


# ═══════════════════════════════════════════════
# CRITIC 法
# ═══════════════════════════════════════════════

def critic_weight(data: np.ndarray) -> np.ndarray:
    """
    CRITIC 法计算客观权重。
    
    同时考虑指标内变异程度（标准差）和指标间冲突性（相关系数）。
    比熵权法更全面，因为还考虑了指标间的相关性。
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    
    Returns
    -------
    np.ndarray
        权重向量
    """
    data = np.array(data, dtype=float)
    n, m = data.shape
    
    # 1. 归一化
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
    
    # 2. 标准差（变异程度）
    std = data_norm.std(axis=0)
    
    # 3. 相关系数矩阵（冲突性）
    corr = np.corrcoef(data_norm.T)
    conflict = 1 - np.abs(corr).sum(axis=0) / (m - 1)
    
    # 4. 信息量
    info = std * conflict
    
    # 5. 权重
    return info / info.sum()


# ═══════════════════════════════════════════════
# 组合赋权
# ═══════════════════════════════════════════════

def combined_weight(
    subjective_weight: np.ndarray,
    objective_weight: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    主观+客观组合赋权。
    
    Parameters
    ----------
    subjective_weight : np.ndarray
        主观权重（如 AHP）
    objective_weight : np.ndarray
        客观权重（如熵权法/CRITIC）
    alpha : float
        主观权重占比，0-1
    
    Returns
    -------
    np.ndarray
        组合权重
    """
    w = alpha * subjective_weight + (1 - alpha) * objective_weight
    return w / w.sum()
