"""
mm_utils.py — 数学建模通用工具集

数据预处理、灵敏度分析、误差指标、参数拟合等常用函数。
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings


# ═══════════════════════════════════════════════
# 数据预处理
# ═══════════════════════════════════════════════

def fill_missing(data: np.ndarray, method: str = "linear") -> np.ndarray:
    """
    填充缺失值。
    
    Parameters
    ----------
    data : np.ndarray
        含 NaN 的数组
    method : str
        "linear" — 线性插值
        "mean" — 均值填充
        "median" — 中位数填充
        "zero" — 填 0
        "forward" — 前向填充
        "backward" — 后向填充
    
    Returns
    -------
    np.ndarray
        填充后的数组
    """
    data = data.copy()
    if method == "linear":
        x = np.arange(len(data))
        mask = ~np.isnan(data)
        data = np.interp(x, x[mask], data[mask])
    elif method == "mean":
        data[np.isnan(data)] = np.nanmean(data)
    elif method == "median":
        data[np.isnan(data)] = np.nanmedian(data)
    elif method == "zero":
        data[np.isnan(data)] = 0
    elif method == "forward":
        for i in range(1, len(data)):
            if np.isnan(data[i]):
                data[i] = data[i-1]
    elif method == "backward":
        for i in range(len(data)-2, -1, -1):
            if np.isnan(data[i]):
                data[i] = data[i+1]
    else:
        raise ValueError(f"Unknown method: {method}")
    return data


def normalize(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    数据归一化。
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    method : str
        "minmax" — 缩放到 [0, 1]
        "zscore" — 标准化为均0方差1
        "max" — 除以最大值
        "sum" — 归一化使和为1
    
    Returns
    -------
    np.ndarray
    """
    data = data.astype(float)
    if method == "minmax":
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        return (data - min_vals) / range_vals
    elif method == "zscore":
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)
        sigma[sigma == 0] = 1
        return (data - mu) / sigma
    elif method == "max":
        max_vals = np.abs(data).max(axis=0)
        max_vals[max_vals == 0] = 1
        return data / max_vals
    elif method == "sum":
        sums = data.sum(axis=0)
        sums[sums == 0] = 1
        return data / sums
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    用 IQR 方法检测异常值。
    
    Parameters
    ----------
    data : np.ndarray
    k : float
        IQR 倍数，默认 1.5（标准），3.0 为极端
    
    Returns
    -------
    np.ndarray
        布尔数组，True 表示异常值
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (data < lower) | (data > upper)


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    用 Z-Score 方法检测异常值。
    """
    z = np.abs((data - np.mean(data)) / np.std(data))
    return z > threshold


def cap_outliers(data: np.ndarray, method: str = "iqr", k: float = 1.5) -> np.ndarray:
    """
    将异常值替换为边界值（Winsorize）。
    """
    data = data.copy()
    if method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
    else:  # zscore
        mean, std = np.mean(data), np.std(data)
        lower = mean - k * std
        upper = mean + k * std
    data[data < lower] = lower
    data[data > upper] = upper
    return data


# ═══════════════════════════════════════════════
# 灵敏度分析
# ═══════════════════════════════════════════════

def sensitivity_analysis(
    model_func: Callable,
    base_params: Dict[str, float],
    param_to_vary: str,
    range_pct: float = 0.2,
    steps: int = 10,
    **kwargs,
) -> List[Dict]:
    """
    单参数灵敏度分析。
    
    Parameters
    ----------
    model_func : Callable
        模型函数，接受 params dict 返回数值结果
    base_params : dict
        基准参数
    param_to_vary : str
        要变化的参数名
    range_pct : float
        变化范围（比例）
    steps : int
        采样步数
    
    Returns
    -------
    List[Dict]
        [{"param_value": ..., "output": ...}, ...]
    """
    base_value = base_params[param_to_vary]
    param_values = np.linspace(
        base_value * (1 - range_pct),
        base_value * (1 + range_pct),
        steps
    )
    results = []
    for val in param_values:
        params = base_params.copy()
        params[param_to_vary] = val
        try:
            output = model_func(params, **kwargs)
        except Exception as e:
            output = np.nan
        results.append({"param_value": val, "output": output})
    return results


def two_factor_sensitivity(
    model_func: Callable,
    base_params: Dict[str, float],
    params: List[str],
    range_pct: float = 0.2,
    steps: int = 10,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    双参数灵敏度分析（用于画热力图）。
    
    Returns
    -------
    X : np.ndarray, shape (steps, steps)
        参数1值网格
    Y : np.ndarray, shape (steps, steps)
        参数2值网格
    Z : np.ndarray, shape (steps, steps)
        模型输出值网格
    """
    p1_name, p2_name = params
    p1_base = base_params[p1_name]
    p2_base = base_params[p2_name]
    
    p1_vals = np.linspace(p1_base*(1-range_pct), p1_base*(1+range_pct), steps)
    p2_vals = np.linspace(p2_base*(1-range_pct), p2_base*(1+range_pct), steps)
    
    X, Y = np.meshgrid(p1_vals, p2_vals)
    Z = np.zeros_like(X)
    
    for i in range(steps):
        for j in range(steps):
            params = base_params.copy()
            params[p1_name] = X[i, j]
            params[p2_name] = Y[i, j]
            try:
                Z[i, j] = model_func(params, **kwargs)
            except Exception:
                Z[i, j] = np.nan
    
    return X, Y, Z


def monte_carlo_sensitivity(
    model_func: Callable,
    param_distributions: Dict[str, Tuple[str, Tuple]],
    n_samples: int = 1000,
    **kwargs,
) -> pd.DataFrame:
    """
    蒙特卡洛灵敏度分析。
    
    Parameters
    ----------
    param_distributions : dict
        {name: ("uniform", (low, high))} 或 {name: ("normal", (mu, sigma))}
    n_samples : int
    
    Returns
    -------
    pd.DataFrame with columns: param values + output
    """
    samples = {}
    for name, (dist, params) in param_distributions.items():
        if dist == "uniform":
            samples[name] = np.random.uniform(*params, n_samples)
        elif dist == "normal":
            samples[name] = np.random.normal(*params, n_samples)
        else:
            raise ValueError(f"Unknown distribution: {dist}")
    
    outputs = []
    for i in range(n_samples):
        params = {k: v[i] for k, v in samples.items()}
        try:
            outputs.append(model_func(params, **kwargs))
        except Exception:
            outputs.append(np.nan)
    
    df = pd.DataFrame(samples)
    df["output"] = outputs
    return df


# ═══════════════════════════════════════════════
# 误差指标
# ═══════════════════════════════════════════════

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对百分比误差 (%)"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """对称平均绝对百分比误差 (%)"""
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator[denominator == 0] = 1
    return np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² 决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """调整 R²"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def error_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """综合误差报告"""
    return {
        "MSE": round(mse(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "MAPE(%)": round(mape(y_true, y_pred), 2),
        "SMAPE(%)": round(smape(y_true, y_pred), 2),
        "R²": round(r2_score(y_true, y_pred), 4),
    }


# ═══════════════════════════════════════════════
# 参数辨识（最小二乘拟合）
# ═══════════════════════════════════════════════

def curve_fit_report(x_data, y_data, func, p0=None):
    """
    曲线拟合并返回拟合报告。
    
    Parameters
    ----------
    x_data, y_data : array-like
    func : callable
        模型函数 func(x, *params)
    p0 : list, optional
        初始参数猜测
    
    Returns
    -------
    dict
        拟合参数、协方差、R²、RMSE
    """
    from scipy.optimize import curve_fit
    
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    y_pred = func(x_data, *popt)
    
    return {
        "params": popt,
        "param_errors": perr,
        "covariance": pcov,
        "R²": round(r2_score(y_data, y_pred), 4),
        "RMSE": round(rmse(y_data, y_pred), 4),
        "y_pred": y_pred,
    }


# ═══════════════════════════════════════════════
# 数学工具
# ═══════════════════════════════════════════════

def cumulative_sum(data: np.ndarray) -> np.ndarray:
    """累加生成（用于灰色预测等）"""
    return np.cumsum(data)


def inverse_cumulative_sum(data: np.ndarray) -> np.ndarray:
    """累减生成"""
    result = data.copy()
    result[1:] = data[1:] - data[:-1]
    return result


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE (百分比)"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
