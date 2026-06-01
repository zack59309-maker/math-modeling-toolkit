"""
mm_utils.py — 数学建模通用工具集

数据预处理、灵敏度分析、误差指标、参数拟合等常用函数。
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ═══════════════════════════════════════════════
# 数据预处理
# ═══════════════════════════════════════════════

def fill_missing(data: np.ndarray, method: str = "linear") -> np.ndarray:
    """
    填充缺失值。

    Parameters
    ----------
    data : np.ndarray
        含 NaN 的数组。支持 1D 和 2D (按列处理)。
    method : str
        "linear"   — 线性插值（1D only）
        "mean"     — 均值填充
        "median"   — 中位数填充
        "zero"     — 填 0
        "ffill"    — 前向填充（forward fill）
        "bfill"    — 后向填充（backward fill）

    Returns
    -------
    np.ndarray
        填充后的数组
    """
    data = data.copy()
    is_2d = data.ndim == 2

    if method in ("ffill", "bfill"):
        axis = 0 if is_2d else None
        return pd.DataFrame(data).fillna(method=method.replace("fill", "fill"), axis=axis).to_numpy(dtype=float)

    if method == "linear":
        if is_2d:
            for col in range(data.shape[1]):
                col_data = data[:, col]
                mask = ~np.isnan(col_data)
                if mask.any():
                    data[:, col] = np.interp(
                        np.arange(len(col_data)),
                        np.where(mask)[0],
                        col_data[mask]
                    )
        else:
            mask = ~np.isnan(data)
            if mask.any():
                data = np.interp(np.arange(len(data)), np.where(mask)[0], data[mask])
        return data

    fill_map = {
        "mean": lambda d: np.nanmean(d, axis=0 if is_2d else None),
        "median": lambda d: np.nanmedian(d, axis=0 if is_2d else None),
        "zero": lambda _: 0.0,
    }
    if method not in fill_map:
        raise ValueError(f"Unknown method: {method}")

    fill_val = fill_map[method](data)
    nan_mask = np.isnan(data)
    if is_2d:
        data[nan_mask] = np.take(fill_val, np.where(nan_mask)[1])
    else:
        data[nan_mask] = fill_val
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
        "max"    — 除以绝对值最大值
        "sum"    — 归一化使每列和为1

    Returns
    -------
    np.ndarray
    """
    data = data.astype(float)
    if method == "minmax":
        min_vals, max_vals = data.min(axis=0), data.max(axis=0)
        denominator = max_vals - min_vals
        denominator[denominator == 0] = 1.0
        return (data - min_vals) / denominator
    elif method == "zscore":
        mu, sigma = data.mean(axis=0), data.std(axis=0)
        sigma[sigma == 0] = 1.0
        return (data - mu) / sigma
    elif method == "max":
        max_vals = np.abs(data).max(axis=0)
        max_vals[max_vals == 0] = 1.0
        return data / max_vals
    elif method == "sum":
        sums = data.sum(axis=0)
        sums[sums == 0] = 1.0
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
    """用 Z-Score 方法检测异常值。"""
    z = np.abs((data - np.mean(data)) / (np.std(data) + 1e-10))
    return z > threshold


def cap_outliers(data: np.ndarray, method: str = "iqr", k: float = 1.5) -> np.ndarray:
    """
    将异常值替换为边界值（Winsorize）。
    """
    data = data.copy()
    if method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
    elif method == "zscore":
        mean, std = np.mean(data), np.std(data)
        lower, upper = mean - k * std, mean + k * std
    else:
        raise ValueError(f"Unknown method: {method}")
    np.clip(data, lower, upper, out=data)
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
    **kwargs: Any,
) -> List[Dict[str, float]]:
    """
    单参数灵敏度分析。

    Parameters
    ----------
    model_func : Callable[[Dict[str, float]], float]
        模型函数，接受 params dict 返回数值结果
    base_params : Dict[str, float]
        基准参数
    param_to_vary : str
        要变化的参数名
    range_pct : float
        变化范围（比例，默认 ±20%）
    steps : int
        采样步数（默认 10）

    Returns
    -------
    List[Dict[str, float]]
        [{"param_value": ..., "output": ...}, ...]
    """
    base_value = base_params[param_to_vary]
    param_values = np.linspace(
        base_value * (1 - range_pct),
        base_value * (1 + range_pct),
        steps,
    )
    results: List[Dict[str, float]] = []
    for val in param_values:
        params = base_params.copy()
        params[param_to_vary] = float(val)
        try:
            output = model_func(params, **kwargs)
        except Exception:
            output = np.nan
        results.append({"param_value": float(val), "output": float(output)})
    return results


def two_factor_sensitivity(
    model_func: Callable,
    base_params: Dict[str, float],
    params: List[str],
    range_pct: float = 0.2,
    steps: int = 10,
    **kwargs: Any,
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
    p1_base, p2_base = base_params[p1_name], base_params[p2_name]

    p1_vals = np.linspace(p1_base * (1 - range_pct), p1_base * (1 + range_pct), steps)
    p2_vals = np.linspace(p2_base * (1 - range_pct), p2_base * (1 + range_pct), steps)

    X, Y = np.meshgrid(p1_vals, p2_vals)
    Z = np.full_like(X, np.nan)

    for i in range(steps):
        for j in range(steps):
            params = base_params.copy()
            params[p1_name] = float(X[i, j])
            params[p2_name] = float(Y[i, j])
            try:
                Z[i, j] = model_func(params, **kwargs)
            except Exception:
                pass  # 保持 NaN
    return X, Y, Z


def monte_carlo_sensitivity(
    model_func: Callable,
    param_distributions: Dict[str, Tuple[str, Tuple[float, ...]]],
    n_samples: int = 1000,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    蒙特卡洛灵敏度分析。

    Parameters
    ----------
    param_distributions : Dict[str, Tuple[str, Tuple]]
        {name: ("uniform", (low, high))} 或 {name: ("normal", (mu, sigma))}
    n_samples : int

    Returns
    -------
    pd.DataFrame with columns: param values + output
    """
    rng = np.random.default_rng()
    samples: Dict[str, np.ndarray] = {}
    for name, (dist, params) in param_distributions.items():
        if dist == "uniform":
            samples[name] = rng.uniform(*params, n_samples)
        elif dist == "normal":
            samples[name] = rng.normal(*params, n_samples)
        else:
            raise ValueError(f"Unknown distribution: {dist}")

    outputs = np.full(n_samples, np.nan)
    for i in range(n_samples):
        params = {k: float(v[i]) for k, v in samples.items()}
        try:
            outputs[i] = model_func(params, **kwargs)
        except Exception:
            pass

    df = pd.DataFrame(samples)
    df["output"] = outputs
    return df


# ═══════════════════════════════════════════════
# 误差指标
# ═══════════════════════════════════════════════

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对百分比误差 (%)"""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """对称平均绝对百分比误差 (%)"""
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator[denominator == 0] = 1.0
    return float(np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² 决定系数"""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """调整 R²"""
    if n - p - 1 <= 0:
        return r2
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

def curve_fit_report(
    x_data: np.ndarray,
    y_data: np.ndarray,
    func: Callable,
    p0: Optional[List[float]] = None,
) -> Dict[str, Any]:
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
