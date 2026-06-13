"""
mm_predict.py — 预测模型工具集

回归、时间序列、灰色预测 GM(1,1)、指数平滑、BP神经网络
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score as sk_r2


# ═══════════════════════════════════════════════
# 灰色预测 GM(1,1)
# ═══════════════════════════════════════════════

def gm11(data: np.ndarray, forecast_steps: int = 5) -> Dict[str, Any]:
    """
    灰色预测 GM(1,1) 模型。

    适用场景：小样本（≥4 个数据点）、指数增长趋势的短期预测。

    Parameters
    ----------
    data : np.ndarray
        原始序列，至少 4 个数据点
    forecast_steps : int
        预测步数

    Returns
    -------
    dict
        {"pred": 预测值, "fitted": 拟合值, "a": 发展系数,
         "b": 灰色作用量, "residual": 残差, "relative_error": 相对误差}
    """
    n = len(data)
    if n < 4:
        raise ValueError(f"GM(1,1) 需要至少 4 个数据点，当前只有 {n}")
    x0 = np.array(data, dtype=float)

    # 1. 累加生成
    x1 = np.cumsum(x0)

    # 2. 紧邻均值序列
    z1 = (x1[:-1] + x1[1:]) / 2.0

    # 3. 最小二乘估计参数
    B = np.vstack([-z1, np.ones_like(z1)]).T
    Y = x0[1:]
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]

    # 4. 预测
    def _predict(k: int) -> float:
        if k == 0:
            return float(x0[0])
        return (1 - np.exp(a)) * (x0[0] - b / a) * np.exp(-a * k)

    fitted = np.array([_predict(i) for i in range(n)])
    pred = np.array([_predict(i) for i in range(n, n + forecast_steps)])

    # 残差
    residual = x0 - fitted
    relative_error = np.abs(residual / (x0 + 1e-10)) * 100

    return {
        "pred": pred,
        "fitted": fitted,
        "a": float(a),
        "b": float(b),
        "residual": residual,
        "relative_error": relative_error,
        "mape": float(np.mean(relative_error)),
    }


def gm11_improved(
    data: np.ndarray,
    forecast_steps: int = 5,
    residual_correct: bool = True,
) -> Dict[str, Any]:
    """
    改进灰色预测：残差修正 GM(1,1)。
    对残差再用 GM(1,1) 建模，叠加到原预测结果上。
    """
    base = gm11(data, forecast_steps)

    if not residual_correct:
        return base

    residuals = np.abs(base["residual"])
    if len(residuals) < 4 or np.std(residuals) < 1e-10:
        return base

    try:
        residual_model = gm11(residuals, forecast_steps)
        signs = np.sign(base["residual"])
        last_sign = float(np.sign(np.mean(signs[-3:]))) if len(signs) >= 3 else float(signs[-1])

        base["fitted"] = base["fitted"] + last_sign * residual_model["fitted"]
        base["pred"] = base["pred"] + last_sign * residual_model["pred"]
        base["note"] = "residual_corrected"
    except Exception:
        pass
    return base


def gm11_test_accuracy(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    检验灰色预测精度等级。

    Returns
    -------
    dict
        {"C": 后验差比, "p": 小误差概率, "grade": 等级, "detail": 详情}
    """
    residuals = result["residual"]
    s1 = float(np.std(result["fitted"]))
    s2 = float(np.std(residuals))
    c = s2 / s1 if s1 != 0 else 999.0

    s0 = 0.6745 * s1
    p = float(np.sum(np.abs(residuals - np.mean(residuals)) < s0) / len(residuals))

    if c < 0.35:
        grade_c = "好"
    elif c < 0.5:
        grade_c = "合格"
    elif c < 0.65:
        grade_c = "勉强"
    else:
        grade_c = "不合格"

    if p > 0.95:
        grade_p = "好"
    elif p > 0.8:
        grade_p = "合格"
    elif p > 0.7:
        grade_p = "勉强"
    else:
        grade_p = "不合格"

    grade = "合格" if c < 0.5 and p > 0.8 else "不合格"
    return {
        "C": round(c, 4),
        "p": round(p, 4),
        "grade": grade,
        "detail": f"C={c:.4f}, p={p:.4f} → {grade_c}(C) {grade_p}(p)",
    }


# ═══════════════════════════════════════════════
# 回归分析
# ═══════════════════════════════════════════════

def linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """
    多元线性回归。

    Returns
    -------
    dict
        {"coef": [...], "intercept": ..., "r2": ..., "y_pred": [...]}
    """
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)
    y_pred = model.predict(X)
    return {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r2": float(sk_r2(y, y_pred)),
        "y_pred": y_pred,
        "model": model,
    }


def polynomial_regression(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
) -> Dict[str, Any]:
    """
    多项式回归。
    """
    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_2d)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r2": float(sk_r2(y, y_pred)),
        "y_pred": y_pred,
        "poly_features": poly.get_feature_names_out().tolist(),
    }


def ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """岭回归（L2正则化）"""
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    y_pred = model.predict(X)
    return {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r2": float(sk_r2(y, y_pred)),
        "y_pred": y_pred,
    }


def lasso_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Lasso回归（L1正则化，特征选择）"""
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X, y)
    y_pred = model.predict(X)
    return {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r2": float(sk_r2(y, y_pred)),
        "y_pred": y_pred,
    }


# ═══════════════════════════════════════════════
# 指数平滑
# ═══════════════════════════════════════════════

def exponential_smoothing(
    data: np.ndarray,
    alpha: float = 0.3,
    forecast_steps: int = 5,
) -> Dict[str, Any]:
    """
    一次指数平滑。

    Parameters
    ----------
    alpha : float
        平滑系数 (0 < alpha < 1)，越大越重视近期数据
    """
    n = len(data)
    s = np.zeros(n + forecast_steps)
    s[0] = data[0]
    for t in range(1, n):
        s[t] = alpha * data[t] + (1 - alpha) * s[t - 1]
    s[n:] = s[n - 1]
    return {
        "fitted": s[:n],
        "pred": s[n:],
        "alpha": alpha,
    }


def holt_winters(
    data: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.1,
    period: int = 4,
    forecast_steps: int = 5,
) -> Dict[str, Any]:
    """
    Holt-Winters 三次指数平滑（含季节性）。

    Parameters
    ----------
    alpha : float
        水平平滑系数
    beta : float
        趋势平滑系数
    gamma : float
        季节性平滑系数
    period : int
        季节周期长度
    """
    n = len(data)
    if n < 2 * period:
        raise ValueError(f"Data length {n} < 2*period={2 * period}")

    level = np.zeros(n + forecast_steps)
    trend = np.zeros(n + forecast_steps)
    season = np.zeros(n + forecast_steps)
    fitted = np.zeros(n)

    # 初始值
    level[period - 1] = np.mean(data[:period])
    for i in range(period):
        season[i] = data[i] - level[period - 1]

    for t in range(period, n):
        prev_level, prev_trend = level[t - 1], trend[t - 1]
        level[t] = alpha * (data[t] - season[t - period]) + (1 - alpha) * (prev_level + prev_trend)
        trend[t] = beta * (level[t] - prev_level) + (1 - beta) * prev_trend
        season[t] = gamma * (data[t] - level[t]) + (1 - gamma) * season[t - period]
        fitted[t] = prev_level + prev_trend + season[t - period]

    # 预测
    pred = np.array([
        level[n - 1] + (t + 1) * trend[n - 1] + season[(n - period + t) % period]
        for t in range(forecast_steps)
    ])

    return {"fitted": fitted, "pred": pred, "level": level, "trend": trend, "season": season}


# ═══════════════════════════════════════════════
# BP 神经网络（从零实现）
# ═══════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1 - s)


class SimpleBP:
    """
    简单 BP 神经网络（单隐藏层）。

    适用场景：小样本非线性回归/分类的备用方案。
    竞赛中更推荐用 sklearn MLPClassifier/MLPRegressor。

    示例:
        bp = SimpleBP(n_input=3, n_hidden=10, n_output=1)
        bp.train(X, y, epochs=1000, lr=0.1)
        pred = bp.predict(X_test)
    """

    def __init__(self, n_input: int, n_hidden: int = 10, n_output: int = 1) -> None:
        self.w1 = np.random.randn(n_input, n_hidden) * 0.1
        self.b1 = np.zeros((1, n_hidden))
        self.w2 = np.random.randn(n_hidden, n_output) * 0.1
        self.b2 = np.zeros((1, n_output))
        self.loss_history: List[float] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.w1 + self.b1
        self.a1 = _sigmoid(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.z2
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        m = X.shape[0]
        dz2 = self.a2 - y.reshape(-1, 1)
        dw2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = dz2 @ self.w2.T * _sigmoid_deriv(self.z1)
        dw1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.1,
        verbose: bool = False,
    ) -> "SimpleBP":
        for i in range(epochs):
            pred = self.forward(X)
            loss = float(np.mean((pred - y.reshape(-1, 1)) ** 2))
            self.loss_history.append(loss)
            self.backward(X, y, lr)
            if verbose and (i + 1) % 200 == 0:
                print(f"Epoch {i + 1}, loss={loss:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).flatten()


# ═══════════════════════════════════════════════
# 时间序列平稳性检验
# ═══════════════════════════════════════════════

def adf_test(data: np.ndarray) -> Dict[str, Any]:
    """
    ADF 单位根检验（判断时间序列是否平稳）。

    Returns
    -------
    dict
        {"adf_stat": ..., "p_value": ..., "is_stationary": bool}
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(data, autolag="AIC")
    return {
        "adf_stat": float(result[0]),
        "p_value": float(result[1]),
        "critical_values": result[4],
        "is_stationary": bool(result[1] < 0.05),
    }


# ═══════════════════════════════════════════════
# 高斯过程回归
# ═══════════════════════════════════════════════

def gp_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Dict[str, Any]:
    """
    高斯过程回归。
    适用小样本、非线性预测，自带置信区间。
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(X_train, y_train)
    y_pred, sigma = gp.predict(X_test, return_std=True)
    return {
        "y_pred": y_pred,
        "y_std": sigma,
        "model": gp,
    }
