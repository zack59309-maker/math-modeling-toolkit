"""
mm_optim.py — 优化模型工具集

线性规划、整数规划、非线性规划、遗传算法、模拟退火、粒子群、多目标优化
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scipy.optimize import minimize, linprog


# ═══════════════════════════════════════════════
# 线性规划
# ═══════════════════════════════════════════════

def solve_lp(
    c: Sequence[float],
    A_ub: Optional[Sequence[Sequence[float]]] = None,
    b_ub: Optional[Sequence[float]] = None,
    A_eq: Optional[Sequence[Sequence[float]]] = None,
    b_eq: Optional[Sequence[float]] = None,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    method: str = "highs",
) -> Dict[str, Any]:
    """
    线性规划求解。
    min c^T x
    s.t. A_ub x ≤ b_ub
         A_eq x = b_eq
         bounds[i] = (lb, ub)

    示例:
        solve_lp(
            c=[-3, -2],           # 求 max 3x+2y → 取负
            A_ub=[[2, 1], [1, 1]],
            b_ub=[18, 8],
            bounds=[(0, None), (0, None)]
        )
    """
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method=method,
    )
    if res.success:
        return {"status": "success", "x": res.x, "fval": res.fun, "nit": res.nit}
    return {"status": "failed", "message": res.message}


# ═══════════════════════════════════════════════
# 整数规划
# ═══════════════════════════════════════════════

try:
    import pulp

    def solve_ilp(
        c: Sequence[float],
        A: Sequence[Sequence[float]],
        b: Sequence[float],
        bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
        sense: int = 1,
        integers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        整数规划（使用 PuLP + CBC 求解器）。

        Parameters
        ----------
        sense : int
            1=最小化, -1=最大化
        integers : list of int, optional
            整数变量索引列表。None 表示全部整数。

        Returns
        -------
        dict
            {"status": ..., "x": [...], "fval": ...}
        """
        n = len(c)
        prob = pulp.LpProblem("ILP", pulp.LpMinimize if sense == 1 else pulp.LpMaximize)

        x = [pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer") for i in range(n)]

        if bounds:
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None:
                    x[i].lowBound = lb
                if ub is not None:
                    x[i].upBound = ub

        if integers is not None:
            for i in range(n):
                if i not in integers:
                    x[i].cat = "Continuous"

        prob += pulp.lpSum(c[i] * x[i] for i in range(n))

        for i in range(len(A)):
            prob += pulp.lpSum(A[i][j] * x[j] for j in range(n)) <= b[i]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] == "Optimal":
            return {
                "status": "success",
                "x": [pulp.value(xi) for xi in x],
                "fval": pulp.value(prob.objective),
            }
        return {"status": "failed", "message": pulp.LpStatus[prob.status]}

except ImportError:
    def solve_ilp(*args: Any, **kwargs: Any) -> Dict[str, str]:
        raise ImportError("PuLP required: pip install pulp")


# ═══════════════════════════════════════════════
# 非线性规划
# ═══════════════════════════════════════════════

def solve_nlp(
    fun: Callable[[np.ndarray], float],
    x0: Sequence[float],
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    constraints: Optional[List[Dict[str, Any]]] = None,
    method: str = "SLSQP",
    **options: Any,
) -> Dict[str, Any]:
    """
    非线性规划求解。

    Parameters
    ----------
    constraints : list of dict, optional
        [{"type": "ineq", "fun": lambda x: ...}, ...]
        注意：ineq 表示 fun(x) ≥ 0

    Returns
    -------
    dict
    """
    res = minimize(fun, x0, method=method, bounds=bounds, constraints=constraints, options=options)
    return {
        "status": "success" if res.success else "failed",
        "x": res.x,
        "fval": res.fun,
        "nit": getattr(res, "nit", None),
        "message": res.message,
    }


# ═══════════════════════════════════════════════
# 遗传算法
# ═══════════════════════════════════════════════

def genetic_algorithm(
    fitness_func: Callable[[np.ndarray], float],
    n_dim: int,
    bounds: Sequence[Tuple[float, float]],
    pop_size: int = 50,
    max_iter: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elite_ratio: float = 0.1,
    maximize: bool = True,
    verbose: bool = False,
    callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
) -> Dict[str, Any]:
    """
    遗传算法（从头实现，不依赖外部库）。

    Parameters
    ----------
    maximize : bool
        True = 最大化, False = 最小化

    Returns
    -------
    dict
        {"best_x": [...], "best_fitness": ..., "history": [...]}
    """
    bounds_arr = np.array(bounds, dtype=float)
    lb, ub = bounds_arr[:, 0], bounds_arr[:, 1]

    # 初始化种群
    pop = np.random.uniform(lb, ub, (pop_size, n_dim))

    # 适应度（统一为最大化方向）
    sign = 1.0 if maximize else -1.0
    def _fitness(x: np.ndarray) -> float:
        return sign * fitness_func(x)

    fitness = np.array([_fitness(ind) for ind in pop])
    best_idx = int(np.argmax(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    history: List[float] = [sign * best_f]

    n_elite = max(1, int(pop_size * elite_ratio))

    for gen in range(max_iter):
        new_pop: List[np.ndarray] = []

        # 精英保留
        elite_indices = np.argsort(fitness)[-n_elite:]
        for idx in elite_indices:
            new_pop.append(pop[idx].copy())

        while len(new_pop) < pop_size:
            # 锦标赛选择
            def _tournament(k: int = 3) -> np.ndarray:
                idx = np.random.randint(0, pop_size, k)
                return pop[idx[np.argmax(fitness[idx])]]

            if np.random.rand() < crossover_rate:
                p1, p2 = _tournament(), _tournament()
                alpha = np.random.rand(n_dim)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = (1 - alpha) * p1 + alpha * p2

                for c in [c1, c2]:
                    mask = np.random.rand(n_dim) < mutation_rate
                    if mask.any():
                        c[mask] = np.random.uniform(lb[mask], ub[mask])
                    c = np.clip(c, lb, ub)
                    new_pop.append(c)
            else:
                new_pop.append(_tournament().copy())

        pop = np.array(new_pop[:pop_size])
        fitness = np.array([_fitness(ind) for ind in pop])

        curr_best_idx = int(np.argmax(fitness))
        if fitness[curr_best_idx] > best_f:
            best_x = pop[curr_best_idx].copy()
            best_f = float(fitness[curr_best_idx])

        best_val = sign * best_f
        history.append(best_val)

        if verbose and (gen + 1) % 10 == 0:
            print(f"Gen {gen + 1}/{max_iter}, Best={best_val:.6f}")

        if callback:
            callback(gen, best_x, best_val)

    return {
        "best_x": best_x.tolist(),
        "best_fitness": sign * best_f,
        "history": history,
    }


# ═══════════════════════════════════════════════
# 模拟退火
# ═══════════════════════════════════════════════

def simulated_annealing(
    func: Callable[[np.ndarray], float],
    x0: Sequence[float],
    bounds: Sequence[Tuple[float, float]],
    T_start: float = 100.0,
    T_end: float = 1e-6,
    cooling_rate: float = 0.95,
    max_iter: int = 1000,
    minimize: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    模拟退火算法。

    Parameters
    ----------
    minimize : bool
        True = 最小化, False = 最大化

    Returns
    -------
    dict
    """
    bounds_arr = np.array(bounds, dtype=float)
    lb, ub = bounds_arr[:, 0], bounds_arr[:, 1]
    x_current = np.array(x0, dtype=float)
    f_current = float(func(x_current))
    best_x = x_current.copy()
    best_f = f_current
    history: List[float] = [f_current]
    sign = 1.0 if minimize else -1.0
    n_dim = len(x0)

    T = T_start
    iter_count = 0
    for iteration in range(max_iter):
        # 邻域扰动：步长随温度衰减
        step = T / T_start * 5.0
        x_new = x_current + np.random.uniform(-step, step, n_dim)
        x_new = np.clip(x_new, lb, ub)
        f_new = float(func(x_new))

        delta = f_new - f_current
        if sign * delta < 0 or np.random.rand() < np.exp(-sign * delta / max(T, 1e-10)):
            x_current = x_new
            f_current = f_new
            if sign * (f_new - best_f) < 0:
                best_x = x_new.copy()
                best_f = f_new

        T *= cooling_rate
        if T < T_end:
            break
        iter_count = iteration + 1
        history.append(best_f)

        if verbose and (iteration + 1) % 100 == 0:
            print(f"Iter {iteration + 1}, T={T:.6e}, Best={best_f:.6f}")

    return {
        "best_x": best_x.tolist(),
        "best_fval": best_f,
        "history": history,
        "iterations": iter_count,
    }


# ═══════════════════════════════════════════════
# 粒子群优化
# ═══════════════════════════════════════════════

def particle_swarm(
    func: Callable[[np.ndarray], float],
    n_dim: int,
    bounds: Sequence[Tuple[float, float]],
    n_particles: int = 30,
    max_iter: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    minimize: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    粒子群优化算法 (PSO)。
    """
    sign = 1.0 if minimize else -1.0
    bounds_arr = np.array(bounds, dtype=float)
    lb, ub = bounds_arr[:, 0], bounds_arr[:, 1]

    positions = np.random.uniform(lb, ub, (n_particles, n_dim))
    velocities = np.random.uniform(-(ub - lb) * 0.1, (ub - lb) * 0.1, (n_particles, n_dim))

    fitness = np.array([sign * func(p) for p in positions])

    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    gbest_idx = int(np.argmin(fitness) if minimize else np.argmax(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = float(fitness[gbest_idx])
    history: List[float] = [sign * gbest_fit]

    rng = np.random.default_rng()
    for iteration in range(max_iter):
        for i in range(n_particles):
            r1, r2 = rng.random(n_dim), rng.random(n_dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_pos[i] - positions[i])
                + c2 * r2 * (gbest_pos - positions[i])
            )
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

            f = sign * float(func(positions[i]))
            if (minimize and f < pbest_fit[i]) or (not minimize and f > pbest_fit[i]):
                pbest_pos[i] = positions[i].copy()
                pbest_fit[i] = f
            if (minimize and f < gbest_fit) or (not minimize and f > gbest_fit):
                gbest_pos = positions[i].copy()
                gbest_fit = f

        history.append(sign * gbest_fit)
        if verbose and (iteration + 1) % 20 == 0:
            print(f"Iter {iteration + 1}, Best={history[-1]:.6f}")

    return {
        "best_x": gbest_pos.tolist(),
        "best_fval": sign * gbest_fit,
        "history": history,
    }


# ═══════════════════════════════════════════════
# 多目标优化
# ═══════════════════════════════════════════════

def multi_objective_weighted(
    objectives: List[Callable[[np.ndarray], float]],
    weights: Sequence[float],
    n_dim: int,
    bounds: Sequence[Tuple[float, float]],
    method: str = "ga",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    加权多目标优化。
    将多个目标加权求和为单目标。
    """
    weights_arr = np.array(weights) / np.sum(weights)

    def weighted_sum(x: np.ndarray) -> float:
        return float(sum(w * obj(x) for w, obj in zip(weights_arr, objectives)))

    if method == "ga":
        return genetic_algorithm(weighted_sum, n_dim, bounds, maximize=False, **kwargs)
    elif method == "sa":
        return simulated_annealing(
            weighted_sum, kwargs.get("x0", [0.0] * n_dim), bounds, minimize=True, **kwargs
        )
    elif method == "pso":
        return particle_swarm(weighted_sum, n_dim, bounds, minimize=True, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def nsga2_mo(
    objectives: List[Callable[[np.ndarray], float]],
    n_dim: int,
    bounds: Sequence[Tuple[float, float]],
    pop_size: int = 100,
    max_iter: int = 200,
) -> Dict[str, Any]:
    """
    简化的 NSGA-II 多目标进化算法。

    Returns
    -------
    dict
        {"pareto_front": [(f1, f2, ...), ...],
         "pareto_solutions": [x1, x2, ...]}
    """
    n_obj = len(objectives)
    bounds_arr = np.array(bounds, dtype=float)
    lb, ub = bounds_arr[:, 0], bounds_arr[:, 1]

    pop = np.random.uniform(lb, ub, (pop_size, n_dim))

    def _evaluate(p: np.ndarray) -> np.ndarray:
        return np.array([obj(p) for obj in objectives])

    fitness = np.array([_evaluate(p) for p in pop])

    for gen in range(max_iter):
        # 交叉变异产生子代
        children: List[np.ndarray] = []
        for _ in range(pop_size):
            idx = np.random.choice(pop_size, 2, replace=False)
            if np.random.rand() < 0.9:
                alpha = np.random.rand(n_dim)
                child = alpha * pop[idx[0]] + (1 - alpha) * pop[idx[1]]
            else:
                child = pop[np.random.randint(pop_size)].copy()
            mask = np.random.rand(n_dim) < 0.1
            child[mask] = np.random.uniform(lb[mask], ub[mask])
            child = np.clip(child, lb, ub)
            children.append(child)

        offspring = np.array(children)
        combined = np.vstack([pop, offspring])
        combined_fit = np.array([_evaluate(p) for p in combined])

        # 非支配排序
        n = len(combined)
        dominated = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(combined_fit[i] <= combined_fit[j]) and np.any(combined_fit[i] < combined_fit[j]):
                    dominated[i] += 1

        pareto_mask = dominated == 0
        pareto_count = int(pareto_mask.sum())

        if pareto_count >= pop_size:
            indices = np.where(pareto_mask)[0]
            selected = np.random.choice(indices, pop_size, replace=False)
        else:
            selected = np.where(pareto_mask)[0].tolist()
            remaining = pop_size - len(selected)
            non_pareto = np.where(~pareto_mask)[0]
            if remaining > 0 and len(non_pareto) > 0:
                extra = np.random.choice(non_pareto, min(remaining, len(non_pareto)), replace=False)
                selected = list(selected) + list(extra)
            selected = np.array(selected, dtype=int)

        pop = combined[selected[:pop_size]]
        fitness = combined_fit[selected[:pop_size]]

    # 最终 Pareto 前沿
    final_fit = np.array([_evaluate(p) for p in pop])
    n_final = len(pop)
    dominated = np.zeros(n_final, dtype=int)
    for i in range(n_final):
        for j in range(n_final):
            if i != j and np.all(final_fit[i] <= final_fit[j]) and np.any(final_fit[i] < final_fit[j]):
                dominated[i] += 1

    pareto_mask = dominated == 0
    return {
        "pareto_front": [tuple(final_fit[i]) for i in range(n_final) if pareto_mask[i]],
        "pareto_solutions": [pop[i].tolist() for i in range(n_final) if pareto_mask[i]],
    }


# ═══════════════════════════════════════════════
# 0-1背包问题
# ═══════════════════════════════════════════════

def knapsack_01(
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
) -> Dict[str, Any]:
    """
    0-1背包问题（动态规划）。

    Returns
    -------
    dict
        {"max_value": ..., "selected": [0,1,...], "total_weight": ...}
    """
    n = len(values)
    # 整数重量 + 容量不太大 → 经典 DP
    if all(isinstance(w, (int, np.integer)) for w in weights) and capacity < 10000:
        cap = int(capacity)
        w_int = [int(w) for w in weights]
        dp = np.zeros((n + 1, cap + 1))
        for i in range(1, n + 1):
            wi, vi = w_int[i - 1], values[i - 1]
            dp[i, :wi] = dp[i - 1, :wi]
            dp[i, wi:] = np.maximum(dp[i - 1, wi:], dp[i - 1, :cap - wi + 1] + vi)

        # 回溯
        selected = [0] * n
        w = cap
        for i in range(n, 0, -1):
            if dp[i, w] != dp[i - 1, w]:
                selected[i - 1] = 1
                w -= w_int[i - 1]
        return {"max_value": float(dp[n, cap]), "selected": selected, "total_weight": float(cap - w)}
    else:
        # 实数权重 → 遗传算法近似
        def fitness(x: np.ndarray) -> float:
            x_bin = x > 0.5
            total_w = float(np.sum(weights * x_bin))
            return 0.0 if total_w > capacity else float(np.sum(values * x_bin))

        result = genetic_algorithm(fitness, n, [(0, 1)] * n, pop_size=100, max_iter=200, maximize=True)
        x = (np.array(result["best_x"]) > 0.5).astype(int)
        return {
            "max_value": result["best_fitness"],
            "selected": x.tolist(),
            "total_weight": float(np.sum(weights * x)),
        }
