"""
mm_optim.py — 优化模型工具集

线性规划、整数规划、非线性规划、遗传算法、模拟退火、粒子群、多目标优化
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize, linprog, differential_evolution
import warnings


# ═══════════════════════════════════════════════
# 线性规划
# ═══════════════════════════════════════════════

def solve_lp(
    c: List[float],
    A_ub: Optional[List[List[float]]] = None,
    b_ub: Optional[List[float]] = None,
    A_eq: Optional[List[List[float]]] = None,
    b_eq: Optional[List[float]] = None,
    bounds: Optional[List[Tuple]] = None,
    method: str = "highs",
) -> Dict:
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
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=method)
    if res.success:
        return {"status": "success", "x": res.x, "fval": res.fun, "nit": res.nit}
    else:
        return {"status": "failed", "message": res.message}


# ═══════════════════════════════════════════════
# 整数规划（暴力分支定界 + PuLP封装）
# ═══════════════════════════════════════════════

try:
    import pulp
    
    def solve_ilp(
        c: List[float],
        A: List[List[float]],
        b: List[float],
        bounds: Optional[List[Tuple]] = None,
        sense: int = 1,  # 1=min, -1=max
        integers: Optional[List[int]] = None,
    ) -> Dict:
        """
        整数规划（使用 PuLP + CBC 求解器）。
        
        Parameters
        ----------
        c : list
            目标函数系数
        A : list of list
            约束矩阵 (A x ≤ b)
        b : list
            约束右端项
        bounds : list of (lb, ub)
            变量界限
        sense : int
            1=最小化, -1=最大化
        integers : list of int
            整数变量索引列表。None 表示全部整数
        
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
                if lb is not None: x[i].lowBound = lb
                if ub is not None: x[i].upBound = ub
        
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
        else:
            return {"status": "failed", "message": pulp.LpStatus[prob.status]}
except ImportError:
    def solve_ilp(*args, **kwargs):
        raise ImportError("PuLP required: pip install pulp")


# ═══════════════════════════════════════════════
# 非线性规划
# ═══════════════════════════════════════════════

def solve_nlp(
    fun: Callable,
    x0: List[float],
    bounds: Optional[List[Tuple]] = None,
    constraints: Optional[List[Dict]] = None,
    method: str = "SLSQP",
    **options,
) -> Dict:
    """
    非线性规划求解。
    
    Parameters
    ----------
    fun : callable
        目标函数 fun(x) → float
    x0 : list
        初始点
    constraints : list of dict
        [{"type": "ineq", "fun": lambda x: ...}, ...]
        注意：ineq 表示 fun(x) ≥ 0
    
    Returns
    -------
    dict
    """
    res = minimize(fun, x0, method=method, bounds=bounds,
                   constraints=constraints, options=options)
    return {
        "status": "success" if res.success else "failed",
        "x": res.x,
        "fval": res.fun,
        "nit": res.nit if hasattr(res, 'nit') else None,
        "message": res.message,
    }


# ═══════════════════════════════════════════════
# 遗传算法
# ═══════════════════════════════════════════════

def genetic_algorithm(
    fitness_func: Callable,
    n_dim: int,
    bounds: List[Tuple],
    pop_size: int = 50,
    max_iter: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elite_ratio: float = 0.1,
    maximize: bool = True,
    verbose: bool = False,
    callback: Optional[Callable] = None,
) -> Dict:
    """
    遗传算法（从头实现，不依赖外部库）。
    
    Parameters
    ----------
    fitness_func : callable
        适应度函数 fitness(x) → float
    n_dim : int
        变量维度
    bounds : list of (low, high)
        每个变量的取值范围
    maximize : bool
        True = 最大化, False = 最小化
    
    Returns
    -------
    dict
        {"best_x": [...], "best_fitness": ..., "history": [...]}
    """
    # 初始化种群
    pop = np.random.uniform(
        [b[0] for b in bounds],
        [b[1] for b in bounds],
        (pop_size, n_dim)
    )
    
    # 适应度
    def _fitness(x):
        f = fitness_func(x)
        return f if maximize else -f
    
    fitness = np.array([_fitness(ind) for ind in pop])
    best_idx = np.argmax(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    history = [best_f if maximize else -best_f]
    
    n_elite = max(1, int(pop_size * elite_ratio))
    
    for gen in range(max_iter):
        new_pop = []
        
        # 精英保留
        elite_indices = np.argsort(fitness)[-n_elite:]
        for idx in elite_indices:
            new_pop.append(pop[idx].copy())
        
        while len(new_pop) < pop_size:
            # 锦标赛选择
            def _tournament(k=3):
                idx = np.random.randint(0, pop_size, k)
                return pop[idx[np.argmax(fitness[idx])]]
            
            if np.random.rand() < crossover_rate:
                p1, p2 = _tournament(), _tournament()
                alpha = np.random.rand(n_dim)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = (1 - alpha) * p1 + alpha * p2
                
                # 变异
                for c in [c1, c2]:
                    mask = np.random.rand(n_dim) < mutation_rate
                    c[mask] = np.random.uniform(
                        [bounds[i][0] for i in range(n_dim)],
                        [bounds[i][1] for i in range(n_dim)],
                        mask.sum()
                    ) if mask.any() else c
                    # 边界处理
                    c = np.clip(c, [b[0] for b in bounds], [b[1] for b in bounds])
                    new_pop.append(c)
            else:
                new_pop.append(_tournament().copy())
        
        pop = np.array(new_pop[:pop_size])
        fitness = np.array([_fitness(ind) for ind in pop])
        
        curr_best_idx = np.argmax(fitness)
        if fitness[curr_best_idx] > best_f:
            best_x = pop[curr_best_idx].copy()
            best_f = fitness[curr_best_idx]
        
        history.append(best_f if maximize else -best_f)
        
        if verbose and (gen + 1) % 10 == 0:
            print(f"Gen {gen+1}/{max_iter}, Best={history[-1]:.6f}")
        
        if callback:
            callback(gen, best_x, best_f if maximize else -best_f)
    
    return {
        "best_x": best_x.tolist(),
        "best_fitness": best_f if maximize else -best_f,
        "history": history,
    }


# ═══════════════════════════════════════════════
# 模拟退火
# ═══════════════════════════════════════════════

def simulated_annealing(
    func: Callable,
    x0: List[float],
    bounds: List[Tuple],
    T_start: float = 100,
    T_end: float = 1e-6,
    cooling_rate: float = 0.95,
    max_iter: int = 1000,
    minimize: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    模拟退火算法。
    
    Parameters
    ----------
    func : callable
        目标函数
    x0 : list
        初始解
    bounds : list of (low, high)
    minimize : bool
        True = 最小化, False = 最大化
    
    Returns
    -------
    dict
    """
    x_current = np.array(x0, dtype=float)
    f_current = func(x_current)
    best_x = x_current.copy()
    best_f = f_current
    history = [f_current]
    sign = 1 if minimize else -1
    
    T = T_start
    n_dim = len(x0)
    
    for iteration in range(max_iter):
        # 生成邻域解
        x_new = x_current + np.random.uniform(-1, 1, n_dim) * T / T_start * 5
        x_new = np.clip(x_new, [b[0] for b in bounds], [b[1] for b in bounds])
        f_new = func(x_new)
        
        delta = f_new - f_current
        if sign * delta < 0 or np.random.rand() < np.exp(-sign * delta / T):
            x_current = x_new
            f_current = f_new
            if sign * f_new < sign * best_f:
                best_x = x_new.copy()
                best_f = f_new
        
        T *= cooling_rate
        if T < T_end:
            break
        
        history.append(best_f)
        if verbose and (iteration + 1) % 100 == 0:
            print(f"Iter {iteration+1}, T={T:.6f}, Best={best_f:.6f}")
    
    return {
        "best_x": best_x.tolist(),
        "best_fval": best_f,
        "history": history,
        "iterations": iteration + 1,
    }


# ═══════════════════════════════════════════════
# 粒子群优化
# ═══════════════════════════════════════════════

def particle_swarm(
    func: Callable,
    n_dim: int,
    bounds: List[Tuple],
    n_particles: int = 30,
    max_iter: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    minimize: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    粒子群优化算法 (PSO)。
    """
    sign = 1 if minimize else -1
    
    # 初始化
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    positions = np.random.uniform(lb, ub, (n_particles, n_dim))
    velocities = np.random.uniform(-(ub-lb)*0.1, (ub-lb)*0.1, (n_particles, n_dim))
    
    fitness = np.array([sign * func(p) for p in positions])
    
    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()
    
    gbest_idx = np.argmin(fitness) if minimize else np.argmax(fitness)
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = fitness[gbest_idx]
    history = [gbest_fit * sign]
    
    for iteration in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(n_dim), np.random.rand(n_dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_pos[i] - positions[i]) +
                             c2 * r2 * (gbest_pos - positions[i]))
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)
            
            f = sign * func(positions[i])
            if (minimize and f < pbest_fit[i]) or (not minimize and f > pbest_fit[i]):
                pbest_pos[i] = positions[i].copy()
                pbest_fit[i] = f
            if (minimize and f < gbest_fit) or (not minimize and f > gbest_fit):
                gbest_pos = positions[i].copy()
                gbest_fit = f
        
        history.append(gbest_fit * sign)
        if verbose and (iteration + 1) % 20 == 0:
            print(f"Iter {iteration+1}, Best={history[-1]:.6f}")
    
    return {
        "best_x": gbest_pos.tolist(),
        "best_fval": gbest_fit * sign,
        "history": history,
    }


# ═══════════════════════════════════════════════
# 多目标优化（加权法 + Pareto前沿）
# ═══════════════════════════════════════════════

def multi_objective_weighted(
    objectives: List[Callable],
    weights: List[float],
    n_dim: int,
    bounds: List[Tuple],
    method: str = "ga",
    **kwargs,
) -> Dict:
    """
    加权多目标优化。
    将多个目标加权求和为单目标。
    
    Parameters
    ----------
    objectives : list of callable
        每个目标函数 objective_i(x) → float
    weights : list of float
        权重，和为1
    
    Returns
    -------
    dict
    """
    weights = np.array(weights) / np.sum(weights)
    
    def weighted_sum(x):
        return sum(w * obj(x) for w, obj in zip(weights, objectives))
    
    if method == "ga":
        result = genetic_algorithm(
            weighted_sum, n_dim, bounds,
            maximize=False, **kwargs
        )
    elif method == "sa":
        result = simulated_annealing(
            weighted_sum, kwargs.get("x0", [0]*n_dim), bounds,
            minimize=True, **kwargs
        )
    elif method == "pso":
        result = particle_swarm(
            weighted_sum, n_dim, bounds,
            minimize=True, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result


def nsga2_mo(
    objectives: List[Callable],
    n_dim: int,
    bounds: List[Tuple],
    pop_size: int = 100,
    max_iter: int = 200,
) -> Dict:
    """
    简化的 NSGA-II 多目标进化算法。
    
    Returns
    -------
    dict
        {"pareto_front": [(f1, f2, ...), ...],
         "pareto_solutions": [x1, x2, ...]}
    """
    n_obj = len(objectives)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # 初始化
    pop = np.random.uniform(lb, ub, (pop_size, n_dim))
    
    def _evaluate(p):
        return np.array([obj(p) for obj in objectives])
    
    fitness = np.array([_evaluate(p) for p in pop])
    
    for gen in range(max_iter):
        # 交叉变异产生子代
        offspring = []
        for _ in range(pop_size):
            idx = np.random.choice(pop_size, 2)
            if np.random.rand() < 0.9:
                alpha = np.random.rand(n_dim)
                child = alpha * pop[idx[0]] + (1 - alpha) * pop[idx[1]]
            else:
                child = pop[np.random.randint(pop_size)].copy()
            # 变异
            mask = np.random.rand(n_dim) < 0.1
            child[mask] = np.random.uniform(lb[mask], ub[mask])
            child = np.clip(child, lb, ub)
            offspring.append(child)
        
        offspring = np.array(offspring)
        combined = np.vstack([pop, offspring])
        combined_fit = np.array([_evaluate(p) for p in combined])
        
        # 非支配排序（简单实现）
        n = len(combined)
        dominated = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(combined_fit[i] <= combined_fit[j]) and np.any(combined_fit[i] < combined_fit[j]):
                        dominated[i] += 1
        
        # 选择非支配解（前沿）
        pareto_mask = dominated == 0
        pareto_count = pareto_mask.sum()
        
        if pareto_count >= pop_size:
            # 从前沿中选 pop_size 个
            indices = np.where(pareto_mask)[0]
            selected = np.random.choice(indices, pop_size, replace=False)
        else:
            # 全选非支配解，其余从被支配中选
            non_pareto = np.where(~pareto_mask)[0]
            selected = np.where(pareto_mask)[0].tolist()
            remaining = pop_size - len(selected)
            if remaining > 0 and len(non_pareto) > 0:
                extra = np.random.choice(non_pareto, min(remaining, len(non_pareto)), replace=False)
                selected = list(selected) + list(extra)
            selected = np.array(selected)
        
        pop = combined[selected[:pop_size]]
        fitness = combined_fit[selected[:pop_size]]
    
    # 最终 Pareto 前沿
    final_fit = np.array([_evaluate(p) for p in pop])
    n = len(pop)
    dominated = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                if np.all(final_fit[i] <= final_fit[j]) and np.any(final_fit[i] < final_fit[j]):
                    dominated[i] += 1
    
    pareto_mask = dominated == 0
    return {
        "pareto_front": [tuple(final_fit[i]) for i in range(n) if pareto_mask[i]],
        "pareto_solutions": [pop[i].tolist() for i in range(n) if pareto_mask[i]],
    }


# ═══════════════════════════════════════════════
# 0-1背包问题（动态规划）
# ═══════════════════════════════════════════════

def knapsack_01(values: List[float], weights: List[float], capacity: float) -> Dict:
    """
    0-1背包问题（动态规划）。
    
    Parameters
    ----------
    values : list
        每个物品的价值
    weights : list
        每个物品的重量
    capacity : float
        背包容量
    
    Returns
    -------
    dict
        {"max_value": ..., "selected": [0,1,...], "total_weight": ...}
    """
    n = len(values)
    # 如果重量为整数且不大，用经典DP
    if all(isinstance(w, (int, np.integer)) for w in weights) and capacity < 10000:
        cap = int(capacity)
        w_int = [int(w) for w in weights]
        dp = np.zeros((n + 1, cap + 1))
        for i in range(1, n + 1):
            for w in range(cap + 1):
                if w_int[i-1] <= w:
                    dp[i, w] = max(dp[i-1, w], dp[i-1, w-w_int[i-1]] + values[i-1])
                else:
                    dp[i, w] = dp[i-1, w]
        
        # 回溯
        selected = [0] * n
        w = cap
        for i in range(n, 0, -1):
            if dp[i, w] != dp[i-1, w]:
                selected[i-1] = 1
                w -= w_int[i-1]
        
        return {
            "max_value": dp[n, cap],
            "selected": selected,
            "total_weight": cap - w,
        }
    else:
        # 用遗传算法近似解
        def fitness(x):
            x = np.array(x) > 0.5
            total_w = np.sum(weights * x)
            if total_w > capacity:
                return 0
            return np.sum(values * x)
        
        result = genetic_algorithm(
            fitness, n, [(0, 1)] * n,
            pop_size=100, max_iter=200,
            maximize=True
        )
        x = np.array(result["best_x"]) > 0.5
        return {
            "max_value": result["best_fitness"],
            "selected": x.astype(int).tolist(),
            "total_weight": np.sum(weights * x),
        }
