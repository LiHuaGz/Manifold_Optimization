import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve, cho_factor, cho_solve, solve_triangular
from numba import njit
import time

@njit(cache=True)
def _compute_step_length(Ap, Ax, b, working_mask, tol):
    """Numba 加速的步长计算"""
    n = len(Ap)
    alpha = 1.0
    blocking_idx = -1
    
    for i in range(n):
        if not working_mask[i] and Ap[i] < -tol:
            dist = (b[i] - Ax[i]) / Ap[i]
            if dist < alpha:
                alpha = dist
                blocking_idx = i
    
    return max(0.0, alpha), blocking_idx


def solve_active_set_qp(G, c, A, b, x0, tol=1e-6, max_iter=1000):
    """
    有效集法求解: min 1/2 x^T G x + c^T x, s.t. Ax >= b
    高度优化版本：预计算 + Schur补 + Numba JIT
    """
    n_vars = len(c)
    n_constraints = len(b)
    
    x = np.array(x0, dtype=np.float64)
    
    # 预计算 G 的 Cholesky 分解
    try:
        G_cho = cho_factor(G)
        use_cholesky = True
    except:
        use_cholesky = False
    
    # 关键优化：预计算 G^{-1} @ A^T，这是 n×m 矩阵
    # 避免每次迭代都调用 cho_solve
    if use_cholesky:
        G_inv_AT = cho_solve(G_cho, A.T)  # n × m
    else:
        G_inv_AT = np.linalg.solve(G, A.T)
    
    # 预计算 A @ G^{-1} @ A^T 的完整矩阵（m × m）
    # 这样 Schur 补只需取子矩阵
    AG_inv_AT = A @ G_inv_AT  # m × m
    
    # 预计算 A @ x
    Ax = A @ x
    
    # 工作集：使用布尔数组代替 set，更快
    working_mask = np.zeros(n_constraints, dtype=np.bool_)
    residuals = Ax - b
    working_mask[np.abs(residuals) < tol] = True
    
    for k in range(max_iter):
        # 获取工作集索引
        working_list = np.where(working_mask)[0]
        n_active = len(working_list)
        
        # 计算梯度
        g = G @ x + c
        
        if n_active > 0:
            if use_cholesky:
                # 使用预计算的矩阵
                G_inv_g = cho_solve(G_cho, g)
                
                # 取 G^{-1} @ A_w^T 的相关列
                G_inv_AwT = G_inv_AT[:, working_list]  # n × n_active
                
                # Schur 补：直接取子矩阵
                S = AG_inv_AT[np.ix_(working_list, working_list)]  # n_active × n_active
                
                # 求解 λ
                rhs_lambda = G_inv_AwT.T @ g  # 等价于 A_w @ G^{-1} @ g
                
                try:
                    S_cho = cho_factor(S)
                    lambdas = cho_solve(S_cho, rhs_lambda)
                except:
                    try:
                        lambdas = solve(S, rhs_lambda, assume_a='sym')
                    except:
                        working_mask[working_list[-1]] = False
                        continue
                
                # p = -G^{-1}@g + G^{-1}@A_w^T @ λ
                p = -G_inv_g + G_inv_AwT @ lambdas
            else:
                # 回退方法
                A_w = A[working_list]
                KKT_size = n_vars + n_active
                KKT = np.zeros((KKT_size, KKT_size), dtype=np.float64)
                KKT[:n_vars, :n_vars] = G
                KKT[:n_vars, n_vars:] = -A_w.T
                KKT[n_vars:, :n_vars] = A_w
                
                rhs = np.zeros(KKT_size, dtype=np.float64)
                rhs[:n_vars] = -g
                
                try:
                    sol = solve(KKT, rhs, assume_a='sym')
                    p = sol[:n_vars]
                    lambdas = sol[n_vars:]
                except:
                    working_mask[working_list[-1]] = False
                    continue
        else:
            if use_cholesky:
                p = cho_solve(G_cho, -g)
            else:
                p = solve(G, -g, assume_a='pos')
            lambdas = np.array([])

        # 检查收敛
        p_norm_sq = np.dot(p, p)
        if p_norm_sq < tol * tol:
            if n_active == 0:
                return x, k, "Optimal (Unconstrained)"
            
            min_lambda_idx = np.argmin(lambdas)
            if lambdas[min_lambda_idx] >= -tol:
                return x, k, "Optimal (KKT Satisfied)"
            else:
                working_mask[working_list[min_lambda_idx]] = False
        else:
            # 计算步长 - 使用 Numba 加速
            Ap = A @ p
            alpha, blocking_idx = _compute_step_length(Ap, Ax, b, working_mask, tol)
            
            # 更新
            x = x + alpha * p
            Ax = Ax + alpha * Ap
            
            if blocking_idx >= 0:
                working_mask[blocking_idx] = True
                
    return x, max_iter, "Max Iterations Reached"

def generate_random_qp(n=100, m=50, seed=42):
    """
    生成随机凸二次规划问题
    n: 变量维度
    m: 不等式约束数量
    """
    np.random.seed(seed)
    
    # 1. 生成正定矩阵 G
    # 方法：生成随机矩阵 M，令 G = M^T M + epsilon * I
    M = np.random.randn(n, n)
    G = np.dot(M.T, M) + 0.1 * np.eye(n)
    
    # 2. 生成随机线性项 c
    c = np.random.randn(n)
    
    # 3. 生成随机约束矩阵 A
    A = np.random.randn(m, n)
    
    # 4. 构造一个必定可行的初始点 x0
    x0 = np.random.randn(n)
    
    # 5. 构造 b，使得 x0 刚好满足约束（或者在可行域内）
    # 令 b = A*x0 - delta，其中 delta >= 0
    # 这样 A*x0 = b + delta >= b，保证 x0 可行
    delta = np.random.rand(m) * 2  # 随机松弛量
    b = np.dot(A, x0) - delta
    
    return G, c, A, b, x0

# ==========================================
# 测试模块：数值实验
# ==========================================

import os
from datetime import datetime

def run_single_experiment(n_vars, n_constr, seed=42, tol=1e-6, max_iter=1000):
    """
    运行单次实验，返回详细结果
    """
    # 生成问题
    G, c, A, b, x0 = generate_random_qp(n=n_vars, m=n_constr, seed=seed)
    
    results = {
        'n_vars': n_vars,
        'n_constr': n_constr,
        'seed': seed,
    }
    
    # 1. 手写有效集法
    start_time = time.time()
    x_ours, iters, status = solve_active_set_qp(G, c, A, b, x0, tol=tol, max_iter=max_iter)
    our_time = time.time() - start_time
    our_obj = 0.5 * x_ours.T @ G @ x_ours + c.T @ x_ours
    
    # 检查约束满足情况
    constraint_violation_ours = np.max(np.maximum(0, b - A @ x_ours))
    
    results['our_time'] = our_time
    results['our_obj'] = our_obj
    results['our_iters'] = iters
    results['our_status'] = status
    results['our_constraint_violation'] = constraint_violation_ours
    
    # 2. OSQP 对比
    P = sparse.csc_matrix(G)
    q = c
    A_osqp = sparse.csc_matrix(-A)
    u_osqp = -b
    l_osqp = np.full(n_constr, -np.inf)
    
    prob = osqp.OSQP()
    prob.setup(P, q, A_osqp, l_osqp, u_osqp, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    
    start_time = time.time()
    res = prob.solve()
    osqp_time = time.time() - start_time
    
    if res.x is not None:
        osqp_obj = 0.5 * res.x.T @ G @ res.x + c.T @ res.x
        constraint_violation_osqp = np.max(np.maximum(0, b - A @ res.x))
    else:
        osqp_obj = np.nan
        constraint_violation_osqp = np.nan
    
    results['osqp_time'] = osqp_time
    results['osqp_obj'] = osqp_obj
    results['osqp_status'] = res.info.status
    results['osqp_iters'] = res.info.iter
    results['osqp_constraint_violation'] = constraint_violation_osqp
    
    # 3. 计算比较指标
    results['obj_diff'] = abs(our_obj - osqp_obj) if not np.isnan(osqp_obj) else np.nan
    results['rel_obj_diff'] = abs(our_obj - osqp_obj) / max(abs(osqp_obj), 1e-10) if not np.isnan(osqp_obj) else np.nan
    results['speedup'] = osqp_time / our_time if our_time > 0 else np.nan
    
    return results


def run_benchmark_suite(verbose=True):
    """
    运行完整的基准测试套件
    """
    results_all = []
    
    # ==========================================
    # 测试1: 固定约束数量，变化变量维度
    # ==========================================
    if verbose:
        print("=" * 60)
        print("测试1: 固定约束数量 m=100，变化变量维度 n")
        print("=" * 60)
    
    fixed_m = 100
    n_list = [50, 100, 200, 500, 1000, 1500, 2000]
    
    for n in n_list:
        if verbose:
            print(f"\n运行 n={n}, m={fixed_m}...")
        
        # 多次运行取平均（不同种子）
        times_ours, times_osqp = [], []
        objs_ours, objs_osqp = [], []
        iters_ours, iters_osqp = [], []
        obj_diffs = []
        violations_ours, violations_osqp = [], []
        
        for seed in range(42, 47):  # 5次运行
            res = run_single_experiment(n, fixed_m, seed=seed)
            times_ours.append(res['our_time'])
            times_osqp.append(res['osqp_time'])
            objs_ours.append(res['our_obj'])
            objs_osqp.append(res['osqp_obj'])
            iters_ours.append(res['our_iters'])
            iters_osqp.append(res['osqp_iters'])
            obj_diffs.append(res['obj_diff'])
            violations_ours.append(res['our_constraint_violation'])
            violations_osqp.append(res['osqp_constraint_violation'])
        
        results_all.append({
            'test_type': 'fixed_m',
            'n': n,
            'm': fixed_m,
            'our_time_mean': np.mean(times_ours),
            'our_time_std': np.std(times_ours),
            'osqp_time_mean': np.mean(times_osqp),
            'osqp_time_std': np.std(times_osqp),
            'our_iters_mean': np.mean(iters_ours),
            'osqp_iters_mean': np.mean(iters_osqp),
            'obj_diff_mean': np.mean(obj_diffs),
            'obj_diff_max': np.max(obj_diffs),
            'our_violation_max': np.max(violations_ours),
            'osqp_violation_max': np.max(violations_osqp),
            'speedup': np.mean(times_osqp) / np.mean(times_ours) if np.mean(times_ours) > 0 else np.nan
        })
        
        if verbose:
            print(f"  手写算法: {np.mean(times_ours):.4f}s (±{np.std(times_ours):.4f}), 迭代: {np.mean(iters_ours):.1f}")
            print(f"  OSQP:     {np.mean(times_osqp):.4f}s (±{np.std(times_osqp):.4f}), 迭代: {np.mean(iters_osqp):.1f}")
            print(f"  目标差异: {np.mean(obj_diffs):.2e}")

    # ==========================================
    # 测试2: 固定变量维度，变化约束数量
    # ==========================================
    if verbose:
        print("\n" + "=" * 60)
        print("测试2: 固定变量维度 n=500，变化约束数量 m")
        print("=" * 60)
    
    fixed_n = 500
    m_list = [50, 100, 200, 300, 400, 500]
    
    for m in m_list:
        if verbose:
            print(f"\n运行 n={fixed_n}, m={m}...")
        
        times_ours, times_osqp = [], []
        objs_ours, objs_osqp = [], []
        iters_ours, iters_osqp = [], []
        obj_diffs = []
        violations_ours, violations_osqp = [], []
        
        for seed in range(42, 47):
            res = run_single_experiment(fixed_n, m, seed=seed)
            times_ours.append(res['our_time'])
            times_osqp.append(res['osqp_time'])
            objs_ours.append(res['our_obj'])
            objs_osqp.append(res['osqp_obj'])
            iters_ours.append(res['our_iters'])
            iters_osqp.append(res['osqp_iters'])
            obj_diffs.append(res['obj_diff'])
            violations_ours.append(res['our_constraint_violation'])
            violations_osqp.append(res['osqp_constraint_violation'])
        
        results_all.append({
            'test_type': 'fixed_n',
            'n': fixed_n,
            'm': m,
            'our_time_mean': np.mean(times_ours),
            'our_time_std': np.std(times_ours),
            'osqp_time_mean': np.mean(times_osqp),
            'osqp_time_std': np.std(times_osqp),
            'our_iters_mean': np.mean(iters_ours),
            'osqp_iters_mean': np.mean(iters_osqp),
            'obj_diff_mean': np.mean(obj_diffs),
            'obj_diff_max': np.max(obj_diffs),
            'our_violation_max': np.max(violations_ours),
            'osqp_violation_max': np.max(violations_osqp),
            'speedup': np.mean(times_osqp) / np.mean(times_ours) if np.mean(times_ours) > 0 else np.nan
        })
        
        if verbose:
            print(f"  手写算法: {np.mean(times_ours):.4f}s (±{np.std(times_ours):.4f}), 迭代: {np.mean(iters_ours):.1f}")
            print(f"  OSQP:     {np.mean(times_osqp):.4f}s (±{np.std(times_osqp):.4f}), 迭代: {np.mean(iters_osqp):.1f}")
            print(f"  目标差异: {np.mean(obj_diffs):.2e}")

    # ==========================================
    # 测试3: 大规模问题测试
    # ==========================================
    if verbose:
        print("\n" + "=" * 60)
        print("测试3: 大规模问题测试")
        print("=" * 60)
    
    large_scale_configs = [
        (1000, 200),
        (1500, 300),
        (2000, 400),
        (2500, 500),
        (3000, 600),
    ]
    
    for n, m in large_scale_configs:
        if verbose:
            print(f"\n运行 n={n}, m={m}...")
        
        times_ours, times_osqp = [], []
        obj_diffs = []
        iters_ours, iters_osqp = [], []
        violations_ours, violations_osqp = [], []
        
        for seed in range(42, 45):  # 3次运行（大规模减少次数）
            res = run_single_experiment(n, m, seed=seed)
            times_ours.append(res['our_time'])
            times_osqp.append(res['osqp_time'])
            obj_diffs.append(res['obj_diff'])
            iters_ours.append(res['our_iters'])
            iters_osqp.append(res['osqp_iters'])
            violations_ours.append(res['our_constraint_violation'])
            violations_osqp.append(res['osqp_constraint_violation'])
        
        results_all.append({
            'test_type': 'large_scale',
            'n': n,
            'm': m,
            'our_time_mean': np.mean(times_ours),
            'our_time_std': np.std(times_ours),
            'osqp_time_mean': np.mean(times_osqp),
            'osqp_time_std': np.std(times_osqp),
            'our_iters_mean': np.mean(iters_ours),
            'osqp_iters_mean': np.mean(iters_osqp),
            'obj_diff_mean': np.mean(obj_diffs),
            'obj_diff_max': np.max(obj_diffs),
            'our_violation_max': np.max(violations_ours),
            'osqp_violation_max': np.max(violations_osqp),
            'speedup': np.mean(times_osqp) / np.mean(times_ours) if np.mean(times_ours) > 0 else np.nan
        })
        
        if verbose:
            print(f"  手写算法: {np.mean(times_ours):.4f}s (±{np.std(times_ours):.4f}), 迭代: {np.mean(iters_ours):.1f}")
            print(f"  OSQP:     {np.mean(times_osqp):.4f}s (±{np.std(times_osqp):.4f}), 迭代: {np.mean(iters_osqp):.1f}")
            print(f"  目标差异: {np.mean(obj_diffs):.2e}")

    # ==========================================
    # 测试4: 条件数测试（矩阵条件数对算法的影响）
    # ==========================================
    if verbose:
        print("\n" + "=" * 60)
        print("测试4: 不同条件数的问题")
        print("=" * 60)
    
    def generate_qp_with_condition_number(n, m, cond_num, seed=42):
        """生成指定条件数的QP问题"""
        np.random.seed(seed)
        
        # 生成指定条件数的正定矩阵
        U, _ = np.linalg.qr(np.random.randn(n, n))
        eigenvalues = np.linspace(1, cond_num, n)
        G = U @ np.diag(eigenvalues) @ U.T
        
        c = np.random.randn(n)
        A = np.random.randn(m, n)
        x0 = np.random.randn(n)
        delta = np.random.rand(m) * 2
        b = np.dot(A, x0) - delta
        
        return G, c, A, b, x0
    
    cond_numbers = [10, 100, 1000, 10000]
    test_n, test_m = 300, 100
    
    for cond_num in cond_numbers:
        if verbose:
            print(f"\n运行 条件数={cond_num}, n={test_n}, m={test_m}...")
        
        times_ours, times_osqp = [], []
        obj_diffs = []
        iters_ours = []
        
        for seed in range(42, 47):
            G, c, A, b, x0 = generate_qp_with_condition_number(test_n, test_m, cond_num, seed)
            
            start_time = time.time()
            x_ours, iters, status = solve_active_set_qp(G, c, A, b, x0)
            our_time = time.time() - start_time
            our_obj = 0.5 * x_ours.T @ G @ x_ours + c.T @ x_ours
            
            P = sparse.csc_matrix(G)
            A_osqp = sparse.csc_matrix(-A)
            u_osqp = -b
            l_osqp = np.full(test_m, -np.inf)
            
            prob = osqp.OSQP()
            prob.setup(P, c, A_osqp, l_osqp, u_osqp, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
            
            start_time = time.time()
            res = prob.solve()
            osqp_time = time.time() - start_time
            osqp_obj = 0.5 * res.x.T @ G @ res.x + c.T @ res.x
            
            times_ours.append(our_time)
            times_osqp.append(osqp_time)
            obj_diffs.append(abs(our_obj - osqp_obj))
            iters_ours.append(iters)
        
        results_all.append({
            'test_type': 'condition_number',
            'n': test_n,
            'm': test_m,
            'cond_num': cond_num,
            'our_time_mean': np.mean(times_ours),
            'our_time_std': np.std(times_ours),
            'osqp_time_mean': np.mean(times_osqp),
            'osqp_time_std': np.std(times_osqp),
            'our_iters_mean': np.mean(iters_ours),
            'obj_diff_mean': np.mean(obj_diffs),
            'obj_diff_max': np.max(obj_diffs),
            'speedup': np.mean(times_osqp) / np.mean(times_ours) if np.mean(times_ours) > 0 else np.nan
        })
        
        if verbose:
            print(f"  手写算法: {np.mean(times_ours):.4f}s, 迭代: {np.mean(iters_ours):.1f}")
            print(f"  OSQP:     {np.mean(times_osqp):.4f}s")
            print(f"  目标差异: {np.mean(obj_diffs):.2e}")

    return results_all


def generate_latex_tables(results, output_dir):
    """
    生成 LaTeX 表格
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # 表格1: 固定约束数量，变化变量维度
    # ==========================================
    fixed_m_results = [r for r in results if r['test_type'] == 'fixed_m']
    
    latex_table1 = r"""\begin{table}[htbp]
\centering
\caption{固定约束数量 $m=100$ 时，不同变量维度 $n$ 的性能比较}
\label{tab:fixed_m}
\begin{tabular}{ccccccc}
\toprule
$n$ & \multicolumn{2}{c}{有效集法} & \multicolumn{2}{c}{OSQP} & 目标差异 & 加速比 \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & 时间 (s) & 迭代次数 & 时间 (s) & 迭代次数 & (绝对值) & \\
\midrule
"""
    
    for r in fixed_m_results:
        latex_table1 += f"{r['n']} & {r['our_time_mean']:.4f}$\\pm${r['our_time_std']:.4f} & {r['our_iters_mean']:.0f} & "
        latex_table1 += f"{r['osqp_time_mean']:.4f}$\\pm${r['osqp_time_std']:.4f} & {r['osqp_iters_mean']:.0f} & "
        latex_table1 += f"{r['obj_diff_mean']:.2e} & {r['speedup']:.2f} \\\\\n"
    
    latex_table1 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'active_set_fixed_m.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table1)
    
    # ==========================================
    # 表格2: 固定变量维度，变化约束数量
    # ==========================================
    fixed_n_results = [r for r in results if r['test_type'] == 'fixed_n']
    
    latex_table2 = r"""\begin{table}[htbp]
\centering
\caption{固定变量维度 $n=500$ 时，不同约束数量 $m$ 的性能比较}
\label{tab:fixed_n}
\begin{tabular}{ccccccc}
\toprule
$m$ & \multicolumn{2}{c}{有效集法} & \multicolumn{2}{c}{OSQP} & 目标差异 & 加速比 \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
 & 时间 (s) & 迭代次数 & 时间 (s) & 迭代次数 & (绝对值) & \\
\midrule
"""
    
    for r in fixed_n_results:
        latex_table2 += f"{r['m']} & {r['our_time_mean']:.4f}$\\pm${r['our_time_std']:.4f} & {r['our_iters_mean']:.0f} & "
        latex_table2 += f"{r['osqp_time_mean']:.4f}$\\pm${r['osqp_time_std']:.4f} & {r['osqp_iters_mean']:.0f} & "
        latex_table2 += f"{r['obj_diff_mean']:.2e} & {r['speedup']:.2f} \\\\\n"
    
    latex_table2 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'active_set_fixed_n.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table2)
    
    # ==========================================
    # 表格3: 大规模问题测试
    # ==========================================
    large_scale_results = [r for r in results if r['test_type'] == 'large_scale']
    
    latex_table3 = r"""\begin{table}[htbp]
\centering
\caption{大规模问题的性能比较}
\label{tab:large_scale}
\begin{tabular}{cccccccc}
\toprule
$n$ & $m$ & \multicolumn{2}{c}{有效集法} & \multicolumn{2}{c}{OSQP} & 目标差异 & 加速比 \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
 &  & 时间 (s) & 迭代次数 & 时间 (s) & 迭代次数 & (绝对值) & \\
\midrule
"""
    
    for r in large_scale_results:
        latex_table3 += f"{r['n']} & {r['m']} & {r['our_time_mean']:.4f}$\\pm${r['our_time_std']:.4f} & {r['our_iters_mean']:.0f} & "
        latex_table3 += f"{r['osqp_time_mean']:.4f}$\\pm${r['osqp_time_std']:.4f} & {r['osqp_iters_mean']:.0f} & "
        latex_table3 += f"{r['obj_diff_mean']:.2e} & {r['speedup']:.2f} \\\\\n"
    
    latex_table3 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'active_set_large_scale.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table3)
    
    # ==========================================
    # 表格4: 条件数测试
    # ==========================================
    cond_results = [r for r in results if r['test_type'] == 'condition_number']
    
    latex_table4 = r"""\begin{table}[htbp]
\centering
\caption{不同条件数下的算法性能比较 ($n=300$, $m=100$)}
\label{tab:condition_number}
\begin{tabular}{ccccccc}
\toprule
条件数 & \multicolumn{2}{c}{有效集法} & \multicolumn{2}{c}{OSQP} & 目标差异 & 加速比 \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
$\kappa(G)$ & 时间 (s) & 迭代次数 & 时间 (s) & 迭代次数 & (绝对值) & \\
\midrule
"""
    
    for r in cond_results:
        latex_table4 += f"${r['cond_num']:.0e}$ & {r['our_time_mean']:.4f}$\\pm${r['our_time_std']:.4f} & {r['our_iters_mean']:.0f} & "
        latex_table4 += f"{r['osqp_time_mean']:.4f}$\\pm${r['osqp_time_std']:.4f} & -- & "
        latex_table4 += f"{r['obj_diff_mean']:.2e} & {r['speedup']:.2f} \\\\\n"
    
    latex_table4 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'active_set_condition_number.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table4)
    
    # ==========================================
    # 表格5: 综合汇总表
    # ==========================================
    latex_table5 = r"""\begin{table}[htbp]
\centering
\caption{有效集法数值实验综合结果汇总}
\label{tab:summary}
\begin{tabular}{lccccc}
\toprule
测试类别 & 问题规模 & 平均时间 (s) & 平均迭代 & 最大目标差异 & 平均加速比 \\
\midrule
"""
    
    # 汇总各类测试
    for test_type, label in [('fixed_m', '固定约束数量'), ('fixed_n', '固定变量维度'), 
                              ('large_scale', '大规模问题'), ('condition_number', '条件数测试')]:
        type_results = [r for r in results if r['test_type'] == test_type]
        if type_results:
            avg_time = np.mean([r['our_time_mean'] for r in type_results])
            avg_iters = np.mean([r['our_iters_mean'] for r in type_results])
            max_diff = np.max([r['obj_diff_max'] for r in type_results])
            avg_speedup = np.mean([r['speedup'] for r in type_results if not np.isnan(r['speedup'])])
            
            if test_type == 'fixed_m':
                scale = f"$n\\in[50,2000]$, $m=100$"
            elif test_type == 'fixed_n':
                scale = f"$n=500$, $m\\in[50,500]$"
            elif test_type == 'large_scale':
                scale = f"$n\\in[1000,3000]$"
            else:
                scale = f"$\\kappa\\in[10,10^4]$"
            
            latex_table5 += f"{label} & {scale} & {avg_time:.4f} & {avg_iters:.0f} & {max_diff:.2e} & {avg_speedup:.2f} \\\\\n"
    
    latex_table5 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'active_set_summary.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table5)
    
    # ==========================================
    # 表格6: 约束满足情况
    # ==========================================
    latex_table6 = r"""\begin{table}[htbp]
\centering
\caption{约束满足情况比较（最大约束违反量）}
\label{tab:constraint_violation}
\begin{tabular}{ccccc}
\toprule
$n$ & $m$ & 有效集法 & OSQP & 测试类型 \\
\midrule
"""
    
    for r in results:
        if 'our_violation_max' in r and r['our_violation_max'] is not None:
            test_type_label = {'fixed_m': '固定$m$', 'fixed_n': '固定$n$', 
                              'large_scale': '大规模', 'condition_number': '条件数'}
            latex_table6 += f"{r['n']} & {r['m']} & {r['our_violation_max']:.2e} & {r['osqp_violation_max']:.2e} & {test_type_label.get(r['test_type'], r['test_type'])} \\\\\n"
    
    latex_table6 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'active_set_constraint_violation.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_table6)
    
    print(f"\nLaTeX 表格已保存到: {output_dir}")
    print("生成的文件:")
    print("  - active_set_fixed_m.tex (固定约束数量测试)")
    print("  - active_set_fixed_n.tex (固定变量维度测试)")
    print("  - active_set_large_scale.tex (大规模问题测试)")
    print("  - active_set_condition_number.tex (条件数测试)")
    print("  - active_set_summary.tex (综合汇总表)")
    print("  - active_set_constraint_violation.tex (约束满足情况)")


def run_quick_demo():
    """
    快速演示：原有的单次测试
    """
    N_VARS = 2000
    N_CONSTR = 500
    
    print(f"正在生成 {N_VARS} 维随机凸 QP 问题...")
    G, c, A, b, x0 = generate_random_qp(n=N_VARS, m=N_CONSTR)
    
    # 1. 使用我们手写的有效集法求解
    start_time = time.time()
    x_ours, iters, status = solve_active_set_qp(G, c, A, b, x0)
    our_time = time.time() - start_time
    
    our_obj = 0.5 * x_ours.T @ G @ x_ours + c.T @ x_ours
    
    print(f"\n[有效集法] 结果:")
    print(f"状态: {status}")
    print(f"迭代次数: {iters}")
    print(f"耗时: {our_time:.4f} 秒")
    print(f"目标函数值: {our_obj:.6f}")
    
    # 2. 使用 OSQP 进行验证对比
    print(f"\n[OSQP] 验证中...")
    
    P = sparse.csc_matrix(G)
    q = c
    A_osqp = sparse.csc_matrix(-A)
    u_osqp = -b
    l_osqp = np.full(N_CONSTR, -np.inf)
    
    prob = osqp.OSQP()
    prob.setup(P, q, A_osqp, l_osqp, u_osqp, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    
    start_time = time.time()
    res = prob.solve()
    osqp_time = time.time() - start_time
    
    osqp_obj = 0.5 * res.x.T @ G @ res.x + c.T @ res.x
    
    print(f"状态: {res.info.status}")
    print(f"耗时: {osqp_time:.4f} 秒")
    print(f"目标函数值: {osqp_obj:.6f}")
    
    # 3. 结果对比
    diff = abs(our_obj - osqp_obj)
    print(f"\n目标函数值差异: {diff:.2e}")
    if diff < 1e-4:
        print(">> 验证成功：手写算法结果与 OSQP 一致！")
    else:
        print(">> 验证失败：结果偏差较大，请检查。")


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    import sys
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    latex_output_dir = os.path.join(parent_dir, 'latex', 'tables')
    
    print("=" * 70)
    print("有效集法 (Active Set Method) 数值实验")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 命令行参数选择模式
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("\n运行模式: 快速演示\n")
        run_quick_demo()
    else:
        print("\n运行模式: 完整数值实验\n")
        print("提示: 使用 '--quick' 参数可运行快速演示\n")
        
        # 运行完整测试套件
        results = run_benchmark_suite(verbose=True)
        
        # 生成 LaTeX 表格
        generate_latex_tables(results, latex_output_dir)
        
        print("\n" + "=" * 70)
        print("数值实验完成！")
        print("=" * 70)