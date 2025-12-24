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
# 主程序：测试几百维的数据
# ==========================================
if __name__ == "__main__":
    # 设置规模：1000个变量，50个约束
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
    
    # OSQP 求解: min 1/2 x^T P x + q^T x, s.t. l <= Ax <= u
    # 转换为 OSQP 格式: Ax >= b 等价于 -Ax <= -b
    P = sparse.csc_matrix(G)
    q = c
    A_osqp = sparse.csc_matrix(-A)
    u_osqp = -b
    l_osqp = np.full(N_CONSTR, -np.inf)
    
    # 创建 OSQP 对象
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