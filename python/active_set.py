import numpy as np
import osqp
from scipy import sparse
import time

def solve_active_set_qp(G, c, A, b, x0, tol=1e-6, max_iter=1000):
    """
    有效集法求解: min 1/2 x^T G x + c^T x, s.t. Ax >= b
    """
    n_vars = len(c)
    n_constraints = len(b)
    
    x = np.array(x0, dtype=float)
    
    # 初始工作集：找出当前所有 active 的约束 (Ax = b)
    # 注意：为了数值稳定性，使用较宽松的 tol
    residuals = A @ x - b
    working_set = [i for i in range(n_constraints) if abs(residuals[i]) < tol]
    
    for k in range(max_iter):
        # 1. 构造 KKT 系统求解 p
        g = G @ x + c
        
        # 构造 KKT 矩阵
        # [ G   -A_w^T ] [ p      ] = [ -g ]
        # [ A_w    0   ] [ lambda ]   [  0 ]
        
        if len(working_set) > 0:
            A_w = A[working_set]
            n_active = len(working_set)
            
            top = np.hstack([G, -A_w.T])
            bot = np.hstack([A_w, np.zeros((n_active, n_active))])
            KKT = np.vstack([top, bot])
            rhs = np.hstack([-g, np.zeros(n_active)])
            
            try:
                sol = np.linalg.solve(KKT, rhs)
                p = sol[:n_vars]
                lambdas = sol[n_vars:]
            except np.linalg.LinAlgError:
                # 遇到奇异矩阵，通常是因为约束线性相关，这里简单处理：剔除最后一个约束
                working_set.pop()
                continue
        else:
            # 无约束情况
            p = np.linalg.solve(G, -g)
            lambdas = []

        # 2. 检查 p 是否足够小 (是否收敛到子问题最优)
        if np.linalg.norm(p) < tol:
            # 检查拉格朗日乘子
            if len(working_set) == 0:
                return x, k, "Optimal (Unconstrained)"
            
            min_lambda_idx = np.argmin(lambdas)
            min_lambda = lambdas[min_lambda_idx]
            
            if min_lambda >= -tol:
                return x, k, "Optimal (KKT Satisfied)"
            else:
                # 剔除乘子为负的约束
                # print(f"Iter {k}: Drop constraint {working_set[min_lambda_idx]}")
                del working_set[min_lambda_idx]
        else:
            # 3. 计算步长 alpha
            alpha = 1.0
            blocking_idx = -1
            
            # 仅检查不在工作集中的约束
            # 我们要保证 A_i(x + alpha*p) >= b_i
            # 即 alpha * A_i * p >= b_i - A_i * x
            for i in range(n_constraints):
                if i not in working_set:
                    ap = np.dot(A[i], p)
                    if ap < -tol: # 只有朝着边界走才需要限制
                        dist = (b[i] - np.dot(A[i], x)) / ap
                        if dist < alpha:
                            alpha = dist
                            blocking_idx = i
            
            # 防止数值误差导致的极小步长
            alpha = max(0.0, alpha)
            
            # 更新 x
            x = x + alpha * p
            
            if alpha < 1.0:
                # print(f"Iter {k}: Hit constraint {blocking_idx}")
                working_set.append(blocking_idx)
                
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
    N_VARS = 1000
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