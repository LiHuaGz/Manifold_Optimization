import os

# --- 优化 1: 必须在 import numpy 之前设置环境变量 ---
# 这样才能确保 OpenBLAS/MKL 正确读取配置
num_cores = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_cores)
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time

print(f"检测到 CPU 核心数: {num_cores}, 线程环境已配置。")

# 配置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def stiefel_retraction(X, Z, svd=True):
    # X: 当前点,Z: 切空间方向
    if svd:
        U, _, Vt = np.linalg.svd(X + Z, full_matrices=False)
        return U @ Vt
    else:
        # QR分解
        Q, _ = np.linalg.qr(X + Z)
        return Q

def compute_f_and_AX(X, A, B):
    """
    【计算优化核心】
    利用 A 的对称性计算 AX。
    """
    AX = A @ X  # np.dot(A, X) 也可以，因为 A 是对称的全矩阵
    
    val = np.sum(X * AX) + 2 * np.sum(B * X)
    return val, AX

class Stiefel_Optimizer:
    def __init__(self, n, p, A, B):
        self.n = n
        self.p = p
        self.A = A
        self.B = B

    def compute_grad_from_AX(self, X, AX):
        """利用已知的 AX 计算梯度"""
        return 2 * (AX + self.B)

    def project_grad_stiefel(self, X, G):
        """
        投影梯度到切空间。
        利用 X.T @ G 是 p x p 矩阵的特性优化。
        """
        XTG = X.T @ G
        # sym 是对称阵，(XTG + XTG.T) / 2
        sym = (XTG + XTG.T) / 2
        return G - X @ sym

    def gradient_descent(self, X_init, max_iters=1000, tol=1e-6, num_history=2, search_method='armijo', alpha_min=1e-5, alpha_max=1e5, t_init_guess=1.0):
        
        # 初始化
        X = X_init
        f_val, AX = compute_f_and_AX(X, self.A, self.B)
        
        f_vals = []
        f_vals.append(f_val)
        
        # 记录 BB 步长所需的旧变量
        X_prev = None
        grad_proj_prev = None
        
        t = t_init_guess
        
        for i in range(max_iters):
            # 1. 计算梯度 (利用缓存的 AX)
            grad = self.compute_grad_from_AX(X, AX)
            
            # 2. 投影梯度
            grad_proj = self.project_grad_stiefel(X, grad)
            
            # 3. 收敛性检查 (Frobenius norm)
            grad_norm_sq = np.sum(grad_proj**2) # 比 linalg.norm 快，且平方和用于 Armijo
            grad_norm = np.sqrt(grad_norm_sq)
            
            if grad_norm < tol:
                print(f"Converged at iter {i}, grad_norm: {grad_norm:.2e}")
                break

            # 4. 确定步长 (BB Step or Constant/Armijo init)
            if search_method == 'BB_step' and i > 0:
                # 向量化计算 s 和 y 的点积，避免 flatten() 的拷贝
                s = X - X_prev
                y = grad_proj - grad_proj_prev
                
                ss = np.sum(s * s)
                sy = np.abs(np.sum(s * y))
                yy = np.sum(y * y)
                
                # Barzilai-Borwein step sizes
                if i % 2 == 0:
                    alpha = ss / sy if sy > 1e-10 else t
                else:
                    alpha = sy / yy if yy > 1e-10 else t
                
                # 截断
                t = min(alpha_max, max(alpha_min, alpha))
            else:
                t = t_init_guess # 重置初始步长猜测

            # 5. 线搜索 (Non-monotone Armijo)
            # 记录历史最大值
            start_idx = max(0, len(f_vals) - num_history)
            f_ref = max(f_vals[start_idx:])
            
            # Armijo 参数
            rho = 0.5
            c = 1e-4
            
            step_found = False
            for _ in range(20): # max line search iters
                # 使用 QR Retraction
                X_new = stiefel_retraction(X, -t * grad_proj)
                
                # 计算新点的函数值 (同时拿到新的 AX_new)
                f_new, AX_new = compute_f_and_AX(X_new, self.A, self.B)
                
                if f_new <= f_ref - c * t * grad_norm_sq:
                    step_found = True
                    break
                t *= rho
            
            # 更新变量
            if not step_found:
                # 极其罕见的情况，步长缩减到极小
                X_new = X
                AX_new = AX
                f_new = f_val
            
            # 保存用于下一次 BB 步长计算
            X_prev = X
            grad_proj_prev = grad_proj
            
            # 更新当前点
            X = X_new
            AX = AX_new # 关键：传递缓存的矩阵乘积结果
            f_val = f_new
            f_vals.append(f_val)

        return X, f_vals
    
    def L_BFGS(self, X_init, max_iters=1000, tol=1e-6, num_history=2, search_method='armijo', alpha_min=1e-5, alpha_max=1e5, t_init_guess=1.0, m=10):
        '''
        Stiefel 流形上的 L-BFGS 方法 (修正版)
        '''
        def L_BFGS_two_loop_recursion(grad_proj, diff_history, grad_history):
            if len(diff_history) == 0:
                return -grad_proj
            
            # 确保分母不为0
            denom = np.sum(grad_history[-1] * grad_history[-1])
            if denom < 1e-20:
                gamma = 1.0
            else:
                gamma = np.sum(diff_history[-1] * grad_history[-1]) / denom

            q = grad_proj.copy()
            alpha_list = [None] * len(diff_history)
            rho_list = [None] * len(diff_history)
            
            for i in reversed(range(len(diff_history))):
                s, y = diff_history[i], grad_history[i]
                sy = np.sum(y * s)
                if abs(sy) < 1e-20: # 安全检查
                    rho = 1.0
                else:
                    rho = 1.0 / sy
                    
                alpha = rho * np.sum(s * q)
                alpha_list[i], rho_list[i] = alpha, rho
                q -= alpha * y
                
            r = q * gamma
            for i in range(len(diff_history)):
                s, y = diff_history[i], grad_history[i]
                rho = rho_list[i]
                beta = rho * np.sum(y * r)
                alpha = alpha_list[i]
                r += s * (alpha - beta)
            return -r

        # 初始化
        X = X_init
        f_val, AX = compute_f_and_AX(X, self.A, self.B)
        f_vals = [f_val]
        
        # 状态变量
        s_last = None         # 存储上一步的 s
        grad_proj_last = None # 存储上一步的梯度
        
        diff_history, grad_history = [], []
        t = t_init_guess
        
        for i in range(max_iters):
            # 1. 计算当前梯度
            grad = self.compute_grad_from_AX(X, AX)
            grad_proj = self.project_grad_stiefel(X, grad)
            
            # 2. 延迟更新历史信息 (在获得新梯度后，配对 s_{k-1} 和 y_{k-1})
            if s_last is not None and grad_proj_last is not None:
                # y_{k-1} = grad(X_k) - grad(X_{k-1})
                y_last = grad_proj - grad_proj_last
                
                # 曲率条件检查 (s^T y > 0)
                sy = np.sum(y_last * s_last)
                if sy > 1e-10: # 只有当曲率满足正定条件时才更新
                    if len(diff_history) >= m:
                        diff_history.pop(0)
                        grad_history.pop(0)
                    diff_history.append(s_last)
                    grad_history.append(y_last)
            
            # 3. 收敛性检查
            grad_norm_sq = np.sum(grad_proj**2)
            grad_norm = np.sqrt(grad_norm_sq)
            
            if grad_norm < tol:
                print(f"Converged at iter {i}, grad_norm: {grad_norm:.2e}")
                break
            
            # 4. 计算搜索方向
            search_direction = L_BFGS_two_loop_recursion(grad_proj, diff_history, grad_history)
            
            # L-BFGS 通常默认步长为 1.0，不需要 BB step 调整初始步长，除非为了特殊目的
            # 这里重置为 1.0 (或者 t_init_guess)
            t = 1.0 

            # 5. 线搜索 (Armijo)
            start_idx = max(0, len(f_vals) - num_history)
            f_ref = max(f_vals[start_idx:])
            rho = 0.5
            c = 1e-4
            
            step_found = False
            for _ in range(20):
                # 尝试更新
                try:
                    X_new = stiefel_retraction(X, t * search_direction)
                except np.linalg.LinAlgError:
                    t *= rho
                    continue

                f_new, AX_new = compute_f_and_AX(X_new, self.A, self.B)
                
                if f_new <= f_ref - c * t * grad_norm_sq: # 注意：严格来说Armijo应用方向导数，这里简化用 grad_norm_sq 近似
                    step_found = True
                    break
                t *= rho
            
            if not step_found:
                X_new = X
                AX_new = AX
                f_new = f_val
                # 步长过小导致无法更新，通常意味着收敛或陷入死角，清空历史重试是个好策略
                diff_history, grad_history = [], [] 
            
            # 6. 保存用于下一次迭代的状态
            # 记录 s_k = X_{k+1} - X_k
            s_last = X_new - X
            # 记录 grad_k (注意：这里保存的是投影梯度)
            grad_proj_last = grad_proj
            
            # 更新变量
            X = X_new
            AX = AX_new
            f_val = f_new
            f_vals.append(f_val)

        return X, f_vals

def main():
    n, p = 500, 5
    print(f"Matrix Size: {n}x{n}, Stiefel Manifold St({n}, {p})")
    
    # 构造数据
    np.random.seed(42)
    A_raw = np.random.rand(n, n)
    A = A_raw.T @ A_raw
    B = np.zeros((n, p)) # B 设为非零更有趣，不过保持原逻辑
    
    x0, _ = np.linalg.qr(np.random.randn(n, p))
    
    optimizer = Stiefel_Optimizer(n, p, A, B)
    
    
    # --- 1. Armijo ---
    start = time.time()
    _, f_vals_armijo = optimizer.gradient_descent(x0, search_method='armijo', max_iters=2000)
    print(f"Armijo Time: {time.time() - start:.4f}s, min f: {min(f_vals_armijo):.6f}")
    
    # --- 2. BB Step ---
    start = time.time()
    _, f_vals_bb = optimizer.gradient_descent(x0, search_method='BB_step', max_iters=2000)
    print(f"BB Step Time: {time.time() - start:.4f}s, min f: {min(f_vals_bb):.6f}")

    # --- 3. L-BFGS ---
    start = time.time()
    _, f_vals_lbfgs = optimizer.L_BFGS(x0, search_method='BB_step', max_iters=2000, m=10)
    print(f"L-BFGS Time: {time.time() - start:.4f}s, min f: {min(f_vals_lbfgs):.6f}")
    
    # 绘图比较
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_vals_armijo, label='Armijo')
    plt.semilogy(f_vals_bb, label='BB Step')
    plt.semilogy(f_vals_lbfgs, label='L-BFGS')
    plt.legend()
    plt.title('Optimization Convergence')
    plt.show()

if __name__ == "__main__":
    main()