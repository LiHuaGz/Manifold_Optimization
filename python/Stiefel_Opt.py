import os
import time
from collections import deque

# --- 1. 关键优化: 在导入 numpy 之前配置线程环境 ---
# 许多后端(MKL, OpenBLAS)只在加载时读取这些变量
num_cores = os.cpu_count() or 1
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_cores)
# 限制 PyTorch 等其他库抢占资源 (如果有的话)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(num_cores)

import numpy as np
import matplotlib.pyplot as plt

print(f"检测到 CPU 核心数: {num_cores}, 线程环境已配置。")

# 配置绘图
plt.rcParams.update(plt.rcParamsDefault)

class StiefelManifold:
    """
    Stiefel 流形基础运算工具类
    """
    @staticmethod
    def retraction(X, Z, method='svd'):
        """将切空间向量 Z 映射回流形"""
        # 保持 SVD 以保证二阶收敛性
        # full_matrices=False 对瘦长矩阵非常关键
        U, _, Vt = np.linalg.svd(X + Z, full_matrices=False)
        return U @ Vt

    @staticmethod
    def project_gradient(X, G):
        """
        将欧氏梯度 G 投影到 X 处的切空间
        Proj(G) = G - X @ sym((X.T @ G))
        """
        # 这一步计算是 O(N*P^2)，相对于 Cost 计算很快
        XTG = X.T @ G
        sym = (XTG + XTG.T) * 0.5
        return G - X @ sym


class QuadraticProblem:
    """
    定义问题: min Tr(X.T A X) + 2 Tr(B.T X)
    """
    def __init__(self, n, p, A, B):
        self.n = n
        self.p = p
        self.A = A
        self.B = B

    def compute_cost_and_cache(self, X):
        """计算 f(X) 并返回中间变量 AX 以供梯度计算使用"""
        # 瓶颈操作: O(N^2 * P)
        AX = self.A @ X
        
        # 优化 1: 使用 vdot 替代 einsum/sum+multiply
        # np.vdot 会将数组视为扁平向量计算内积，利用 BLAS 极快
        # Cost = Tr(X.T @ AX) + 2 Tr(B.T @ X)
        term1 = np.vdot(X, AX)
        term2 = 2.0 * np.vdot(self.B, X)
        
        return term1 + term2, AX

    def compute_euclidean_grad(self, X, AX):
        """利用缓存的 AX 计算欧氏梯度"""
        # G = 2 * (AX + B)
        # 优化 2: 减少临时内存分配
        # 显式创建 G，然后就地乘法
        G = AX + self.B
        G *= 2.0
        return G


class LineSearch:
    """
    非单调 Armijo 线搜索
    """
    def __init__(self, problem, num_history=2, rho=0.5, c=1e-4, max_ls_iters=20):
        self.problem = problem
        self.history = []
        self.num_history = num_history
        self.rho = rho
        self.c = c
        self.max_ls_iters = max_ls_iters

    def search(self, X, f_val, AX, grad_proj, search_dir, initial_step, grad_norm_sq):
        self.history.append(f_val)
        if len(self.history) > self.num_history:
            self.history.pop(0)
        f_ref = max(self.history)

        t = initial_step
        
        # 预先计算方向导数项: c * <grad, dir>
        # 这里的 dot product 也是 O(NP)
        grad_dot_dir = np.vdot(grad_proj, search_dir)
        descent_factor = self.c * grad_dot_dir
        
        for _ in range(self.max_ls_iters):
            try:
                # 尝试更新点 (SVD 开销)
                X_new = StiefelManifold.retraction(X, t * search_dir)
                
                # 计算新 Cost (矩阵乘法开销)
                f_new, AX_new = self.problem.compute_cost_and_cache(X_new)
                
                if f_new <= f_ref + t * descent_factor:
                    return X_new, f_new, AX_new, t, True
                
            except np.linalg.LinAlgError:
                pass
            
            t *= self.rho

        return X, f_val, AX, 0.0, False

# --- 策略模块: 搜索方向 ---

class DirectionStrategy:
    def compute_direction(self, grad_proj):
        raise NotImplementedError
    
    def update(self, s, y):
        pass

class SteepestDescent(DirectionStrategy):
    def compute_direction(self, grad_proj):
        return -grad_proj

class LBFGS(DirectionStrategy):
    def __init__(self, m=10):
        self.m = m
        # 优化 3: 使用 deque 高效管理队列，并存储 (s, y, rho) 元组
        # 避免每次迭代重新计算 rho
        self.memory = deque(maxlen=m)

    def compute_direction(self, grad_proj):
        if not self.memory:
            return -grad_proj

        q = grad_proj.copy()
        alphas = [0.0] * len(self.memory)
        
        # Backward pass
        # memory 是从旧到新存储，reversed 遍历即从新到旧
        for i, (s, y, rho) in enumerate(reversed(self.memory)):
            # 优化 4: 使用 vdot
            sq = np.vdot(s, q)
            alpha = rho * sq
            alphas[i] = alpha
            q -= alpha * y
            
        # Scaling
        # 使用最新的一对数据进行初始 Hessian 缩放
        s_last, y_last, rho_last = self.memory[-1]
        # rho = 1/sy, so sy = 1/rho
        sy_last = 1.0 / rho_last if abs(rho_last) > 1e-20 else 1.0
        yy_last = np.vdot(y_last, y_last)
        gamma = sy_last / yy_last if yy_last > 1e-20 else 1.0
        
        r = q # Alias
        r *= gamma # In-place scaling
        
        # Forward pass
        # alphas 列表也是按 reversed 顺序填充的 (最新在 0)，所以正常遍历 memory (旧->新) 
        # 对应的 alpha 需要反向取出
        for (s, y, rho), alpha in zip(self.memory, reversed(alphas)):
            yr = np.vdot(y, r)
            beta = rho * yr
            r += s * (alpha - beta)
            
        return -r

    def update(self, s, y):
        # 优化: 仅在此处计算一次 inner product
        sy = np.vdot(s, y)
        if sy > 1e-10:
            rho = 1.0 / sy
            # deque 会自动处理溢出
            self.memory.append((s, y, rho))

class DampedLBFGS(LBFGS):
    def __init__(self, m=10, delta=1.0):
        # 初始化父类，利用其 memory 结构
        super().__init__(m)
        self.delta = delta 

    def update(self, s, y):
        sy = np.vdot(s, y)
        ss = np.vdot(s, s)
        sBs = self.delta * ss

        if sBs < 1e-20:
            return

        if sy >= 0.25 * sBs:
            theta = 1.0
        else:
            theta = (0.75 * sBs) / (sBs - sy)

        # 这里的 r 就是 damped y
        r = theta * y + (1.0 - theta) * (self.delta * s)
        
        # 重新计算 r 和 s 的内积作为新的 denominator
        sr = np.vdot(s, r)
        
        if sr > 1e-10:
            rho = 1.0 / sr
            self.memory.append((s, r, rho))

class SubspaceLBFGS(DirectionStrategy):
    """
    Subspace L-BFGS (保持逻辑优化)
    """
    def __init__(self, max_dim=20, delta=1.0):
        self.max_dim = max_dim
        self.delta = delta
        self.Z_buffer = None 
        self.dim = 0
        self.H_sub = None
        self.g_prev_flat = None

    def _reset(self, g_flat):
        gnorm = np.linalg.norm(g_flat)
        if gnorm < 1e-20: gnorm = 1.0
        
        if self.Z_buffer is None or self.Z_buffer.shape[0] != g_flat.size:
            self.Z_buffer = np.zeros((g_flat.size, self.max_dim), dtype=g_flat.dtype)
        
        self.Z_buffer[:, 0] = g_flat / gnorm
        self.dim = 1
        self.H_sub = np.eye(1) * (1.0 / self.delta)

    def compute_direction(self, grad_proj):
        g_flat = grad_proj.reshape(-1)
        if self.dim == 0: self._reset(g_flat)
        self.g_prev_flat = g_flat.copy()

        Z_active = self.Z_buffer[:, :self.dim]
        # vdot 优化: Matrix-Vector 乘法内部已优化，此处显式转置
        g_sub = Z_active.T @ g_flat
        p_sub = -self.H_sub @ g_sub
        p_flat = Z_active @ p_sub
        
        return p_flat.reshape(grad_proj.shape)

    def update(self, s, y):
        if self.dim == 0 or self.g_prev_flat is None: return

        s_flat = s.reshape(-1)
        y_flat = y.reshape(-1)
        
        g_next_flat = y_flat + self.g_prev_flat
        Z_active = self.Z_buffer[:, :self.dim]
        
        ZTg = Z_active.T @ g_next_flat
        u = g_next_flat - Z_active @ ZTg
        u_norm = np.linalg.norm(u)

        if u_norm > 1e-6 and self.dim < self.max_dim:
            self.Z_buffer[:, self.dim] = u / u_norm
            new_dim = self.dim + 1
            H_new = np.eye(new_dim) * (1.0 / self.delta)
            H_new[:self.dim, :self.dim] = self.H_sub
            self.H_sub = H_new
            self.dim = new_dim
        elif self.dim >= self.max_dim:
            self.dim = 0
            return

        Z_active = self.Z_buffer[:, :self.dim]
        s_sub = Z_active.T @ s_flat
        y_sub = Z_active.T @ y_flat

        sy_sub = np.dot(s_sub, y_sub)
        if sy_sub > 1e-10:
            rho = 1.0 / sy_sub
            I = np.eye(self.dim)
            # 子空间维度很小 (20x20)，这里的 outer 和 matmul 开销忽略不计
            V = I - rho * np.outer(s_sub, y_sub)
            self.H_sub = V @ self.H_sub @ V.T + rho * np.outer(s_sub, s_sub)

# --- 策略模块: 步长选择 ---

class StepSizeStrategy:
    def get_initial_step(self, current_t, iter_idx, s=None, y=None):
        return 1.0

class FixedStep(StepSizeStrategy):
    def __init__(self, initial_guess=1.0):
        self.guess = initial_guess

    def get_initial_step(self, current_t, iter_idx, s=None, y=None):
        return self.guess

class BBStep(StepSizeStrategy):
    def __init__(self, alpha_min=1e-5, alpha_max=1e5):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def get_initial_step(self, current_t, iter_idx, s=None, y=None):
        if iter_idx == 0 or s is None or y is None:
            return 1.0
        
        # 优化: 使用 vdot
        ss = np.vdot(s, s)
        sy = abs(np.vdot(s, y)) # 取绝对值保证正定性猜测
        yy = np.vdot(y, y)
        
        if iter_idx % 2 == 0:
            alpha = ss / sy if sy > 1e-10 else current_t
        else:
            alpha = sy / yy if yy > 1e-10 else current_t
            
        return min(self.alpha_max, max(self.alpha_min, alpha))


# --- 统一求解器 ---

class StiefelSolver:
    def __init__(self, n, p, A, B):
        self.problem = QuadraticProblem(n, p, A, B)
        self.n = n
        self.p = p

    def solve(self, X_init, direction_strategy, step_strategy, max_iters=1000, tol=1e-6):
        X = X_init
        f_val, AX = self.problem.compute_cost_and_cache(X)
        f_vals = [f_val]
        
        grad_proj_prev = None
        X_prev = None
        current_step = 1.0
        
        initial_grad_norm = None
        
        ls = LineSearch(self.problem)
        strategy_name = direction_strategy.__class__.__name__
        print(f"Start Opt: Dir={strategy_name}, Step={step_strategy.__class__.__name__}")

        grad = self.problem.compute_euclidean_grad(X, AX)
        grad_proj = StiefelManifold.project_gradient(X, grad)
        grad_norm_sq = np.vdot(grad_proj, grad_proj)
        
        for i in range(max_iters):
            grad_norm = np.sqrt(grad_norm_sq)
            
            if initial_grad_norm is None:
                initial_grad_norm = grad_norm
                print(f"  Init Grad Norm: {initial_grad_norm:.2e}")
            
            # 收敛判定
            if grad_norm < tol or grad_norm < tol * initial_grad_norm:
                print(f"  Converged at iter {i}, grad_norm: {grad_norm:.2e}")
                break

            s, y = None, None
            if i > 0:
                s = X - X_prev
                y = grad_proj - grad_proj_prev
                direction_strategy.update(s, y)

            t_guess = step_strategy.get_initial_step(current_step, i, s, y)
            direction = direction_strategy.compute_direction(grad_proj)

            X_new, f_new, AX_new, t_actual, success = ls.search(
                X, f_val, AX, grad_proj, direction, t_guess, grad_norm_sq
            )

            if not success:
                # 策略重启
                if isinstance(direction_strategy, LBFGS) and len(direction_strategy.memory) > 0:
                    print(f"  Line search stuck at iter {i}. Restarting L-BFGS...")
                    direction_strategy.memory.clear()
                    direction = -grad_proj
                    X_new, f_new, AX_new, t_actual, success = ls.search(
                        X, f_val, AX, grad_proj, direction, 1.0, grad_norm_sq
                    )
                
                if not success:
                    print(f"  Line search failed completely at iter {i}")
                    break

            X_prev = X
            grad_proj_prev = grad_proj
            
            X = X_new
            f_val = f_new
            AX = AX_new
            current_step = t_actual
            
            grad = self.problem.compute_euclidean_grad(X, AX)
            grad_proj = StiefelManifold.project_gradient(X, grad)
            grad_norm_sq = np.vdot(grad_proj, grad_proj)
            
            f_vals.append(f_val)
            
            if i % 100 == 0:
                print(f"  Iter {i}: f={f_val:.4e}, |g|={grad_norm:.2e}")

        return X, f_vals

def main():
    n, p = 500, 5
    print(f"Matrix Size: {n}x{n}, Stiefel Manifold St({n}, {p})")
    
    # 构造数据
    np.random.seed(42)
    # 保证 A 为对称阵，符合问题定义
    A_raw = np.random.rand(n, n)
    A = A_raw.T @ A_raw
    B = np.zeros((n, p))
    
    # 随机初始化正交矩阵
    x0, _ = np.linalg.qr(np.random.randn(n, p))
    
    solver = StiefelSolver(n, p, A, B)
    
    # 运行配置
    strategies = [
        (SteepestDescent(), FixedStep(1.0), "Armijo"),
        (SteepestDescent(), BBStep(), "BB Step"),
        (LBFGS(m=10), FixedStep(1.0), "L-BFGS"),
        (DampedLBFGS(m=10, delta=20.0), FixedStep(1.0), "Damped L-BFGS"),
        (SubspaceLBFGS(max_dim=20, delta=1.0), FixedStep(1.0), "Subspace L-BFGS")
    ]
    
    results = {}
    
    for direction_strat, step_strat, name in strategies:
        start = time.time()
        # 传入 x0 的副本，确保每次从同一起点开始
        _, f_vals = solver.solve(x0.copy(), direction_strat, step_strat, max_iters=2000)
        elapsed = time.time() - start
        min_f = min(f_vals) if f_vals else float('inf')
        results[name] = f_vals
        print(f"{name} Time: {elapsed:.4f}s, min f: {min_f:.6f}\n")

    # 绘图比较
    plt.figure(figsize=(10, 6))
    for name, f_vals in results.items():
        style = '--' if 'Damped' in name else ('-.' if 'Subspace' in name else '-')
        plt.semilogy(f_vals, label=name, linestyle=style, linewidth=1.5)
    
    plt.legend()
    plt.title(f'Optimization Convergence Comparison (n={n}, p={p})')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Value (log scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()