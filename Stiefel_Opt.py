import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- 环境配置 ---
num_cores = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_cores)

print(f"检测到 CPU 核心数: {num_cores}, 线程环境已配置。")

# 配置绘图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 配置绘图 (使用默认英文字体，避免兼容性问题)
plt.rcParams.update(plt.rcParamsDefault)
# 或者直接删除上面那两行 SimHei 的配置


class StiefelManifold:
    """
    Stiefel 流形基础运算工具类
    """
    @staticmethod
    def retraction(X, Z, method='svd'):
        """将切空间向量 Z 映射回流形"""
        if method == 'svd':
            U, _, Vt = np.linalg.svd(X + Z, full_matrices=False)
            return U @ Vt
        else:
            Q, _ = np.linalg.qr(X + Z)
            return Q

    @staticmethod
    def project_gradient(X, G):
        """
        将欧氏梯度 G 投影到 X 处的切空间
        Proj(G) = G - X @ sym((X.T @ G))
        """
        XTG = X.T @ G
        sym = (XTG + XTG.T) / 2
        return G - X @ sym


class QuadraticProblem:
    """
    定义问题: min Tr(X.T A X) + 2 Tr(B.T X)
    负责计算函数值、梯度，并缓存 AX 矩阵以提高效率。
    """
    def __init__(self, n, p, A, B):
        self.n = n
        self.p = p
        self.A = A
        self.B = B

    def compute_cost_and_cache(self, X):
        """计算 f(X) 并返回中间变量 AX 以供梯度计算使用"""
        AX = self.A @ X
        # 优化计算: Tr(X.T A X) = sum(X * AX)
        val = np.sum(X * AX) + 2 * np.sum(self.B * X)
        return val, AX

    def compute_euclidean_grad(self, X, AX):
        """利用缓存的 AX 计算欧氏梯度"""
        return 2 * (AX + self.B)


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
        """
        执行线搜索
        Returns: X_new, f_new, AX_new, step_size, success
        """
        # 记录历史最大值用于非单调搜索
        self.history.append(f_val)
        if len(self.history) > self.num_history:
            self.history.pop(0)
        f_ref = max(self.history)

        t = initial_step
        
        for _ in range(self.max_ls_iters):
            try:
                # 尝试更新点
                X_new = StiefelManifold.retraction(X, t * search_dir)
                f_new, AX_new = self.problem.compute_cost_and_cache(X_new)
                
                # Armijo 条件 (使用 grad_norm_sq 近似方向导数，假设方向为下降方向)
                # 严格来说应为 c * t * <grad, dir>，在 GD 中 <grad, -grad> = -norm^2
                # 在 LBFGS 中需保证方向下降性。这里沿用原始逻辑。
                descent_term = self.c * t * np.sum(grad_proj * search_dir)
                
                # 如果是 GD, search_dir = -grad_proj, descent_term = -c * t * norm_sq
                # 为了兼容通用性，直接计算点积
                if f_new <= f_ref + descent_term:
                    return X_new, f_new, AX_new, t, True
                
            except np.linalg.LinAlgError:
                pass # SVD 可能失败，虽罕见，缩小步长重试
            
            t *= self.rho

        # 线搜索失败，保持原位
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
        self.s_history = []
        self.y_history = []

    def compute_direction(self, grad_proj):
        # 双循环递归 (Two-loop recursion)
        if not self.s_history:
            return -grad_proj

        q = grad_proj.copy()
        alphas = []
        
        # Backward pass
        for s, y in zip(reversed(self.s_history), reversed(self.y_history)):
            rho = 1.0 / np.sum(y * s) if np.abs(np.sum(y * s)) > 1e-20 else 1.0
            alpha = rho * np.sum(s * q)
            alphas.append(alpha)
            q -= alpha * y
            
        # Scaling
        s_last, y_last = self.s_history[-1], self.y_history[-1]
        gamma = np.sum(s_last * y_last) / np.sum(y_last * y_last) if np.sum(y_last * y_last) > 1e-20 else 1.0
        r = gamma * q
        
        # Forward pass
        for s, y, alpha in zip(self.s_history, self.y_history, reversed(alphas)):
            rho = 1.0 / np.sum(y * s) if np.abs(np.sum(y * s)) > 1e-20 else 1.0
            beta = rho * np.sum(y * r)
            r += s * (alpha - beta)
            
        return -r

    def update(self, s, y):
        # 曲率条件检查 s^T y > 0
        if np.sum(s * y) > 1e-10:
            if len(self.s_history) >= self.m:
                self.s_history.pop(0)
                self.y_history.pop(0)
            self.s_history.append(s)
            self.y_history.append(y)


# --- 策略模块: 步长选择 ---

class StepSizeStrategy:
    def get_initial_step(self, current_t, iter_idx, s=None, y=None):
        return 1.0

class FixedStep(StepSizeStrategy):
    def __init__(self, initial_guess=1.0):
        self.guess = initial_guess

    def get_initial_step(self, current_t, iter_idx, s=None, y=None):
        # L-BFGS 通常每次重置为 1.0，GD 如果不是 BB 则重置为默认猜测
        return self.guess

class BBStep(StepSizeStrategy):
    def __init__(self, alpha_min=1e-5, alpha_max=1e5):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def get_initial_step(self, current_t, iter_idx, s=None, y=None):
        if iter_idx == 0 or s is None or y is None:
            return 1.0
        
        ss = np.sum(s * s)
        sy = np.abs(np.sum(s * y))
        yy = np.sum(y * y)
        
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
        
        # 记录初始梯度范数，用于相对误差判定
        initial_grad_norm = None
        
        ls = LineSearch(self.problem)
        print(f"Start Opt: Dir={direction_strategy.__class__.__name__}, Step={step_strategy.__class__.__name__}")

        for i in range(max_iters):
            grad = self.problem.compute_euclidean_grad(X, AX)
            grad_proj = StiefelManifold.project_gradient(X, grad)
            
            grad_norm_sq = np.sum(grad_proj**2)
            grad_norm = np.sqrt(grad_norm_sq)
            
            # --- 改进 1: 相对误差判定 ---
            if initial_grad_norm is None:
                initial_grad_norm = grad_norm
                print(f"  Init Grad Norm: {initial_grad_norm:.2e}")
            
            # 判定标准：梯度范数小于绝对阈值 OR 相比初始梯度下降了 1e-6 倍
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

            # --- 改进 2: L-BFGS 重启机制 ---
            if not success:
                # 如果是 L-BFGS 且失败了，尝试清空历史重启，而不是直接退出
                if isinstance(direction_strategy, LBFGS) and len(direction_strategy.s_history) > 0:
                    print(f"  Line search stuck at iter {i}. Restarting L-BFGS...")
                    direction_strategy.s_history = []
                    direction_strategy.y_history = []
                    # 重新尝试用纯梯度方向搜索
                    direction = -grad_proj
                    t_guess = 1.0 
                    X_new, f_new, AX_new, t_actual, success = ls.search(
                        X, f_val, AX, grad_proj, direction, t_guess, grad_norm_sq
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
            
            f_vals.append(f_val)
            
            # 调试打印
            if i % 100 == 0:
                print(f"  Iter {i}: f={f_val:.4e}, |g|={grad_norm:.2e}")

        return X, f_vals

def main():
    n, p = 500, 5
    print(f"Matrix Size: {n}x{n}, Stiefel Manifold St({n}, {p})")
    
    # 构造数据
    np.random.seed(42)
    A_raw = np.random.rand(n, n)
    A = A_raw.T @ A_raw
    #B = np.random.randn(n, p)
    B = np.zeros((n, p))
    
    x0, _ = np.linalg.qr(np.random.randn(n, p))
    
    solver = StiefelSolver(n, p, A, B)
    
    # --- 1. Armijo (普通梯度下降 + 固定/Armijo步长猜测) ---
    start = time.time()
    # 策略: 梯度方向 + 固定初始步长1.0 (然后由线搜索缩减)
    _, f_vals_armijo = solver.solve(x0, SteepestDescent(), FixedStep(1.0), max_iters=2000)
    print(f"Armijo Time: {time.time() - start:.4f}s, min f: {min(f_vals_armijo):.6f}")
    
    # --- 2. BB Step (梯度下降 + BB步长) ---
    start = time.time()
    _, f_vals_bb = solver.solve(x0, SteepestDescent(), BBStep(), max_iters=2000)
    print(f"BB Step Time: {time.time() - start:.4f}s, min f: {min(f_vals_bb):.6f}")

    # --- 3. L-BFGS (L-BFGS方向 + 固定初始步长1.0) ---
    start = time.time()
    _, f_vals_lbfgs = solver.solve(x0, LBFGS(m=10), FixedStep(1.0), max_iters=2000)
    print(f"L-BFGS Time: {time.time() - start:.4f}s, min f: {min(f_vals_lbfgs):.6f}")
    
    # 绘图比较
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_vals_armijo, label='GD (Armijo)')
    plt.semilogy(f_vals_bb, label='GD (BB Step)')
    plt.semilogy(f_vals_lbfgs, label='L-BFGS')
    plt.legend()
    plt.title('Optimization Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Value (log scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()