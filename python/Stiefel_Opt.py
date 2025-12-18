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

class DampedLBFGS(LBFGS):
    """
    带阻尼的 L-BFGS 策略
    参考 PDF 第 147 页 (5.2.96)-(5.2.97) 及 第 151 页 (5.2.102)
    """
    def __init__(self, m=10, delta=1.0):
        super().__init__(m)
        self.delta = delta  # B_k0 的缩放因子，对应 PDF 中的 B_{k,0} = delta * I

    def update(self, s, y):
        """
        在更新历史之前，对 y 进行阻尼修正以保证正定性
        """
        # 计算 s^T y
        sy = np.sum(s * y)
        
        # 计算 s^T B_{k,0} s
        # 这里假设 B_{k,0} = delta * I，所以结果是 delta * (s^T s)
        ss = np.sum(s * s)
        sBs = self.delta * ss

        # 如果 sBs 非常小（s 接近 0），直接跳过更新以防除零
        if sBs < 1e-20:
            return

        # 计算 theta (PDF 公式 5.2.97)
        # 条件: s^T y >= 0.25 * s^T B s
        if sy >= 0.25 * sBs:
            theta = 1.0
        else:
            theta = (0.75 * sBs) / (sBs - sy)

        # 计算修正后的向量 r (PDF 公式 5.2.96 / 5.2.102)
        # r = theta * y + (1 - theta) * B_{k,0} * s
        r = theta * y + (1.0 - theta) * (self.delta * s)

        # 将修正后的对 (s, r) 存入历史
        # 此时必然满足 s^T r > 0，因此不需要像标准 LBFGS 那样做 sy > 1e-10 的判断
        if len(self.s_history) >= self.m:
            self.s_history.pop(0)
            self.y_history.pop(0)
        
        self.s_history.append(s)
        self.y_history.append(r)

class SubspaceLBFGS(DirectionStrategy):
    """
    流形子空间 L-BFGS 方法 (Subspace L-BFGS)
    参考 PDF 第 153-156 页 (Section 5.2.2)
    核心思想：在由梯度张成的低维子空间 G_k 中维护拟牛顿矩阵 \bar{H}_k。
    """
    def __init__(self, max_dim=20, delta=1.0):
        self.max_dim = max_dim     # 子空间最大维数，超过则重启
        self.delta = delta         # 初始 Hessian 缩放
        self.Z = None              # 正交基矩阵 (Flattened: np x dim)
        self.H_sub = None          # 子空间内的逆 Hessian (dim x dim)
        self.g_prev_flat = None    # 缓存上一步梯度用于基的扩展
        self.dim = 0               # 当前子空间维度

    def _reset(self, g_flat):
        """重启子空间：以当前梯度为第一个基向量"""
        # 归一化梯度作为第一个基
        gnorm = np.linalg.norm(g_flat)
        if gnorm < 1e-20:
            gnorm = 1.0
        
        self.Z = (g_flat / gnorm).reshape(-1, 1)
        self.dim = 1
        
        # 初始化子空间 Hessian 为单位阵 (或缩放单位阵)
        self.H_sub = np.eye(1) * (1.0 / self.delta)

    def compute_direction(self, grad_proj):
        # 将 (n, p) 的梯度展平为向量，以便进行欧氏空间的子空间运算
        # 流形上的内积 <A, B> = Tr(A.T B) 等价于展平后的点积
        g_flat = grad_proj.reshape(-1)
        
        # 1. 初始化或重启判断
        if self.Z is None:
            self._reset(g_flat)
        
        # 保存当前梯度供 update 阶段使用 (用于构建 Z_{k+1})
        self.g_prev_flat = g_flat.copy()

        # 2. 投影梯度到子空间: \bar{g} = Z^T g (PDF Eq 5.2.110)
        g_sub = self.Z.T @ g_flat

        # 3. 在子空间计算方向: \bar{p} = -\bar{H} \bar{g}
        # 这里 H_sub 近似的是 Hessian 的逆，所以直接乘
        p_sub = -self.H_sub @ g_sub

        # 4. 映射回全空间: p = Z \bar{p}
        p_flat = self.Z @ p_sub
        
        # 恢复形状 (n, p)
        direction = p_flat.reshape(grad_proj.shape)
        
        return direction

    def update(self, s, y):
        """
        更新 Z 和 H_sub
        依据 PDF Eq 5.2.112 (基更新) 和 Eq 5.2.116 (矩阵更新)
        """
        if self.Z is None or self.g_prev_flat is None:
            return

        # 展平 s 和 y
        s_flat = s.reshape(-1)
        y_flat = y.reshape(-1)

        # ---------------------------------------------------------
        # 1. 扩展子空间 Z (PDF Eq 5.2.112)
        # ---------------------------------------------------------
        # 新的基向量来自于当前的梯度 g_{k+1}
        # 注意：y = g_{k+1} - g_k  =>  g_{k+1} = y + g_k
        g_next_flat = y_flat + self.g_prev_flat

        # 计算 g_{k+1} 在当前 Z 上的投影残差
        # u = g_{k+1} - Z Z^T g_{k+1}
        ZTg = self.Z.T @ g_next_flat
        u = g_next_flat - self.Z @ ZTg
        
        u_norm = np.linalg.norm(u)

        # 如果残差足够大，且未达到最大维数，则扩展子空间
        if u_norm > 1e-6 and self.dim < self.max_dim:
            # 添加新基向量 z_{new}
            z_new = (u / u_norm).reshape(-1, 1)
            self.Z = np.hstack([self.Z, z_new])
            
            # 扩展 H_sub
            # PDF Eq 5.2.114 暗示新维度初始与其他维度解耦 (或是单位阵)
            # 简单的策略：扩充 H_sub，对角线补 1/delta
            new_dim = self.dim + 1
            H_new = np.eye(new_dim) * (1.0 / self.delta)
            H_new[:self.dim, :self.dim] = self.H_sub
            self.H_sub = H_new
            self.dim = new_dim
        elif self.dim >= self.max_dim:
            # 达到最大维数，标记需要在下一次 compute_direction 时重启
            # 但为了保持当前迭代的连贯性，这里我们可以选择不扩展，
            # 仅在当前子空间更新 H，或者直接重置 self.Z = None
            # 这里采用“软重启”策略：本次只更新 H，下次 compute_direction 会检测并可能硬重启
            # 为简单起见，我们直接置空，强制下次重启
            self.Z = None
            return

        # ---------------------------------------------------------
        # 2. 更新子空间矩阵 H_sub (PDF Eq 5.2.116)
        # ---------------------------------------------------------
        # 计算投影后的 s 和 y: \tilde{s} = Z^T s, \tilde{y} = Z^T y
        s_sub = self.Z.T @ s_flat
        y_sub = self.Z.T @ y_flat

        # 检查曲率条件
        sy_sub = np.dot(s_sub, y_sub)
        if sy_sub > 1e-10:
            # 标准 BFGS 更新公式 (作用在小矩阵 H_sub 上)
            rho = 1.0 / sy_sub
            I = np.eye(self.dim)
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
    
    # --- 4. Damped L-BFGS (带阻尼) ---
    start = time.time()
    # 阻尼版：强制修正 y，使得所有步骤的数据都被利用
    _, f_vals_damped = solver.solve(x0, DampedLBFGS(m=10, delta=20.0), FixedStep(1.0), max_iters=2000)
    print(f"Damped L-BFGS Time: {time.time() - start:.4f}s, min f: {min(f_vals_damped):.6f}")

    # --- 5. Subspace L-BFGS (New!) ---
    # max_dim 控制子空间大小，相当于 L-BFGS 的 memory
    start = time.time()
    subspace_strategy = SubspaceLBFGS(max_dim=20, delta=1.0)
    _, f_vals_subspace = solver.solve(x0, subspace_strategy, FixedStep(1.0), max_iters=2000)
    print(f"Subspace L-BFGS Time: {time.time() - start:.4f}s, min f: {min(f_vals_subspace):.6f}")

    # 绘图比较
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_vals_armijo, label='GD (Armijo)')
    plt.semilogy(f_vals_bb, label='GD (BB Step)')
    plt.semilogy(f_vals_lbfgs, label='L-BFGS')
    plt.semilogy(f_vals_damped, label='Damped L-BFGS', linestyle='--', linewidth=1.5)
    plt.semilogy(f_vals_subspace, label='Subspace L-BFGS', linestyle='-.', linewidth=1.5)
    plt.legend()
    plt.title('Optimization Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Value (log scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()