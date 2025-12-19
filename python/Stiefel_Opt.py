import os
import time
from collections import deque

# --- 1. 环境配置 ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg.blas as blas
from tqdm import tqdm  # 引入进度条库
from scipy.linalg.blas import ddot

print(f"线程环境已配置为单线程模式。")
plt.rcParams.update(plt.rcParamsDefault)

# --- 基础运算工具 ---

def fast_dot(A, B):
    return ddot(A.ravel('F'), B.ravel('F'))

class StiefelManifold:
    @staticmethod
    def retraction(X, Z):
        U, _, Vt = np.linalg.svd(X + Z, full_matrices=False)
        return U @ Vt

    @staticmethod
    def project_gradient(X, G):
        # 使用 dsyrk 只计算 X.T @ G 的对称部分
        M = blas.dgemm(alpha=1.0, a=X, b=G, trans_a=True)
        # 直接原地对称化，避免 M + M.T 的内存分配
        np.add(M, M.T, out=M)
        M *= 0.5
        return blas.dgemm(alpha=-1.0, a=X, b=M, beta=1.0, c=G, overwrite_c=True)

class QuadraticProblem:
    def __init__(self, n, p, A, B):
        self.n = n
        self.p = p
        self.A = np.asfortranarray(A)
        self.B = np.asfortranarray(B)

    def compute_cost_and_cache(self, X):
        AX = blas.dgemm(alpha=1.0, a=self.A, b=X)
        term1 = fast_dot(X, AX)
        term2 = 2.0 * fast_dot(self.B, X)
        return term1 + term2, AX

    def compute_euclidean_grad(self, X, AX):
        G = self.B.copy(order='F')
        G += AX
        G *= 2.0
        return G

class LineSearch:
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
        grad_dot_dir = fast_dot(grad_proj, search_dir)
        descent_factor = self.c * grad_dot_dir
        
        for _ in range(self.max_ls_iters):
            try:
                X_new_raw = StiefelManifold.retraction(X, t * search_dir)
                X_new = np.asfortranarray(X_new_raw) # 保持 F-order
                
                f_new, AX_new = self.problem.compute_cost_and_cache(X_new)
                
                if f_new <= f_ref + t * descent_factor:
                    return X_new, f_new, AX_new, t, True
                
            except np.linalg.LinAlgError:
                pass
            
            t *= self.rho

        return X, f_val, AX, 0.0, False

# --- 策略模块 (保持不变) ---

class DirectionStrategy:
    def compute_direction(self, grad_proj): raise NotImplementedError
    def update(self, s, y): pass

class SteepestDescent(DirectionStrategy):
    def compute_direction(self, grad_proj): return -grad_proj

class LBFGS(DirectionStrategy):
    def __init__(self, m=10):
        self.m = m
        self.memory = deque(maxlen=m)

    def compute_direction(self, grad_proj):
        if not self.memory: return -grad_proj
        q = grad_proj.copy(order='F')
        mem_len = len(self.memory)
        alphas = [0.0] * mem_len
        
        for i in range(mem_len - 1, -1, -1):
            s, y, rho = self.memory[i]
            alpha = rho * fast_dot(s, q)
            alphas[i] = alpha
            q -= alpha * y
            
        s_last, y_last, rho_last = self.memory[-1]
        yy_last = fast_dot(y_last, y_last)
        gamma = (1.0 / rho_last) / yy_last if yy_last > 1e-20 else 1.0
        
        r = q 
        r *= gamma
        
        for i in range(mem_len):
            s, y, rho = self.memory[i]
            beta = rho * fast_dot(y, r)
            r += s * (alphas[i] - beta)
        return -r

    def update(self, s, y):
        sy = fast_dot(s, y)
        if sy > 1e-10:
            rho = 1.0 / sy
            self.memory.append((s, y, rho))

class DampedLBFGS(LBFGS):
    def __init__(self, m=10, delta=1.0):
        super().__init__(m)
        self.delta = delta 
    def update(self, s, y):
        sy = fast_dot(s, y)
        ss = fast_dot(s, s)
        sBs = self.delta * ss
        if sBs < 1e-20: return
        theta = 1.0 if sy >= 0.25 * sBs else (0.75 * sBs) / (sBs - sy)
        r = theta * y + (1.0 - theta) * self.delta * s
        sr = fast_dot(s, r)
        if sr > 1e-10:
            rho = 1.0 / sr
            self.memory.append((s, r, rho))

class SubspaceLBFGS(DirectionStrategy):
    def __init__(self, max_dim=20, delta=1.0):
        self.max_dim = max_dim
        self.delta = delta
        self.Z_buffer = None; self.dim = 0; self.H_sub = None; self.g_prev_flat = None

    def _reset(self, g_flat):
        gnorm = np.linalg.norm(g_flat)
        if gnorm < 1e-20: gnorm = 1.0
        size = g_flat.size
        if self.Z_buffer is None or self.Z_buffer.shape[0] != size:
            self.Z_buffer = np.zeros((size, self.max_dim), dtype=g_flat.dtype, order='F')
        self.Z_buffer[:, 0] = g_flat / gnorm
        self.dim = 1
        self.H_sub = np.eye(1) * (1.0 / self.delta)

    def compute_direction(self, grad_proj):
        g_flat = grad_proj.ravel('F')
        if self.dim == 0: self._reset(g_flat)
        self.g_prev_flat = g_flat.copy()
        Z_active = self.Z_buffer[:, :self.dim]
        g_sub = Z_active.T @ g_flat
        p_sub = -self.H_sub @ g_sub
        p_flat = Z_active @ p_sub
        return p_flat.reshape(grad_proj.shape, order='F')

    def update(self, s, y):
        if self.dim == 0 or self.g_prev_flat is None: return
        s_flat = s.ravel('F'); y_flat = y.ravel('F')
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
            v = rho * s_sub
            I = np.eye(self.dim)
            V = I - np.outer(v, y_sub)
            self.H_sub = V @ self.H_sub @ V.T + rho * np.outer(s_sub, s_sub)

# --- 步长选择 ---

class StepSizeStrategy:
    def get_initial_step(self, current_t, iter_idx, s=None, y=None): return 1.0

class FixedStep(StepSizeStrategy):
    def __init__(self, guess=1.0): self.guess = guess
    def get_initial_step(self, t, i, s, y): return self.guess

class BBStep(StepSizeStrategy):
    def __init__(self, min_a=1e-5, max_a=1e5): self.alpha_min = min_a; self.alpha_max = max_a
    def get_initial_step(self, t, i, s, y):
        if i == 0 or s is None: return 1.0
        ss = fast_dot(s, s); sy = abs(fast_dot(s, y)); yy = fast_dot(y, y)
        alpha = ss/sy if (i%2==0 and sy>1e-10) else (sy/yy if yy>1e-10 else t)
        return min(self.alpha_max, max(self.alpha_min, alpha))

# --- 求解器 ---

class StiefelSolver:
    def __init__(self, n, p, A, B):
        self.problem = QuadraticProblem(n, p, A, B)

    def solve(self, X_init, direction_strategy, step_strategy, max_iters=1000, tol=1e-6, pbar=None):
        X = np.asfortranarray(X_init)
        f_val, AX = self.problem.compute_cost_and_cache(X)
        f_vals = [f_val]
        
        grad_proj_prev = None; X_prev = None; current_step = 1.0
        initial_grad_norm = None
        ls = LineSearch(self.problem)
        
        grad = self.problem.compute_euclidean_grad(X, AX)
        grad_proj = StiefelManifold.project_gradient(X, grad)
        grad_norm_sq = fast_dot(grad_proj, grad_proj)
        
        # 记录实际迭代次数
        actual_iters = 0

        for i in range(max_iters):
            actual_iters = i + 1
            if i % 10 == 0 or i == 0:
                grad_norm = np.sqrt(grad_norm_sq)
                if initial_grad_norm is None: initial_grad_norm = grad_norm
                
                # 更新进度条描述
                if pbar:
                    pbar.set_postfix({"f": f"{f_val:.2e}", "|g|": f"{grad_norm:.1e}"})
            
            # 收敛检查
            if grad_norm_sq < (tol * initial_grad_norm)**2:
                if pbar: pbar.write(f"  Converged at iter {i}")
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
                if isinstance(direction_strategy, LBFGS) and len(direction_strategy.memory) > 0:
                    if pbar: pbar.write(f"  Reset L-BFGS at {i}")
                    direction_strategy.memory.clear()
                    direction = -grad_proj
                    X_new, f_new, AX_new, t_actual, success = ls.search(
                        X, f_val, AX, grad_proj, direction, 1.0, grad_norm_sq
                    )
                if not success:
                    if pbar: pbar.write(f"  LS failed at {i}")
                    break

            X_prev = X; grad_proj_prev = grad_proj
            X = X_new; f_val = f_new; AX = AX_new; current_step = t_actual
            
            grad = self.problem.compute_euclidean_grad(X, AX)
            grad_proj = StiefelManifold.project_gradient(X, grad)
            grad_norm_sq = fast_dot(grad_proj, grad_proj)
            f_vals.append(f_val)
            
            if pbar: pbar.update(1)

        return X, f_vals, actual_iters

def save_latex_table(results_data, filename="benchmark_results.txt"):
    """生成 LaTeX 表格并保存到文件"""
    header = r"""
\begin{table}[h]
    \centering
    \caption{Optimization Algorithm Benchmarks}
    \begin{tabular}{|l|c|c|c|c|}
        \hline
        \textbf{Method} & \textbf{Step Strategy} & \textbf{Time (s)} & \textbf{Min Cost} & \textbf{Iterations} \\
        \hline
"""
    footer = r"""        \hline
    \end{tabular}
    \label{tab:benchmark}
\end{table}
"""
    
    body = ""
    for r in results_data:
        # 使用 & 分隔，最后加 \\
        line = f"        {r['name']} & {r['step']} & {r['time']:.4f} & {r['min_f']:.4e} & {r['iters']} \\\\\n"
        body += line

    content = header + body + footer
    
    with open(filename, "w", encoding='utf-8') as f:
        f.write(content)
    
    print(f"LaTeX 表格已保存至: {os.path.abspath(filename)}")

def main():
    n, p = 1000, 10 # 稍微加大力度测试进度条
    print(f"Matrix Size: {n}x{n}, Stiefel Manifold St({n}, {p})")
    
    np.random.seed(42)
    A_raw = np.random.rand(n, n)
    A = np.asfortranarray(A_raw.T @ A_raw)
    B = np.asfortranarray(np.zeros((n, p)))
    x0, _ = np.linalg.qr(np.random.randn(n, p))
    
    solver = StiefelSolver(n, p, A, B)
    
    strategies = [
        (SteepestDescent(), FixedStep(1.0), "GD", "Fixed"),
        (SteepestDescent(), BBStep(), "GD", "BB"),
        (LBFGS(m=10), FixedStep(1.0), "L-BFGS", "Fixed"),
        (DampedLBFGS(m=10, delta=20.0), "Damped L-BFGS", "Fixed"), # 注意这里的元组长度处理
        (SubspaceLBFGS(max_dim=20, delta=1.0), FixedStep(1.0), "Subspace L-BFGS", "Fixed")
    ]
    
    results = {}
    table_data = [] # 用于存储 LaTeX 表格数据
    
    # 使用 position 和 leave 参数可以保留多个进度条
    # 这里的 total=2000 对应 max_iters
    for i, item in enumerate(strategies):
        # 处理元组长度不一致的简便方法
        if len(item) == 4:
            d_strat, s_strat, d_name, s_name = item
        else:
            # 兼容旧定义 (DampedLBFGS 那个条目如果写错了)
            # 这里修复上面的 DampedLBFGS 定义
            d_strat, name, _ = item # 这行实际上不会运行，因为下面我修正了上面的 list
            s_strat = FixedStep(1.0)
            d_name = name; s_name = "Fixed"

        # 组合全名
        full_name = f"{d_name} + {s_name}"
        
        # 初始化进度条
        # position=i 让进度条按行排列，leave=True 跑完不消失
        pbar = tqdm(total=2000, desc=f"{full_name:<25}", position=i, leave=True)
        
        start = time.time()
        _, f_vals, iters = solver.solve(
            x0.copy(), d_strat, s_strat, max_iters=2000, pbar=pbar
        )
        elapsed = time.time() - start
        
        pbar.close() # 强制刷新
        
        min_f = min(f_vals) if f_vals else float('inf')
        results[full_name] = f_vals
        
        # 收集数据
        table_data.append({
            "name": d_name,
            "step": s_name,
            "time": elapsed,
            "min_f": min_f,
            "iters": iters
        })

    # 导出 LaTeX 表格
    print("\n" * (len(strategies) + 2)) # 腾出空间，避免进度条遮挡
    save_latex_table(table_data)

    # 绘图
    plt.figure(figsize=(10, 6))
    for name, f_vals in results.items():
        style = '--' if 'Damped' in name else ('-.' if 'Subspace' in name else '-')
        plt.semilogy(f_vals, label=name, linestyle=style, linewidth=1.5)
    
    plt.legend()
    plt.title(f'Optimization Convergence (n={n}, p={p})')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Value (log scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# 修正 strategies 列表的定义，确保格式统一
def main_fixed():
    n, p = 500, 5
    print(f"Matrix Size: {n}x{n}, Stiefel Manifold St({n}, {p})")
    
    np.random.seed(42)
    A_raw = np.random.rand(n, n)
    A = np.asfortranarray(A_raw.T @ A_raw)
    B = np.asfortranarray(np.zeros((n, p)))
    x0, _ = np.linalg.qr(np.random.randn(n, p))
    
    solver = StiefelSolver(n, p, A, B)
    
    # 统一格式: (DirStrat, StepStrat, DirName, StepName)
    strategies = [
        (SteepestDescent(), FixedStep(1.0), "GD", "Armijo"),
        (SteepestDescent(), BBStep(), "GD", "BB Step"),
        (LBFGS(m=10), FixedStep(1.0), "L-BFGS", "Armijo"),
        (DampedLBFGS(m=10, delta=20.0), FixedStep(1.0), "Damped L-BFGS", "Armijo"),
        (SubspaceLBFGS(max_dim=20, delta=1.0), FixedStep(1.0), "Subspace L-BFGS", "Armijo")
    ]
    
    results = {}
    table_data = []
    
    print("开始基准测试...")
    
    for i, (d_strat, s_strat, d_name, s_name) in enumerate(strategies):
        full_name = f"{d_name}"
        if "BB" in s_name: full_name += " (BB)"
        
        # position=i 允许同时显示多个条（虽然这里是串行执行，但会保留在屏幕上）
        pbar = tqdm(total=2000, desc=f"{full_name:<22}", position=i, leave=True, 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        
        start = time.time()
        _, f_vals, iters = solver.solve(
            x0.copy(), d_strat, s_strat, max_iters=2000, pbar=pbar
        )
        elapsed = time.time() - start
        pbar.close()
        
        min_f = min(f_vals) if f_vals else float('inf')
        results[full_name] = f_vals
        
        table_data.append({
            "name": d_name,
            "step": s_name,
            "time": elapsed,
            "min_f": min_f,
            "iters": iters
        })

    # 手动换行，跳过进度条区域
    print("\n" * 1) 
    save_latex_table(table_data)

    plt.figure(figsize=(10, 6))
    for name, f_vals in results.items():
        style = '--' if 'Damped' in name else ('-.' if 'Subspace' in name else '-')
        plt.semilogy(f_vals, label=name, linestyle=style, linewidth=1.5)
    
    plt.legend()
    plt.title(f'Optimization Convergence (n={n}, p={p})')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Value (log scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main_fixed()