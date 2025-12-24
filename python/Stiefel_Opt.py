import os
import time
from collections import deque
import gc

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

# 图片支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII的减号而不是Unicode减号

# --- 基础运算工具 ---

def fast_dot(A, B):
    '''计算两个矩阵的 Frobenius 内积'''
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
        '''
        二次型目标函数: f(X) = Tr(X^T A X) + 2 Tr(B^T X)
        '''
        AX = blas.dgemm(alpha=1.0, a=self.A, b=X)
        term1 = fast_dot(X, AX)
        term2 = 2.0 * fast_dot(self.B, X)
        return term1 + term2, AX

    def compute_euclidean_grad(self, X, AX):
        '''
        ∇f(X) = 2 A X + 2 B
        '''
        G = self.B.copy(order='F')
        G += AX
        G *= 2.0
        return G

class LineSearch:
    def __init__(self, problem, strategy='grippo', num_history=5, rho=0.5, c=1e-4, zh_rho=0.85, max_ls_iters=20):
        self.problem = problem
        self.strategy = strategy  # 'monotone', 'grippo', 'zhang_hager'
        self.rho = rho
        self.c = c
        self.max_ls_iters = max_ls_iters
        
        # Grippo 专用
        self.num_history = num_history
        self.history = []
        
        # Zhang-Hager 专用 (定义 4.2.3)
        self.zh_rho = zh_rho  # 参数 rho (varrho)
        self.C = None         # 参照值 C_k
        self.Q = 0.0          # 权重和 Q_k

    def search(self, X, f_val, AX, grad_proj, search_dir, initial_step, grad_norm_sq):
        # --- 步骤 1: 确定参考函数值 f_ref ---
        if self.strategy == 'monotone':
            f_ref = f_val
        elif self.strategy == 'grippo':
            # 确保当前值在历史中，以便取 max
            self.history.append(f_val)
            if len(self.history) > self.num_history:
                self.history.pop(0)
            f_ref = max(self.history)
        elif self.strategy == 'zhang_hager':
            if self.C is None:
                self.C = f_val
                self.Q = 1.0
            f_ref = self.C
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # --- 步骤 2: 回退线搜索---
        t = initial_step
        grad_dot_dir = fast_dot(grad_proj, search_dir)
        descent_factor = self.c * grad_dot_dir
        
        success = False
        X_new = X
        f_new = f_val
        AX_new = AX

        for _ in range(self.max_ls_iters):
            try:
                X_new_raw = StiefelManifold.retraction(X, t * search_dir)
                X_new_curr = np.asfortranarray(X_new_raw)
                f_new_curr, AX_new_curr = self.problem.compute_cost_and_cache(X_new_curr)

                # 检查 Armijo 条件 (使用上面确定的 f_ref)
                if f_new_curr <= f_ref + t * descent_factor:
                    X_new = X_new_curr
                    f_new = f_new_curr
                    AX_new = AX_new_curr
                    success = True
                    break
            except:
                pass
            t *= self.rho

        # --- 步骤 3: 搜索成功后的状态更新 ---
        if success:
            if self.strategy == 'zhang_hager':
                # 更新 C_{k+1} 和 Q_{k+1}
                Q_next = self.zh_rho * self.Q + 1
                C_next = (self.zh_rho * self.Q * self.C + f_new) / Q_next
                self.Q = Q_next
                self.C = C_next
            # Grippo 的 history 更新已经在步骤 1 中做了append
            # Monotone 不需要状态更新

        return X_new, f_new, AX_new, t, success

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
    """
    实现了 Algorithm 4.4 中的 BB 步长选择。
    mode: 
      - 'bb1': 总是使用 s's / s'y (长步长)
      - 'bb2': 总是使用 s'y / y'y (短步长)
      - 'alt': 奇数步用 bb1，偶数步用 bb2 (交替)
      - 'adaptive': 根据 step ratio 自适应选择 (可选扩展)
    """
    def __init__(self, mode='alt', min_a=1e-10, max_a=1e10):
        self.mode = mode
        self.alpha_min = min_a
        self.alpha_max = max_a

    def get_initial_step(self, t, i, s, y):
        if i == 0 or s is None: 
            return 1.0
        
        # 计算点积
        ss = fast_dot(s, s)
        sy = fast_dot(s, y)
        yy = fast_dot(y, y)
        
        # 防止除零
        if sy <= 1e-20 or yy <= 1e-20:
            return t # 保持上一步步长

        # BB1: (s, s) / (s, y)
        bb1 = ss / sy
        # BB2: (s, y) / (y, y)
        bb2 = sy / yy
        
        alpha = t
        if self.mode == 'bb1':
            alpha = bb1
        elif self.mode == 'bb2':
            alpha = bb2
        elif self.mode == 'alt':
            # Algorithm 4.4 常见的交替策略
            alpha = bb1 if (i % 2 == 1) else bb2
        
        # 截断保护
        return min(self.alpha_max, max(self.alpha_min, alpha))

# --- 求解器 ---

class StiefelSolver:
    def __init__(self, n, p, A, B):
        self.problem = QuadraticProblem(n, p, A, B)

    def solve(self, X_init, direction_strategy, step_strategy, 
          max_iters=1000, tol=1e-6, pbar=None, 
          ls_strategy='monotone', ls_memory=1, zh_rho=0.85, max_ls_iters=20):
        '''
        在 Stiefel 流形上求解优化问题
        X_init: 初始点 (n x p 矩阵)
        direction_strategy: 方向选择策略 (DirectionStrategy 实例)
        step_strategy: 步长选择策略 (StepSizeStrategy 实例)
        max_iters: 最大迭代次数
        tol: 收敛容忍度
        pbar: 进度条对象 (tqdm 实例)，如果为 None 则不显示进度条
        ls_strategy: 线搜索策略 ('monotone', 'grippo', 'zhang_hager')
        ls_memory: Grippo非单调线搜索的历史长度
        zh_rho: Zhang-Hager 策略中的 rho 参数
        max_ls_iters: 线搜索的最大迭代次数
        '''
        # 初始化计算
        X = np.asfortranarray(X_init)
        f_val, AX = self.problem.compute_cost_and_cache(X)
        f_vals = [f_val]
        grad_norms = []
        
        grad_proj_prev = None; X_prev = None; current_step = 1.0
        
        ls = LineSearch(self.problem, strategy=ls_strategy, num_history=ls_memory, zh_rho=zh_rho, max_ls_iters=max_ls_iters)   # 线搜索器
        
        grad = self.problem.compute_euclidean_grad(X, AX)
        grad_proj = StiefelManifold.project_gradient(X, grad)
        grad_norm_sq = fast_dot(grad_proj, grad_proj)
        
        actual_iters = 0

        # 计算A的条件数，用于修正tol
        #cond_A = np.linalg.cond(self.problem.A)
        #tol_modified_sq = (tol * cond_A)**2
        tol_sq = tol**2

        for i in range(max_iters):
            actual_iters = i + 1
            grad_norm = np.sqrt(grad_norm_sq)
            grad_norms.append(grad_norm) # 记录
            
            if pbar and (i % 10 == 0):
                pbar.set_postfix({"f": f"{f_val:.2e}", "|g|": f"{grad_norm:.1e}"})
            
            # 收敛检查
            if grad_norm_sq < tol_sq:
                if pbar: pbar.write(f"  Converged at iter {i}")
                break
            
            s, y = None, None
            if i > 0:
                s = X - X_prev  # 当前点与上一步点的差
                y = grad_proj - grad_proj_prev  # 当前梯度与上一步梯度的差
                direction_strategy.update(s, y) # 用于L-BFGS

            t_guess = step_strategy.get_initial_step(current_step, i, s, y) # 计算初始步长, 普通梯度速降为1
            direction = direction_strategy.compute_direction(grad_proj)

            X_new, f_new, AX_new, t_actual, success = ls.search(
                X, f_val, AX, grad_proj, direction, t_guess, grad_norm_sq
            )

            if not success:
                # 策略重置逻辑...
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

        return X, f_vals, grad_norms, actual_iters
    
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


# 修正 strategies 列表的定义，确保格式统一
def main():
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

def plot_benchmark_results(results_dict, group_name="Benchmark", plot_fx=False, plot_gradnorm=True, save_name=('','')):
    """
    通用绘图函数，兼顾屏幕阅读（颜色）和黑白打印（线型+标记）。
    save_name: (directory, filename_prefix)
    """
    # 预设样式循环：(颜色, 线型, 标记)
    styles = [
        {'c': 'black',    'ls': '-',  'marker': None},  # 基准/第一条线通常用实线黑
        {'c': '#D55E00',  'ls': '--', 'marker': 'o'},   # 朱红 + 虚线 + 圆圈
        {'c': '#0072B2',  'ls': '-.', 'marker': 's'},   # 蓝色 + 点划线 + 方块
        {'c': '#009E73',  'ls': ':',  'marker': '^'},   # 绿色 + 点线 + 三角
        {'c': '#CC79A7',  'ls': '--', 'marker': 'D'},   # 紫色 + 虚线 + 菱形
        {'c': '#56B4E9',  'ls': '-',  'marker': 'x'},   # 浅蓝 + 实线 + 叉号
    ]
    
    # --- 创建两个独立的 Figure 对象 ---
    # 图1：目标函数值
    if plot_fx:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
    # 图2：梯度范数
    if plot_gradnorm:
        fig2, ax2 = plt.subplots(figsize=(8, 6))

    for idx, (name, data) in enumerate(results_dict.items()):
        # 循环获取样式
        style = styles[idx % len(styles)]
        
        f_vals = data['f']
        g_norms = data['g']
        iterations = len(f_vals)
        
        # 计算标记间隔，避免标记太密
        mark_step = max(1, iterations // 10)
        
        # 1. 在图1 (ax1) 上绘制目标函数值
        if plot_fx:
            min_fx = min(f_vals)
            if min_fx <= 0:
                f_vals = [fv - min_fx + 1e-8 for fv in f_vals]
            ax1.semilogy(f_vals, 
                     label=name, 
                     color=style['c'], 
                     linestyle=style['ls'],
                     marker=style['marker'],
                     markevery=mark_step,
                     markersize=6,
                     markerfacecolor='none', 
                     linewidth=1.5)

        # 2. 在图2 (ax2) 上绘制梯度范数
        if plot_gradnorm:
            ax2.semilogy(g_norms, 
                     label=name, 
                     color=style['c'], 
                     linestyle=style['ls'],
                     marker=style['marker'],
                     markevery=mark_step,
                     markersize=6,
                     markerfacecolor='none',
                     linewidth=1.5)

    # --- 设置图1 (Cost Value) 的装饰 ---
    if plot_fx:
        ax1.set_xlabel("迭代次数")
        ax1.set_ylabel(r"log($f(X_k)-\min f$)")
        ax1.grid(True, which="both", ls="--", alpha=0.3)
        ax1.legend(loc='upper right', fontsize='small', framealpha=0.9)
        fig1.tight_layout()
        if save_name[0] and save_name[1]:
            fig1.savefig(os.path.join(save_name[0], f"{save_name[1]}_cost.png"), dpi=300)

    # --- 设置图2 (Gradient Norm) 的装饰 ---
    if plot_gradnorm:
        ax2.set_xlabel("迭代次数")
        ax2.set_ylabel(r"log($\| \mathrm{grad} f(X_k) \|$)")
        ax2.grid(True, which="both", ls="--", alpha=0.3)
        ax2.legend(loc='upper right', fontsize='small', framealpha=0.9)
        fig2.tight_layout()
        if save_name[0] and save_name[1]:
            fig2.savefig(os.path.join(save_name[0], f"{save_name[1]}_gradnorm.png"), dpi=300)


def Q1():
    '''
    第一题：编程实现 Algorithm 4.3 与 Algorithm 4.4 。数值实验，自己随机生成 Stiefel 流形的一个二次函数，用  Algorithm 4.3 和 4.4 极小化这个二次函数，比较两个算法的计算效果，探索非单调的作用，或者交替使用 BB步长两个公式的作用。
    '''
    # --- 实验设置 ---
    n, p = 500, 10  # Stiefel 流形 St(n, p)
    print(f"=== Algorithm 4.3 & 4.4 Benchmarking ===")
    print(f"Stiefel Manifold St({n}, {p}) with Quadratic Cost")
    
    # 随机生成二次函数数据: min Tr(X.T A X) + 2 Tr(B.T X)
    # 构造具有指定条件数的正定矩阵 A
    cond_num = 1e4
    min_eig, max_eig = 1.0, cond_num
    eig_vals = np.logspace(0, np.log10(cond_num), n)   # 方式A: 对数分布
    # eig_vals = np.linspace(min_eig, max_eig, n)   # 方式B: 线性分布
    Lambda = np.diag(eig_vals)
    X = np.random.randn(n, n)   # 生成随机高斯矩阵
    Q, _ = np.linalg.qr(X)  # QR分解获取正交阵 Q
    A = Q @ Lambda @ Q.T
    A = np.asfortranarray((A + A.T) / 2)    # 强制对称性 (消除浮点数计算误差)

    #B = np.asfortranarray(np.random.randn(n, p))
    B = np.asfortranarray(np.zeros((n, p)))  # 只考虑纯二次项
    
    # 初始点
    x0, _ = np.linalg.qr(np.random.randn(n, p))
    
    solver = StiefelSolver(n, p, A, B)
    
    # --- 定义对比策略 ---
    # 第一组: 比较三种非单调线搜索策略(Armijo)
    # 1. Steepest Descent + Monotone (M=1) + Armijo
    # 2. Steepest Descent + Grippo (M=10) + Armijo
    # 3. Steepest Descent + ZH (rho=0.5) + Armijo

    # 第二组: 比较三种非单调线搜索策略(BB步长)
    # 1. Steepest Descent + Monotone (M=1) + Alternating BB
    # 2. Steepest Descent + Grippo (M=10) + Alternating BB
    # 3. Steepest Descent + Zhang-Hager (rho=0.5) + Alternating BB

    # 第三组: 比较三种 BB 步长公式 (均使用 ZH 非单调线搜索)
    # 1. Steepest Descent + Monotone (M=1) + Armijo
    # 2. Steepest Descent + ZH (rho=0.5) + BB1
    # 3. Steepest Descent + ZH (rho=0.5) + BB2
    # 4. Steepest Descent + ZH (rho=0.5) + Alternating BB

    # 通用参数
    common_params = {
        'max_iters': 1500,
        'tol': 1e-9,
        'max_ls_iters': 30
    }

    # --- 定义三组实验配置 ---
    # 格式: (Label, Direction, StepStrategy, LS_Strategy, LS_Memory, ZH_Rho)
    
    groups = []

    # Group 1: 固定步长(Armijo)下，不同线搜索策略的对比
    groups.append({
        "name": "Group 1: Different Line Search Strategies (Fixed Step)",
        "configs": [
            ("SD + Monotone + Armijo", SteepestDescent(), FixedStep(1.0), 'monotone', 1, 0.5),
            ("SD + Grippo(M=10) + Armijo", SteepestDescent(), FixedStep(1.0), 'grippo', 10, 0.5),
            ("SD + ZH(rho=0.5) + Armijo", SteepestDescent(), FixedStep(1.0), 'zhang_hager', 1, 0.5),
        ]
    })

    # Group 2: BB步长(Alternating)下，不同线搜索策略的对比
    groups.append({
        "name": "Group 2: Different Line Search Strategies (BB Alt Step)",
        "configs": [
            ("SD + Monotone + BB(Alt)", SteepestDescent(), BBStep(mode='alt'), 'monotone', 1, 0.5),
            ("SD + Grippo(M=10) + BB(Alt)", SteepestDescent(), BBStep(mode='alt'), 'grippo', 10, 0.5),
            ("SD + ZH(rho=0.5) + BB(Alt)", SteepestDescent(), BBStep(mode='alt'), 'zhang_hager', 1, 0.5),
        ]
    })

    # Group 3: 固定线搜索(ZH/Monotone)下，不同步长公式的对比
    # 注：题目要求 Baseline 为 Monotone+Armijo，其余为 ZH+BB 变种
    groups.append({
        "name": "Group 3: Different Step Sizes (Baseline vs ZH+BB)",
        "configs": [
            ("SD + Monotone + Armijo", SteepestDescent(), FixedStep(1.0), 'monotone', 1, 0.5),
            ("SD + ZH + BB1", SteepestDescent(), BBStep(mode='bb1'), 'zhang_hager', 1, 0.5),
            ("SD + ZH + BB2", SteepestDescent(), BBStep(mode='bb2'), 'zhang_hager', 1, 0.5),
            ("SD + ZH + BB(Alt)", SteepestDescent(), BBStep(mode='alt'), 'zhang_hager', 1, 0.5),
        ]
    })

    # --- 运行循环 ---
    for group in groups:
        print(f"\n{'='*20}\nRunning {group['name']}\n{'='*20}")
        group_results = {}
        
        # 使用 enumerate 主要是为了让 tqdm 的 position 正确显示
        for i, conf in enumerate(group['configs']):
            label, d_strat, s_strat, ls_strat, ls_mem, zh_rho = conf
            
            pbar = tqdm(total=common_params['max_iters'], desc=f"{label:<30}", leave=True)
            
            start_t = time.time()
            _, f_vals, g_norms, iters = solver.solve(
                x0.copy(),
                direction_strategy=d_strat,
                step_strategy=s_strat,
                max_iters=common_params['max_iters'],
                tol=common_params['tol'],
                pbar=pbar,
                ls_strategy=ls_strat,
                ls_memory=ls_mem,
                zh_rho=zh_rho,
                max_ls_iters=common_params['max_ls_iters']
            )
            elapsed = time.time() - start_t
            pbar.close()
            
            # 记录结果
            group_results[label] = {
                'f': np.array(f_vals),
                'g': np.array(g_norms),
                'time': elapsed,
                'iter': iters
            }
        
        # --- 每组跑完后立即画图 ---
        # 打印简单统计表
        print(f"\nSummary for {group['name']}:")
        print(f"{'Method':<35} | {'Time(s)':<8} | {'Iter':<5} | {'Min Cost':<12}")
        print("-" * 75)
        for label, res in group_results.items():
            print(f"{label:<35} | {res['time']:.4f}   | {res['iter']:<5} | {res['f'].min():.4e}")
        
        # 调用独立绘图函数
        plot_benchmark_results(group_results, group_name=group['name'])

def Q3(fig_save_dir=None, text_save_dir=None):
    '''
    第三题：实现 Stiefel 流形 St(n, p) 上的 L-BFGS 算法，并以二次函数作为测试函数。针对不同的矩阵维度 (n, p) 进行数值实验，性能评估指标至少包括以下几项：代码运行时间、迭代次数、最优解处梯度的范数，最优解目标函数值。
    '''
    print(f"\n{'='*65}\nRunning Q3: Strictly Fair Scalability Benchmark\n{'='*65}")
    print("Correction: Using consistent Random Matrix generation for ALL sizes")
    print("to ensure spectral difficulty is comparable across dimensions.")
    print(f"{'-'*65}")

    experiments = {
        "Fixed_p": {
            "desc": "Fixed p=10, Increasing n. Test pure computational load.",
            "dims": [
                (1000, 10),
                # (2000, 10),
                # (4000, 10),
                # (6000, 10) 
            ]
        },
        "Fixed_n": {
            "desc": "Fixed n=2000, Increasing p. Test manifold geometry complexity.",
            "dims": [
                (2000, 10),
                # (2000, 50),
                # (2000, 100),
                # (2000, 200)
            ]
        }
    }
    
    # 统一参数
    lbfgs_m = 10
    max_iters = 1000
    tol = 1e-6
    
    for exp_name, exp_config in experiments.items():
        print(f"\n>>> Running {exp_name}: {exp_config['desc']}")
        results_store = {}
        table_metrics = []
        
        for idx, (n, p) in enumerate(exp_config['dims']):
            label = f"St({n}, {p})"
            
            # --- 【关键修正】统一数据生成 ---
            # 不再使用 QR 分解（太慢）或 Low-Rank 近似（太简单）
            # 使用 GOE (Gaussian Orthogonal Ensemble) 生成法
            # 保证所有维度的矩阵都具有“同等难度”的全秩特征
            np.random.seed(42 + n) 
            
            # 1. 生成随机矩阵 (使用 float32 稍微加速生成，求解时会自动转 float64)
            # 注意：对于 n=6000，生成矩阵约需 0.5s，内存约 280MB
            A_raw = np.random.randn(n, n).astype(np.float32)
            
            # 2. 对称化
            A = (A_raw + A_raw.T) / 2
            # 显式转回 float64 保证精度
            A = np.asfortranarray(A, dtype=np.float64) 
            
            # 释放原内存
            del A_raw; gc.collect()
            
            B = np.asfortranarray(np.zeros((n, p)))
            x0, _ = np.linalg.qr(np.random.randn(n, p))
            
            # --- 求解 ---
            solver = StiefelSolver(n, p, A, B)
            
            # 进度条
            pbar = tqdm(total=max_iters, desc=f"   {label:<12}", leave=True,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            
            start_t = time.time()
            # 依然使用 Zhang-Hager 线搜索，保证鲁棒性
            _, f_vals, g_norms, iters = solver.solve(
                x0, LBFGS(m=lbfgs_m), FixedStep(1.0), 
                max_iters=max_iters, tol=tol, pbar=pbar,
                ls_strategy='zhang_hager', zh_rho=0.5
            )
            elapsed = time.time() - start_t
            pbar.close()
            
            t_per_k = elapsed / iters if iters > 0 else 0.0
            
            # --- 记录 ---
            results_store[label] = {'f': np.array(f_vals), 'g': np.array(g_norms)}
            table_metrics.append({
                "n": n, "p": p,
                "time": elapsed,
                "iter": iters,
                "t_per_k": t_per_k,
                "grad": g_norms[-1]
            })
            
            del A, solver, x0
            gc.collect()

        # --- 1. 打印 ASCII 表格到控制台 ---
        print(f"\nResults for {exp_name}:")
        print(f"{'Dims (n, p)':<15} | {'Total Time':<10} | {'Iters':<6} | {'Time/Iter (s)':<13} | {'Final |Grad|':<12}")
        print("-" * 75)
        for row in table_metrics:
            dims_str = f"({row['n']}, {row['p']})"
            print(f"{dims_str:<15} | {row['time']:.4f}s    | {row['iter']:<6} | {row['t_per_k']:.6f}      | {row['grad']:.4e}")
        print("-" * 75)

        # --- 2. 保存 LaTeX 表格到文件 (新增功能) ---
        if text_save_dir:
            tex_filename = os.path.join(text_save_dir, 'Q3'+f"{exp_name}.tex")
            
            # LaTeX 表格模板
            header = r"""\begin{table}[htbp]
    \centering
    \caption{Performance Benchmarks for """ + exp_config['desc'] + r"""}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Dims} $(n, p)$ & \textbf{Time (s)} & \textbf{Iter} & \textbf{Time/Iter (s)} & \textbf{Final} $\|\nabla f\|$ \\
        \midrule
"""
            footer = r"""        \bottomrule
    \end{tabular}
    \label{tab:""" + exp_name + r"""}
\end{table}
"""
            try:
                with open(tex_filename, "w", encoding='utf-8') as f:
                    f.write(header)
                    for row in table_metrics:
                        # 格式化每行数据
                        dims_str = f"$({row['n']}, {row['p']})$"
                        # 梯度用科学计数法 1.23e-05，其他用浮点
                        line = f"        {dims_str} & {row['time']:.4f} & {row['iter']} & {row['t_per_k']:.6f} & {row['grad']:.2e} \\\\\n"
                        f.write(line)
                    f.write(footer)
                print(f"LaTeX 表格已保存至: {os.path.abspath(tex_filename)}")
            except Exception as e:
                print(f"保存 LaTeX 表格失败: {e}")
        
        # 绘图
        plot_benchmark_results(results_store, group_name=f"{exp_name} Analysis", save_name=(fig_save_dir, 'Q3'+f"{exp_name}"), plot_fx=True, plot_gradnorm=True)

if __name__ == "__main__":
    current_dir = os.getcwd()
    fig_save_dir = os.path.join(current_dir, 'latex/figures')
    text_save_dir = os.path.join(current_dir, 'latex/tables')
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    if not os.path.exists(text_save_dir):
        os.makedirs(text_save_dir)
    Q3(fig_save_dir=fig_save_dir, text_save_dir=text_save_dir)