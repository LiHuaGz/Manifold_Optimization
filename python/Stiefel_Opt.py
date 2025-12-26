import os
import time
from collections import deque
import gc
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
    \small
    \caption{优化算法性能对比}
    \begin{tabular}{|l|c|c|c|c|}
        \hline
        \textbf{方法} & \textbf{步长策略} & \textbf{时间 (秒)} & \textbf{最小} $f$ & \textbf{迭代次数} \\
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

def plot_benchmark_results(results_dict, group_name="Benchmark", plot_fx=False, plot_gradnorm=True, save_name=('','')):
    """
    通用绘图函数
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


def Q1(fig_save_dir=None, text_save_dir=None):
    '''
    第一题：数值实验，多维度测试，并输出 LaTeX 表格和图片。
    '''
    # --- 1. 实验维度设置 ---
    # 在这里添加你想测试的维度 (n, p)
    dims_list = [
        (500, 10), 
        (1000, 20)
    ]
    
    # 通用参数
    common_params = {
        'max_iters': 1500,
        'tol': 1e-9,
        'max_ls_iters': 30
    }

    print(f"\n{'='*65}\nRunning Q1: Algorithm 4.3 & 4.4 Benchmarking (Multi-dim)\n{'='*65}")

    # 用于存储生成 LaTeX 表格的数据
    # 结构: data[group_name] = { (n,p): [row_dict, row_dict, ...] }
    latex_tables_data = {}

    # --- 2. 遍历每一个维度配置 ---
    for n, p in dims_list:
        np.random.seed(42 + n) 
        print(f"\n>>> Processing Dimensions: n={n}, p={p}")
        
        # --- 数据生成 (对该维度下的所有方法保持一致) ---
        cond_num = 1e4
        eig_vals = np.logspace(0, np.log10(cond_num), n)
        Lambda = np.diag(eig_vals)
        X_rnd = np.random.randn(n, n)
        Q_mat, _ = np.linalg.qr(X_rnd)
        A = Q_mat @ Lambda @ Q_mat.T
        A = np.asfortranarray((A + A.T) / 2)
        B = np.asfortranarray(np.zeros((n, p)))
        x0, _ = np.linalg.qr(np.random.randn(n, p))
        
        solver = StiefelSolver(n, p, A, B)

        # --- 定义三组实验配置 (保持原有实验组设定) ---
        # 注意：在这里重新定义是为了方便在循环中引用，并确保对象是新的
        
        experiment_groups = []

        # Group 1: 固定步长(Armijo)下，不同线搜索策略的对比
        experiment_groups.append({
            "id": "G1",
            "name": "Different Line Search Strategies (Fixed Step)",
            "cn_name": "不同线搜索策略对比 (固定步长)",
            "configs": [
                ("SD + Monotone + Armijo", SteepestDescent(), FixedStep(1.0), 'monotone', 1, 0.5),
                ("SD + Grippo(M=10) + Armijo", SteepestDescent(), FixedStep(1.0), 'grippo', 10, 0.5),
                ("SD + ZH(rho=0.5) + Armijo", SteepestDescent(), FixedStep(1.0), 'zhang_hager', 1, 0.5),
            ]
        })

        # Group 2: BB步长(Alternating)下，不同线搜索策略的对比
        experiment_groups.append({
            "id": "G2",
            "name": "Different Line Search Strategies (BB Alt Step)",
            "cn_name": "不同线搜索策略对比 (BB交替步长)",
            "configs": [
                ("SD + Monotone + BB(Alt)", SteepestDescent(), BBStep(mode='alt'), 'monotone', 1, 0.5),
                ("SD + Grippo(M=10) + BB(Alt)", SteepestDescent(), BBStep(mode='alt'), 'grippo', 10, 0.5),
                ("SD + ZH(rho=0.5) + BB(Alt)", SteepestDescent(), BBStep(mode='alt'), 'zhang_hager', 1, 0.5),
            ]
        })

        # Group 3: 固定线搜索(ZH/Monotone)下，不同步长公式的对比
        experiment_groups.append({
            "id": "G3",
            "name": "Different Step Sizes (Baseline vs ZH+BB)",
            "cn_name": "不同步长公式对比",
            "configs": [
                ("SD + Monotone + Armijo", SteepestDescent(), FixedStep(1.0), 'monotone', 1, 0.5),
                ("SD + ZH + BB1", SteepestDescent(), BBStep(mode='bb1'), 'zhang_hager', 1, 0.5),
                ("SD + ZH + BB2", SteepestDescent(), BBStep(mode='bb2'), 'zhang_hager', 1, 0.5),
                ("SD + ZH + BB(Alt)", SteepestDescent(), BBStep(mode='alt'), 'zhang_hager', 1, 0.5),
            ]
        })

        # --- 运行实验组 ---
        for group in experiment_groups:
            group_id = group['id']
            group_cn_name = group['cn_name']
            
            # 初始化该组的数据存储结构
            if group_id not in latex_tables_data:
                latex_tables_data[group_id] = {
                    'cn_name': group_cn_name,
                    'dims_data': {}
                }
            if (n, p) not in latex_tables_data[group_id]['dims_data']:
                latex_tables_data[group_id]['dims_data'][(n, p)] = []

            group_results_for_plot = {} # 用于当前维度绘图的数据

            print(f"  Running {group['id']}: {group['name']}")
            
            for i, conf in enumerate(group['configs']):
                label, d_strat, s_strat, ls_strat, ls_mem, zh_rho = conf
                
                # 复制初始点
                X_init = x0.copy()

                pbar = tqdm(total=common_params['max_iters'], desc=f"    {label:<28}", leave=False)
                
                start_t = time.time()
                _, f_vals, g_norms, iters = solver.solve(
                    X_init,
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
                
                min_cost = min(f_vals) if f_vals else 0.0
                final_grad = g_norms[-1] if g_norms else 0.0

                # 1. 收集绘图数据
                group_results_for_plot[label] = {
                    'f': np.array(f_vals),
                    'g': np.array(g_norms)
                }

                # 2. 收集表格数据
                row_data = {
                    "method": label,
                    "time": elapsed,
                    "iter": iters,
                    "min_f": min_cost,
                    "final_g": final_grad
                }
                latex_tables_data[group_id]['dims_data'][(n, p)].append(row_data)

            # --- 为当前维度、当前实验组绘图并保存 ---
            # 文件名示例: Q1_G1_n500_p10_cost.png
            file_suffix = f"{group_id}_n{n}_p{p}"
            plot_benchmark_results(
                group_results_for_plot, 
                group_name=f"{group['name']} (n={n}, p={p})",
                plot_fx=True, 
                plot_gradnorm=True,
                save_name=(fig_save_dir, f"Q1_{file_suffix}")
            )
            
    # --- 3. 生成 LaTeX 表格 ---
    # 定义内部函数用于写入 LaTeX
    def save_q1_latex(group_id, info_dict, save_dir):
        if not save_dir: return
        
        cn_title = info_dict['cn_name']
        dims_data_map = info_dict['dims_data']
        
        filename = os.path.join(save_dir, f"Q1_{group_id}.tex")
        
        header = r"""\begin{table}[H]
    \centering
    \small
    \caption{""" + cn_title + r"""}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{维度} $(n, p)$ & \textbf{方法} & \textbf{时间 (s)} & \textbf{迭代次数} & \textbf{最小} $f$ & \textbf{最终} $\|\nabla f\|$ \\
        \midrule
"""
        footer = r"""        \bottomrule
    \end{tabular}
    \label{tab:Q1_""" + group_id + r"""}
\end{table}
"""
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(header)
            
            # 按维度排序写入
            sorted_dims = sorted(dims_data_map.keys())
            
            for dim_idx, dim_key in enumerate(sorted_dims):
                n_val, p_val = dim_key
                rows = dims_data_map[dim_key]
                num_rows = len(rows)
                dim_str = f"$({n_val}, {p_val})$"
                
                # 第一行带有维度 Multirow
                first = rows[0]
                if num_rows > 1:
                    f.write(f"        \\multirow{{{num_rows}}}{{*}}{{{dim_str}}} ")
                else:
                    f.write(f"        {dim_str} ")
                
                f.write(f"& {first['method']} & {first['time']:.4f} & {first['iter']} & {first['min_f']:.4e} & {first['final_g']:.2e} \\\\\n")
                
                # 后续行
                for r in rows[1:]:
                    f.write(f"        & {r['method']} & {r['time']:.4f} & {r['iter']} & {r['min_f']:.4e} & {r['final_g']:.2e} \\\\\n")
                
                # 分隔线 (最后一行除外)
                if dim_idx < len(sorted_dims) - 1:
                    f.write(r"        \cmidrule{1-6}" + "\n")
            
            f.write(footer)
        print(f"LaTeX 表格已保存: {filename}")

    # 循环生成所有组的表格
    if text_save_dir:
        print("\n=== Generating LaTeX Tables ===")
        for gid, info in latex_tables_data.items():
            save_q1_latex(gid, info, text_save_dir)
        print("注意：需要在 LaTeX 导言区添加 \\usepackage{multirow} 和 \\usepackage{booktabs}")

def Q3(fig_save_dir=None, text_save_dir=None, test_damped=False,test_subspace=False):
    '''
    第三题：实现 Stiefel 流形 St(n, p) 上的 L-BFGS 算法性能评估。
    compare_variants: bool, 若为 True，则同时测试 SubspaceLBFGS 和 DampedLBFGS
    '''
    print(f"\n{'='*65}\nRunning Q3: Strictly Fair Scalability Benchmark\n{'='*65}")
    if test_damped:
        if test_subspace:
            print("Mode: Comparing [L-BFGS, Damped L-BFGS, Subspace L-BFGS]")
        else:
            print("Mode: Comparing [L-BFGS, Damped L-BFGS]")
    else:
        print("Mode: Standard [L-BFGS] only")
    print(f"{'-'*65}")

    experiments = {
        "Fixed_p": {
            "desc": "固定p=10, 增加 n",
            "dims": [
                (1000, 10),
                (2000, 10),
                (4000, 10),
                (6000, 10)            
            ]
        },
        "Fixed_n": {
            "desc": "固定n=2000, 增加 p",
            "dims": [
                (2000, 10),
                (2000, 50),
                (2000, 100),
                (2000, 200)            
            ]
        }
    }
    
    # 统一参数
    lbfgs_m = 10
    max_iters = 1000
    tol = 1e-6

    # 定义要测试的方法列表
    methods_to_test = [("L-BFGS", LBFGS(m=lbfgs_m))]
    if test_damped:
        methods_to_test.append(("Damped", DampedLBFGS(m=lbfgs_m, delta=20.0)))
    if test_subspace:
        methods_to_test.append(("Subspace", SubspaceLBFGS(max_dim=20, delta=1.0)))
    
    for exp_name, exp_config in experiments.items():
        print(f"\n>>> Running {exp_name}: {exp_config['desc']}")
        results_store = {}
        table_metrics = []
        
        for idx, (n, p) in enumerate(exp_config['dims']):
            # --- 统一数据生成 (保证对所有方法公平) ---
            np.random.seed(42 + n) 
            
            # 生成随机矩阵
            A_raw = np.random.randn(n, n).astype(np.float32)
            A = (A_raw + A_raw.T) / 2
            A = np.asfortranarray(A, dtype=np.float64) 
            del A_raw; gc.collect()
            
            B = np.asfortranarray(np.zeros((n, p)))
            x0_base, _ = np.linalg.qr(np.random.randn(n, p))
            
            # 初始化 Solver
            solver = StiefelSolver(n, p, A, B)

            # --- 遍历所选方法 ---
            for method_name, d_strat in methods_to_test:
                label = f"St({n},{p})-{method_name}"
                
                # 必须复制 x0，确保起点一致
                x0 = x0_base.copy()
                if isinstance(d_strat, LBFGS):
                    d_instance = type(d_strat)(m=lbfgs_m)
                    if hasattr(d_strat, 'delta'): # Damped
                        d_instance.delta = d_strat.delta
                elif isinstance(d_strat, SubspaceLBFGS):
                    d_instance = SubspaceLBFGS(max_dim=d_strat.max_dim, delta=d_strat.delta)
                else:
                    d_instance = d_strat

                # 进度条
                pbar = tqdm(total=max_iters, desc=f"   {label:<22}", leave=True,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
                
                start_t = time.time()
                _, f_vals, g_norms, iters = solver.solve(
                    x0, d_instance, FixedStep(1.0), 
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
                    "method": method_name,
                    "time": elapsed,
                    "iter": iters,
                    "t_per_k": t_per_k,
                    "grad": g_norms[-1],
                    "final_f": f_vals[-1]
                })

            # 清理大矩阵内存
            del A, solver, x0_base
            gc.collect()

        # --- 1. 打印 ASCII 表格到控制台 ---
        print(f"\nResults for {exp_name}:")
        print(f"{'Dims (n, p)':<15} | {'Method':<10} | {'Total Time':<10} | {'Iters':<6} | {'Time/Iter':<10} | {'Final |Grad|':<12} | {'Final f(X)':<14}")
        print("-" * 105)
        for row in table_metrics:
            dims_str = f"({row['n']}, {row['p']})"
            print(f"{dims_str:<15} | {row['method']:<10} | {row['time']:.4f}s    | {row['iter']:<6} | {row['t_per_k']:.6f}   | {row['grad']:.4e}   | {row['final_f']:.4e}")
        print("-" * 105)

        # --- 2. 保存 LaTeX 表格到文件（分组格式，使用 multirow）---
        if text_save_dir:
            tex_filename = os.path.join(text_save_dir, 'Q3'+f"{exp_name}.tex")
            
            # LaTeX 表格头部（需要 \usepackage{multirow} 和 \usepackage{booktabs}）
            header = r"""\begin{table}[H]
    \centering
    \small
    \caption{ """ + exp_config['desc'] + r"""}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{维度} $(n, p)$ & \textbf{方法} & \textbf{时间 (秒)} & \textbf{迭代次数} & \textbf{每次迭代时间 (秒)} & \textbf{最终} $\|\nabla f\|$ & \textbf{最优解} $f(X^*)$ \\
        \midrule
"""
            footer = r"""        \bottomrule
    \end{tabular}
    \label{tab:""" + "Q3" + exp_name + r"""}
\end{table}
"""
            try:
                with open(tex_filename, "w", encoding='utf-8') as f:
                    f.write(header)
                    
                    # 按维度分组
                    from collections import defaultdict
                    grouped_data = defaultdict(list)
                    for row in table_metrics:
                        dims_key = (row['n'], row['p'])
                        grouped_data[dims_key].append(row)
                    
                    # 遍历每个维度组
                    for group_idx, (dims_key, group_rows) in enumerate(grouped_data.items()):
                        n, p = dims_key
                        dims_str = f"$({n}, {p})$"
                        num_methods = len(group_rows)
                        
                        # 第一行：使用 multirow 合并维度列
                        first_row = group_rows[0]
                        if num_methods > 1:
                            f.write(f"        \\multirow{{{num_methods}}}{{*}}{{{dims_str}}} ")
                        else:
                            f.write(f"        {dims_str} ")
                        f.write(f"& {first_row['method']} & {first_row['time']:.4f} & {first_row['iter']} & {first_row['t_per_k']:.6f} & {first_row['grad']:.2e} & {first_row['final_f']:.4e} \\\\\n")
                        
                        # 后续行：维度列为空（由 multirow 填充）
                        for row in group_rows[1:]:
                            f.write(f"        & {row['method']} & {row['time']:.4f} & {row['iter']} & {row['t_per_k']:.6f} & {row['grad']:.2e} & {row['final_f']:.4e} \\\\\n")
                        
                        # 组间分隔线（最后一组除外）
                        if group_idx < len(grouped_data) - 1:
                            f.write(r"        \cmidrule{1-7}" + "\n")
                    
                    f.write(footer)
                print(f"LaTeX 表格已保存至: {os.path.abspath(tex_filename)}")
                print(f"注意：需要在 LaTeX 导言区添加 \\usepackage{{multirow}} 和 \\usepackage{{booktabs}}")
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
    Q1(fig_save_dir=fig_save_dir, text_save_dir=text_save_dir)
    Q3(fig_save_dir=fig_save_dir, text_save_dir=text_save_dir, test_damped=True, test_subspace=False)