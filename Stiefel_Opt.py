import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time
import os
from numba import jit, prange
import heapq
# 自动检测CPU核心数并自动设置
num_cores = os.cpu_count()
print(f"检测到 CPU 核心数: {num_cores}, 已设置相关环境变量以优化性能。")
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_cores)

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

def project_grad_stiefel(X, G):
    # 投影梯度到Stiefel流形的切空间
    sym = (X.T @ G + G.T @ X) / 2
    return G - X @ sym

class Stiefel_Optimizer:
    def __init__(self, n, p, A, B, step_size=0.01):
        self.n = n
        self.p = p
        self.A = A
        self.B = B
        self.step_size = step_size

    def gradient_descent(self, X, max_iters=1000, tol=1e-6, num_history=2, search_method='armijo', alpha_min=1e-5, alpha_max=1e5):
        '''
        在Stiefel流形上执行梯度下降
        X: 初始点，形状为(n, p)
        max_iters: 最大迭代次数
        tol: 收敛容忍度
        returns: 优化后的点X, 目标函数值变化列表f_vals
        '''
        def linear_search(X, f_X, grad_proj, grad_normFro_sq, t_init=1.0, rho=0.5, c=1e-4, max_iters=20, last_max_f=float('inf')):
            '''
            先搜索
            X: 当前点
            f_X: 当前点的函数值
            grad_proj: 投影到切空间的梯度
            t_init: 初始步长
            rho: 步长缩减因子
            c: Armijo条件参数
            max_iters: 最大迭代次数
            returns: 合适的步长t, 新点X_new, 新点处的函数值f_X_new
            '''
            t = t_init
            for _ in range(max_iters):
                X_new = stiefel_retraction(X, -t * grad_proj, svd=True)
                f_X_new = f(X_new, self.A, self.B)
                if f_X_new <= last_max_f - c * t * grad_normFro_sq:
                    return t, X_new, f_X_new
                t *= rho
            return t, X_new, f_X_new
        
        f_X = f(X, self.A, self.B)
        f_vals = [None] * max_iters
        f_vals[0] = f_X
        t_init = 1.0

        for i in range(1, max_iters):
            f_X = f_vals[i-1]

            # 计算投影到切空间的梯度 
            grad_proj = project_grad_stiefel(X, grad_f(X, self.A, self.B))
            
            # 【优化】提前计算梯度范数用于Armijo和收敛检查
            grad_normFro_sq = np.linalg.norm(grad_proj, 'fro')**2

            # 读取过去num_history个函数值的最大值
            last_max_f = max(f_vals[max(0, i - num_history):i])

            # 选择步长搜索方法
            if search_method == 'armijo':
                t, X_new, f_X_new = linear_search(X, f_X, grad_proj, grad_normFro_sq, t_init=t_init, last_max_f=last_max_f)
            elif search_method == 'BB_step':
                if i > 1:
                    s, y = X - X_prev, grad_proj - grad_proj_prev
                    alpha_LBB = np.dot(s.flatten(), s.flatten()) / np.abs(np.dot(s.flatten(), y.flatten()))
                    alpha_SBB = np.abs(np.dot(s.flatten(), y.flatten()) / np.dot(y.flatten(), y.flatten()))
                    alpha_init = alpha_LBB if i % 2 == 0 else alpha_SBB
                else:
                    alpha_init = t_init
                alpha_init = min(alpha_max, max(alpha_min, alpha_init))
                t, X_new, f_X_new = linear_search(X, f_X, grad_proj, grad_normFro_sq, t_init=alpha_init, last_max_f=last_max_f)
            else:
                # 报错
                raise ValueError("未知的步长搜索方法")

            f_vals[i] = f_X_new

            # 【优化】使用已计算的梯度范数检查收敛
            if np.sqrt(grad_normFro_sq) < tol:
                break
            X_prev, grad_proj_prev = X, grad_proj
            X = X_new

        return X, f_vals[:, i+1]

def f(X, A, B):
    # 目标函数: f(X) = trace(X^T A X) + 2 * trace(B^T X)
    AX = A @ X
    return np.sum(X * AX) + 2 * np.sum(B * X)

def grad_f(X, A, B):
    return 2 * (A @ X + B)

def main():
    # 定义St(n, p)空间的维度
    n, p = 500, 5

    # 定义目标二次函数
    A = np.random.rand(n, n)
    A = A.T @ A  # 使A对称正定
    B = np.zeros((n, p))

    # 初始化优化器
    optimizer = Stiefel_Optimizer(n, p, A, B, step_size=0.1)
    
    start_time = time.time()

    # 初始化Stiefel流形上的点
    x0 = np.linalg.qr(np.random.randn(n, p))[0]

    # 运行梯度下降
    X_opt, f_vals_armijo = optimizer.gradient_descent(x0, max_iters=5000, tol=1e-8, num_history=2, search_method='armijo')

    armijo_time = time.time()
    print(f"优化完成，耗时: {armijo_time - start_time:.2f} 秒")

    # 使用BB步长运行梯度下降
    X_opt, f_vals_BB = optimizer.gradient_descent(x0, max_iters=5000, tol=1e-8, num_history=2, search_method='BB_step')
    end_time = time.time()
    print(f"优化完成，耗时: {end_time - armijo_time:.2f} 秒")

    # 绘制目标函数值变化曲线, semilogy
    plt.semilogy(f_vals_armijo, label='Armijo步长')
    plt.semilogy(f_vals_BB, label='BB步长')
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 初始化随机种子
    np.random.seed(42)
    main()