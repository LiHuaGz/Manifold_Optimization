import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time

# 配置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def stiefel_retraction(X, Z):
    # X: 当前点，Z: 切空间方向
    U, _, Vt = np.linalg.svd(X + Z, full_matrices=False)
    return U @ Vt

def project_grad_stiefel(X, G):
    # 投影梯度到Stiefel流形的切空间
    sym = (X.T @ G + G.T @ X) / 2
    return G - X @ sym

class Stiefel_Optimizer:
    def __init__(self, n, p, f, grad_f, step_size=0.01):
        self.n = n
        self.p = p
        self.f = f
        self.grad_f = grad_f
        self.step_size = step_size

    def gradient_descent(self, X, max_iters=1000, tol=1e-6):
        '''
        在Stiefel流形上执行梯度下降
        X: 初始点，形状为(n, p)
        max_iters: 最大迭代次数
        tol: 收敛容忍度
        returns: 优化后的点X, 目标函数值变化列表f_vals
        '''
        def armijo_step_size(X, grad_proj, t_init=1.0, rho=0.5, c=1e-4, max_iters=20):
            '''
            使用Armijo条件确定步长
            X: 当前点
            grad_proj: 投影到切空间的梯度
            t_init: 初始步长
            rho: 步长缩减因子
            c: Armijo条件参数
            max_iters: 最大迭代次数
            returns: 合适的步长t, 新点X_new, 新点处的函数值f_X_new
            '''
            t = t_init
            f_X = self.f(X)
            for _ in range(max_iters):
                X_new = stiefel_retraction(X, -t * grad_proj)
                f_X_new = self.f(X_new)
                grad_normFro = np.linalg.norm(grad_proj, 'fro')**2
                if f_X_new <= f_X - c * t * grad_normFro:
                    return t, X_new, f_X_new
                t *= rho
            return t, X_new, f_X_new
        
        f_vals = [self.f(X)]
        for _ in range(max_iters):
            # 计算投影到切空间的梯度
            grad_proj = project_grad_stiefel(X, self.grad_f(X))

            # armijo找步长
            _, X_new, f_X_new = armijo_step_size(X, grad_proj)
            f_vals.append(f_X_new)

            # 检查收敛
            if np.linalg.norm(X_new - X, 'fro') < tol:
                break
            X = X_new

        return X, f_vals

def main():
    # 定义St(n, p)空间的维度
    n, p = 500, 5

    # 定义目标二次函数
    A = np.random.randn(n, n)
    A = A.T @ A  # 使A对称正定
    B = np.zeros((n, p))
    def f(X):
        # 目标函数: f(X) = trace(X^T A X) + 2 * trace(B^T X)
        return np.einsum('ij,ij->', X, A @ X) + 2 * np.einsum('ij,ij->', B, X)
    
    def grad_f(X):
        return 2 * (A @ X + B)
    
    # 初始化优化器
    optimizer = Stiefel_Optimizer(n, p, f, grad_f, step_size=0.1)
    
    start_time = time.time()

    # 初始化Stiefel流形上的点
    x0 = np.linalg.qr(np.random.randn(n, p))[0]

    # 运行梯度下降
    X_opt, f_vals = optimizer.gradient_descent(x0, max_iters=5000, tol=1e-8)

    end_time = time.time()
    print(f"优化完成，耗时: {end_time - start_time:.2f} 秒")

    # 绘制目标函数值变化曲线, semilogy
    plt.semilogy(f_vals)
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title('目标函数值随迭代次数的变化')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 初始化随机种子
    np.random.seed(42)
    main()