# coding: utf-8
import numpy as np

# 单个点求梯度
def _numerical_gradient_1d(func, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # 计算f(x+h)
        x[idx] = tmp_val + h
        fxh1 = func(x)
        # 计算f(x-h)
        x[idx] = tmp_val - h
        fxh2 = func(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        # 还原x
        x[idx] = tmp_val
    return grad


def numerical_gradient(func, X):
    # 单个点
    if X.ndim == 1:
        return _numerical_gradient_1d(func, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(func, x)
        return grad

