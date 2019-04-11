# coding: utf-8
import sys, os
from common.functions import sigmoid, softmax, corss_entropy_error
from common.gradient import numerical_gradient
import numpy as np

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 进行初始化。
        # 参数从头开始依次表示输入层的神经元数、隐藏层的神经元数、输出层的神经元数
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        # 进行识别（推理）
        # x: 图像数据
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    def loss(self, x, t):
        # 计算损失函数的值。
        # 参数x 是图像数据，t 是正确解标签
        y = self.predict(x)
        return corss_entropy_error(y, t)
    
    def accuracy(self, x, t):
        # 计算识别精度
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 使用数值法求梯度
    def numerical_gradient(self, x, t):
        # 计算权重参数的梯度
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads