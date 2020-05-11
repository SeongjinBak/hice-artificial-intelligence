# B511086 백성진 함수 모음

import numpy as np

# 시그모이드 함수
eMin = -np.log(np.finfo(type(0.1)).max)


def sigmoid(z):
    zSafe = np.array(np.maximum(z, eMin))
    return 1.0 / (1 + np.exp(-zSafe))


# 소프트맥스 함수
def softmax(x):
    # exp_x = np.exp(x)   # x가 커질수록 값이 너무 커지므로 컴퓨터가 감당 불가 하므로 아래와같이 오버플로우 대비한다
    exp_x = np.exp(x - np.max(x))
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


# 렐루 함수
def reLU(x):
    return np.maximum(x, 0)

