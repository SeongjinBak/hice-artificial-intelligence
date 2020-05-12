"""
B511086 백성진
Iris 데이터를 logistic classification 하는 python file 입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris
from logistic_regression import LogisticRegression

# Iris 데이터 로드
iris = load_iris()

X = iris.data
y = iris.target
y_name = iris.target_names

# Training : Test  = 6 : 4 for Iris classification
# 150개 중 train 데이터로 중복없는 100개의 데이터를, 테스트 데이터로 나머지 50개 데이터를 얻는다.
for_train = np.array(random.sample(range(y.shape[0]), int((y.shape[0] * 4) / 6)))
for_test = np.array([i for i in range(y.shape[0]) if np.int(i) not in for_train])
x_train = X[for_train]

x_test = X[for_test]

# 타겟들을 One-hot encoding 으로 변환
num = np.unique(y[for_train], axis=0)
num = num.shape[0]
y_train = np.eye(num)[y[for_train]].astype(np.int)

num = np.unique(y[for_test], axis=0)
num = num.shape[0]
y_test = np.eye(num)[y[for_test]].astype(np.int)

feature_num = 4
class_num = 3
learning_rate = 0.001
single_lr = True

# x 데이터에 1로 된 1개열 추가(바이어스)
offset = np.array([[1] for i in range(x_train.shape[0])])
x_train = np.append(x_train, offset, 1)

# 멀티 LR
if not single_lr:
    # weight matrix
    li = np.random.randint(1, 5, (feature_num + 1) * class_num).reshape(feature_num + 1, class_num).astype(np.float64)
    logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, y_train, li, learning_rate)

# 싱글 LR
if single_lr:
    # weight matrix
    li = np.random.rand((feature_num + 1)).reshape(feature_num + 1, 1).astype(np.float64)
    # targets
    # setosa 를 분류하고 싶은 경우, 인덱스가 0인 것을 제외하고 다 0으로.
    search_class_num = 2
    y_train = np.zeros(y[for_train].shape[0])
    for i in range(y[for_train].shape[0]):
        if y[for_train][i] == search_class_num:
            y_train[i] = 1
        else:
            y_train[i] = 0
    print(y_train.shape)
    logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, y_train, li, learning_rate,
                                             True)
# 학습과 cost 계산
for i in range(10000):
    logistic_regression.learn(8)
    cost = logistic_regression.cost(8)
    print("epoch", i, "|", cost)

if not single_lr:
    logistic_regression.predict(x_test, y_test, x_test.shape[0])
if single_lr:
    logistic_regression.predict(x_test, y_test, x_test.shape[0], search_class_num)
