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

li = np.random.randint(1, 5, (feature_num + 1) * class_num).reshape(feature_num + 1, class_num).astype(np.float64)
offset = np.array([[1] for i in range(x_train.shape[0])])
x_train = np.append(x_train, offset, 1)

logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, y_train, li, learning_rate)

for i in range(1000):
    logistic_regression.learn(100)
    print("epoch", i, logistic_regression.cost(100))

for i in range(x_test.shape[0]):
    logistic_regression.predict(x_test[i], y_test[i])

for i in range(3):
    print(i, logistic_regression.ailist[i] / logistic_regression.aicount[i])
