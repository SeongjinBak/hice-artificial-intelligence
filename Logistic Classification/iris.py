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

# 훈련용 데이터, 테스트용 데이터 x 설정.
x_train = X[for_train]
x_test = X[for_test]

# 타겟들을 One-hot encoding 으로 변환
num = np.unique(y[for_train], axis=0)
num = num.shape[0]
y_train = np.eye(num)[y[for_train]].astype(np.int)

num = np.unique(y[for_test], axis=0)
num = num.shape[0]
y_test = np.eye(num)[y[for_test]].astype(np.int)

# Logistic Regression 을 위한 상세 파라미터 지정
feature_num = 4
class_num = 3
learning_rate = 0.001
single_lr = False  # True 면, Single class lr, False 면 Multiple class lr

# x 데이터에 1로 된 1개열 추가(바이어스 열)
offset = np.array([[1] for i in range(x_train.shape[0])])
x_train = np.append(x_train, offset, 1)

# Multiple class logistic regression
if not single_lr:
    # weight matrix
    li = np.random.rand((feature_num + 1) * class_num).reshape(feature_num + 1, class_num).astype(np.float64)
    # Logistic Regression 오브젝트 생성
    logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, y_train, li, learning_rate)

# Single class logistic regression
if single_lr:
    # weight matrix
    li = np.random.rand((feature_num + 1)).reshape(feature_num + 1, 1).astype(np.float64)
    # targets
    # 싱글 클래스로 분류하고 싶은 클래스의 번호를 지정한다.
    search_class_num = 2
    # 'setosa' 를 분류하고 싶은 경우, One hot encoding 된 것중 0번째 인덱스가 1인 것만을 1로, 나머지는 0으로 지정.
    y_train = np.zeros(y[for_train].shape[0])
    for i in range(y[for_train].shape[0]):
        if y[for_train][i] == search_class_num:  # 인덱스와 찾고자 하는 클래스의 번호가 일치하면 1로, 그 외는 0으로.
            y_train[i] = 1
        else:
            y_train[i] = 0
    # Single class Logistic Regression 오브젝트 생성
    logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, y_train, li, learning_rate,
                                             True)

# Cost/ Epoch 그래프를 위한 리스트
cost_list = []
epoch_list = []

# 학습과 그 Cost 계산. 함수의 매개변수는 몇개의 데이터를 한번에 학습시킬지를 의미.

# 1000 8 8 0.001
# 1000 40 40 0.001

# Epoch
epoch = 1000
for i in range(epoch):
    logistic_regression.learn(50)  # 학습 실시
    cost = logistic_regression.cost(50)  # Cost 계산
    cost_list.append(cost)
    epoch_list.append(i + 1)
    print("epoch", i, "|", cost)

# plt 를 사용하여 그래프 생성
plt.title('B511086 Seongjin Bak Logistic Regression for Iris datum')
plt.plot(epoch_list, cost_list)
plt.xlabel('Number of iterations', fontsize=10)
plt.ylabel('Cost', fontsize=10)
# 그래프 출력
plt.show()

# Multiple class logistic regression 에서의 예측
if not single_lr:
    logistic_regression.predict(x_test, y_test, x_test.shape[0], 0, epoch, "iris")

# Single class logistic regression 에서의 예측
if single_lr:
    logistic_regression.predict(x_test, y_test, x_test.shape[0], search_class_num, epoch, "iris")
