"""
B511086 백성진
MNIST 데이터를 logistic classification 하는 python file 입니다.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# 부모 디렉터리에서 import 할 수 있도록 설정
sys.path.append(os.pardir)

# mnist data load
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False, one_hot_label=True)

label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# test data 개수
size = 100

# 0~10000 사이의 랜덤한 size 개 만큼의 데이터 인덱스 선정
sample = np.random.randint(0, t_test.shape[0], size)
sample_test_image = []
sample_test_label = []

# 랜덤하게 선정된 데이터 인덱스의 실제 데이터를 리스트에 저장
for i in sample:
    sample_test_image.append(x_test[i])
    sample_test_label.append(t_test[i])

# Logistic Regression 을 위한 상세 파라미터 지정
feature_num = 784
class_num = 10
learning_rate = 0.001

# True 면, Single class lr, False 면 Multiple class lr
single_lr = True

# x 데이터에 1로 된 1개열 추가(바이어스)
offset = np.array([[1] for i in range(x_train.shape[0])])
x_train = np.append(x_train, offset, 1)

# Multiple class logistic regression
if not single_lr:
    # weight matrix
    li = np.random.randint(1, 5, (feature_num + 1) * class_num).reshape((feature_num + 1), class_num).astype(np.float64)
    # Logistic Regression 오브젝트 생성
    logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, t_train, li, learning_rate)

# Single class logistic regression
if single_lr:
    lr_list = []
    # Single class Logistic Regression 오브젝트 생성
    for iters in range(10):
        # targets
        # 숫자 0 을 분류하고 싶은 경우, 인덱스가 0인 것을 제외하고 다 0으로 변환.
        search_class_num = iters
        y_train = np.zeros(t_train.shape[0])  # 0인 배열 생성
        for i in range(t_train.shape[0]):
            ai = np.argmax(t_train[i])  # one hot encoding 된 것들 중 가장 큰것은 1이며, 그 인덱스 반환
            if ai == search_class_num:  # 인덱스와 찾고자 하는 클래스의 번호가 일치하면 1로, 그 외는 0으로.
                y_train[i] = 1
            else:
                y_train[i] = 0
        y_train = y_train.reshape(t_train.shape[0], 1)
        # weight matrix
        li = np.random.rand((feature_num + 1)).reshape(feature_num + 1, 1).astype(np.float64)
        lr_list.append(LogisticRegression(x_train, x_train.shape[0], feature_num + 1, y_train, li, learning_rate, True))

# Cost/ Epoch 그래프를 위한 리스트
cost_list = []
count_list = []

# 학습과 그 Cost 계산. 함수의 매개변수는 몇개의 데이터를 한번에 학습시킬지를 의미.
# epoch
epoch = 100
for i in range(epoch):
    if not single_lr:
        logistic_regression.learn(60000)  # 학습
        cost = logistic_regression.cost(60000)  # 코스트
        print("epoch", i, cost)
        cost_list.append(cost)
        count_list.append(i + 1)
    else:
        tmp_li = []     # 각 class 별로 cost 를 한꺼번에 출력하기 위한 리스트
        for j in range(10):
            lr_list[j].learn(60000)  # 학습
            cost = lr_list[j].cost(60000)  # 코스트
            tmp_li.append(cost)
        cost_list.append(tmp_li)
        count_list.append(i + 1)
        print("epoch", i, "|", tmp_li)

# plt 를 사용하여 산포도 그래프 생성
plt.title('B511086 Seongjin Bak Logistic Regression for MNIST datum')
plt.plot(count_list, cost_list)
plt.legend(labels=('class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8',
                   'class 9'))
plt.xlabel('Number of iterations', fontsize=10)
plt.ylabel('Cost', fontsize=10)
# 그래프 출력
plt.show()

# Multiple class logistic regression 에서의 예측
if not single_lr:
    logistic_regression.predict(sample_test_image, sample_test_label, size, 0, epoch, "MNIST")

# Single class logistic regression 에서의 예측
if single_lr:
    for i in range(10):
        lr_list[i].predict(sample_test_image, sample_test_label, size, i, epoch, "MNIST")
