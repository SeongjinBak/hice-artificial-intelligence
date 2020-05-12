"""
B511086 백성진
MNIST 데이터를 logistic classification 하는 python file 입니다.
"""
import sys
import os
import numpy as np
from PIL import Image
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

feature_num = 784
class_num = 10
learning_rate = 0.005  # multy  0.005 잘됨
single_lr = True

# x 데이터에 1로 된 1개열 추가(바이어스)
offset = np.array([[1] for i in range(x_train.shape[0])])
x_train = np.append(x_train, offset, 1)
# 멀티 LR
if not single_lr:
    # weight matrix
    li = np.random.randint(1, 5, (feature_num + 1) * class_num).reshape((feature_num + 1), class_num).astype(np.float64)
    logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, t_train, li, learning_rate)

if single_lr:
    # weight matrix
    li = np.random.rand((feature_num + 1)).reshape(feature_num + 1, 1).astype(np.float64)
    # targets
    # 숫자 0 을 분류하고 싶은 경우, 인덱스가 0인 것을 제외하고 다 0으로.
    search_class_num = 5
    y_train = np.zeros(t_train.shape[0])
    for i in range(t_train.shape[0]):
        ai = np.argmax(t_train[i])
        if ai == search_class_num:
            y_train[i] = 1
        else:
            y_train[i] = 0
    y_train = y_train.reshape(60000, 1)
    logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, y_train, li, learning_rate,
                                             True)
# epoch
# multi epoch 100, 1000, 0.005
for i in range(1000):
    logistic_regression.learn(1000)
    print("epoch", i, logistic_regression.cost(1000))

if not single_lr:
    logistic_regression.predict(sample_test_image, sample_test_label, size)
if single_lr:
    logistic_regression.predict(sample_test_image, sample_test_label, size, search_class_num)
