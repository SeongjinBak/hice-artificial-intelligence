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
learning_rate = 0.005

li = np.random.randint(1, 5, (feature_num + 1) * class_num).reshape((feature_num + 1), class_num).astype(np.float64)

offset = np.array([[1] for i in range(x_train.shape[0])])
x_train = np.append(x_train, offset, 1)

logistic_regression = LogisticRegression(x_train, x_train.shape[0], feature_num + 1, t_train, li, learning_rate)

# epoch
for i in range(100):
    logistic_regression.learn(1000)
    print("epoch", i, logistic_regression.cost(1000))

for i in range(size):
    logistic_regression.predict(sample_test_image[i], sample_test_label[i])

for i in range(10):
    if logistic_regression.aicount[i] != 0:
        print(i, logistic_regression.ailist[i] / logistic_regression.aicount[i])
