# B511086 백성진 HW4

import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_iris

from neural_network_class import NeuralNetworkClass

# Iris 데이터 로드
iris = load_iris()

X = iris.data
y = iris.target
y_name = iris.target_names

# 150개 중 train 데이터로 중복없는 120개의 데이터를, 테스트 데이터로 나머지 30개 데이터를 얻습니다.
for_train = np.array(random.sample(range(y.shape[0]), 120))
for_test = np.array([i for i in range(y.shape[0]) if np.int(i) not in for_train])

# 훈련용 데이터, 테스트용 데이터 x 설정.
x_train = X[for_train]
x_test = X[for_test]

# 타겟들을 One-hot encoding 으로 변환합니다.
num = np.unique(y[for_train], axis=0)
num = num.shape[0]
y_train = np.eye(num)[y[for_train]].astype(np.int)

num = np.unique(y[for_test], axis=0)
num = num.shape[0]
y_test = np.eye(num)[y[for_test]].astype(np.int)

# 학습 반복 횟수
epoch = 20
# 입력 훈련용 데이터의 크기
train_size = x_train.shape[0]
# 배치 사이즈. 120개 중 batch_size 개를 사용합니다.
batch_size = 20
# 학습률
learning_rate = 0.005
# 히든 레이어의 개수
hidden_layer_num = 5

# 클래스 오브젝트 생성. 4는 input, 3은 output class 을 의미합니다.
two_layer_network = NeuralNetworkClass(4, hidden_layer_num, 3)

# 훈련용 데이터 저장
two_layer_network.set_train_data(x_train, y_train)

# 테스트용 데이터 저장
two_layer_network.set_test_data(x_test, y_test)

# 미니배치로 학습을 실시합니다.
# 학습은 총 epoch 번 이루어 지며, epoch 당 batch_size 만큼 나누어 진행됩니다.
two_layer_network.learn(learning_rate, epoch, batch_size)

# 그래프를 출력합니다.
plt.plot(two_layer_network.loss_list, label='loss')
plt.plot(two_layer_network.acc_list, label='training accuracy')
plt.legend(loc='upper right')
plt.show()
