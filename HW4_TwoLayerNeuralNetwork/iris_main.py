# B511086 백성진

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

# 150개 중 train 데이터로 중복없는 120개의 데이터를, 테스트 데이터로 나머지 30개 데이터를 얻는다.
for_train = np.array(random.sample(range(y.shape[0]), 120))
for_test = np.array([i for i in range(y.shape[0]) if np.int(i) not in for_train])

# 훈련용 데이터, 테스트용 데이터 x 설정.
x_train = X[for_train]
#y_train = y[for_train]

x_test = X[for_test]
#y_test = y[for_test]

# 타겟들을 One-hot encoding 으로 변환
num = np.unique(y[for_train], axis=0)
num = num.shape[0]
y_train = np.eye(num)[y[for_train]].astype(np.int)

num = np.unique(y[for_test], axis=0)
num = num.shape[0]
y_test = np.eye(num)[y[for_test]].astype(np.int)


# 2000 20 0.05 5
iter_num = 1000
train_size = x_train.shape[0]
batch_size = 20
learning_rate = 0.05
hidden_layer_num = 5

two_layer_network = NeuralNetworkClass(4, hidden_layer_num, 3, learning_rate)

iter_per_epoch = max(train_size / batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iter_num):
    # 미니배치 획득 ( 120개 중 batch size 개 만큼)
    batch_sample = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_sample]
    y_batch = y_train[batch_sample]

    # 배치 데이터 저장
    two_layer_network.set_batch_data(x_batch, y_batch)

    # 학습
    two_layer_network.learn()

    # 학습 진행상황 기록
    loss = two_layer_network.loss()

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = two_layer_network.accuracy(x_train, y_train)
        test_acc = two_layer_network.accuracy(x_test, y_test)
        train_loss_list.append(loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i, "train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기


plt.plot(train_acc_list, label='train acc')
plt.plot(train_loss_list, label='train loss')

plt.legend(loc='upper right')
plt.show()