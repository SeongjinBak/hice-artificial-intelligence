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

# Training : Test  = 6 : 4 for Iris classification
# 150개 중 train 데이터로 중복없는 100개의 데이터를, 테스트 데이터로 나머지 50개 데이터를 얻는다.
for_train = np.array(random.sample(range(y.shape[0]), 120))
for_test = np.array([i for i in range(y.shape[0]) if np.int(i) not in for_train])

# 훈련용 데이터, 테스트용 데이터 x 설정.
x_train = X[for_train]
y_train = y[for_train]

x_test = X[for_test]
y_test = y[for_test]

two_layer_network = NeuralNetworkClass(4, 5, 3)

iter_num = 2000
train_size = x_train.shape[0]
batch_size = 20


learning_rate = 0.05

iter_per_epoch = max(train_size/batch_size, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iter_num):
    # 미니배치 획득 ( 120개 중 batch size 개 만큼)
    batch_sample = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_sample]
    y_batch = y_train[batch_sample]

    # 기울기 계산
    grad = two_layer_network.numerical_gradient(x_batch, y_batch)

    # 매개변수 갱신
    two_layer_network.params['W1'] -= learning_rate * grad['W1']
    two_layer_network.params['b1'] -= learning_rate * grad['b1']
    two_layer_network.params['W2'] -= learning_rate * grad['W2']
    two_layer_network.params['b2'] -= learning_rate * grad['b2']

    # 학습 진행상황 기록
    loss = two_layer_network.loss(x_batch, y_batch)
    #train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = two_layer_network.accuracy(x_train, y_train)
        test_acc = two_layer_network.accuracy(x_test, y_test)
        train_loss_list.append(loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i, "train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, train_loss_list, label='train loss')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()