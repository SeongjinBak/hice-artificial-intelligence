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

# 2000 20 0.05 5

# 학습 반복 횟수
iter_num = 900
# 입력 훈련용 데이터의 크기
train_size = x_train.shape[0]
# 배치 사이즈. 120개 중 batch_size 개를 사용합니다.
batch_size = 20
# 학습률
learning_rate = 0.01
# 히든 레이어의 개수
hidden_layer_num = 5

# 클래스 오브젝트 생성. 4는 input, 3은 output class 을 의미합니다.
two_layer_network = NeuralNetworkClass(4, hidden_layer_num, 3, learning_rate)


# 그래프를 위한 매개 리스트
loss_list = []
train_acc_list = []
test_acc_list = []

# 최종 정확도를 위한 변수 입니다.
final_test_acc = None
final_training_acc = None

for i in range(iter_num):
    # 미니배치 획득 ( 120개 중 batch size 개 만큼)
    batch_sample = np.random.choice(train_size, batch_size)

    # 얻어낸 샘플 데이터의 인덱스를 batch 데이터로 할당합니다.
    x_batch = x_train[batch_sample]
    y_batch = y_train[batch_sample]

    # 배치 데이터를 클래스 오브젝트에 저장합니다.
    two_layer_network.set_batch_data(x_batch, y_batch)

    # 학습 실시합니다.
    two_layer_network.learn()

    # 손실 값 기록
    loss = two_layer_network.loss()

    # 훈련용 데이터의 정확도 기록
    train_acc = two_layer_network.accuracy(x_train, y_train)

    # 테스트 데이터의 정확도 기록
    test_acc = two_layer_network.accuracy(x_test, y_test)
    
    loss_list.append(loss)
    train_acc_list.append(train_acc)
    print(i, 'cost, accuracy', loss, train_acc)

    if i + 1 == iter_num:
        final_test_acc = test_acc
        final_training_acc = train_acc

# 최종 정확도 출력합니다.
print('Training Accuracy =', final_training_acc)
print('Test Accuracy =', final_test_acc)

# 그래프 출력
plt.plot(loss_list, label='loss')
plt.plot(train_acc_list, label='training accuracy')
plt.legend(loc='upper right')
plt.show()
