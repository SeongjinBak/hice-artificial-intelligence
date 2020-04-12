import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from knn import KNN

iris = load_iris()
# print(iris)

X = iris.data  # iris data input
y = iris.target  # iris target (label)
y_name = iris.target_names

# print(y_name)

# 트레이닝 데이터와 테스트 데이터를 나누는 부분.
l = 15

# y 가 타겟이므로 (150,)이다. 150만 가져오기 위해 y.shape[0].
# 0부터 149까지의 i에 대해 l로 나누어 나머지가 l-1과 같은 원소를 테스트 데이터로 삼는다.(true)
# 나머지 테스트 데이터가 아닌 데이터들은 트레이닝 데이터로 삼는다.(false) 그 후 넘파이의 array로 바꾼다,
for_test = np.array([(i % l == (l - 1)) for i in range(y.shape[0])])
for_train = ~for_test

# print(for_test)
# print(for_train)

# 원래의 것에 넣어주면, true인 것만 x_train에 들어간다.
X_train = X[for_train]
y_train = y[for_train]

X_test = X[for_test]
y_test = y[for_test]

# print(X_train)
# print(y_train)
# knn_iris = KNN(10, X_train, y_train, y_name)
# knn_iris.show_dim()

# print(X_train[0])
# print(knn_iris.distance_metric(X_train[0], X_train[1]))

# for i in range(y_test.shape[0]):
    # knn_iris.get_nearest_k(X_test[i])
    # print("Test Data: ", i, " Computed class: ", knn_iris.weighted_majority_vote(), ", \tTrue class: ", y_name[y_test[i]])
    # knn_iris.reset()
