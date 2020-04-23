# B511086 백성진 KNN 과제.

import math


class KNN:
    def __init__(self, k, x, y, iris_names):
        self.k = k  # k 값
        self.features = x
        self.target = y
        self.iris_names = iris_names
        self.weighted_majority_vote_value = 0  # 가중치 고려된 majority vote 후 근접한 꽃
        self.majority_vote_value = ''  # majority vote 후 근접한 꽃
        self.nearest_k = []  # K 개의 테스트 데이터와 근접한 꽃의 인덱스
        self.nearest_dist = []  # 테스트 데이터와 K 개의 근접한 꽃의 유클리디안 길이

    # 두 4차원 벡터의 거리를 유클리디안 거리 계산 방식으로 구한다.
    def distance_metric(self, a, b):
        dist = 0
        for i in range(len(self.features[0])):
            sub = a[i] - b[i]
            dist += (sub * sub)
        return math.sqrt(dist)

    # 가장 가까운 k개의 데이터가 어떤 것인지 찾아내는 함수(k는 3, 5, 10..)
    def get_nearest_k(self, x_test):
        distance_dic = {}
        for iter_cnt in range(len(self.features)):
            # distance_dic 딕셔너리에 key 값이 거리, value 값이 데이터의 번호인 원소를 넣는다.
            distance_dic[self.distance_metric(x_test, self.features[iter_cnt])] = iter_cnt

        # 계산된 거리에 대해 오름차순으로 정렬한다.
        sdic = sorted(distance_dic.items())

        # 거리가 짧은 원소부터 k개 수집한다.
        for iter_cnt in range(self.k):
            self.nearest_k.append(sdic[iter_cnt][1])
            self.nearest_dist.append((sdic[iter_cnt][0]))

    # 일반적으로 근접한 k 개 중 가장 많은 수를 차지한 class 를 선정하는 함수.
    def majority_vote(self):
        majority_cnt = []

        # class 개수의 길이를 가지는 리스트를 만든다.
        for i in range(len(self.iris_names)):
            majority_cnt.append(0)

        # nearest k의 이름에 해당하는 class 의 카운트를 증가시킨다.
        for i in range(self.k):
            majority_cnt[self.target[self.nearest_k[i]]] += 1

        # 가장 많은 꽃 class 가 속한 이름을 찾아 저장한다.
        self.majority_vote_value = self.iris_names[majority_cnt.index(max(majority_cnt))]
        return self.majority_vote_value

    # 가중치 계산된 majority vote 를 계산하는 함수.
    # 가중치는 거리의 역수의 로그값이며, 해당하는 class 에 계산된 가중치를 곱한것을 더하는 식으로 분류를 하였음.
    def weighted_majority_vote(self):
        majority_cnt = []

        # class 개수의 길이를 가지는 리스트를 만든다.
        for i in range(len(self.iris_names)):
            majority_cnt.append(0)

        # nearest k의 이름에 해당하는 class 의 카운트를 증가시킨다.
        for i in range(self.k):
            # 가중치는 거리의 역수로 취하고, 가중치가 너무 커지는 것을 방지하기 위해 거리+0.01의 역수의 로그값을 가중치로 한다.
            # 거리 + 0.01 한 이유는, 0으로 나눗셈을 방지하기 위함.
            weight = math.log10(1 / (self.nearest_dist[i] + 0.01))
            majority_cnt[self.target[self.nearest_k[i]]] += weight

        # 가장 높은 수치를 가진 꽃 class 가 속한 이름을 찾아 저장한다.
        self.weighted_majority_vote_value = self.iris_names[majority_cnt.index(max(majority_cnt))]
        return self.weighted_majority_vote_value

    # 저장되어 있는 모든 데이터를 리셋한다.
    def reset(self):
        self.weighted_majority_vote_value = 0
        self.majority_vote_value = ''
        self.nearest_k = []
        self.nearest_dist = []

    # dimension 을 반환한다.
    def show_dim(self):
        return len(self.features[0])
