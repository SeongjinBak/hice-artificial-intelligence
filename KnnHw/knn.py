import math


class KNN:
    def __init__(self, k, x, y, iris_names):
        self.k = k
        self.features = x
        self.target = y
        self.iris_names = iris_names
        self.weighted_majority_vote = 0
        self.nearest_k = []

    # 두 4차원 벡터의 거리를 구한다.
    def distance_metric(self, a, b):
        dist = 0
        i = 0
        while i < 4:
            sub = a[i] - b[i]
            dist += (sub * sub)
            i += 1
        return math.sqrt(dist)

    # 가장 가까운 k개의 데이터가 어떤 것인지 찾아내는 함수(k는 3, 5, 10..)
    def get_nearest_k(self, x_test):
        iter_cnt = 0
        distance_dic = {}
        while iter_cnt < len(self.features):
            distance_dic[self.distance_metric(x_test, self.features[iter_cnt])] = iter_cnt
            iter_cnt += 1
        sdic = sorted(distance_dic.items())
        iter_cnt = 0
        while iter_cnt < self.k:
            self.nearest_k.append(sdic[iter_cnt][1])
            iter_cnt += 1

    # 아웃풋은 MV WMV가 있기 때문에, K가 3개이므로 총 6개를 레포트에 붙혀넣기.
    def obtain_majority_vote(self):
        return 'a'

    def obtain_weighted_majority_vote(self):
        return 'k'

    def reset(self):
        return ';'

