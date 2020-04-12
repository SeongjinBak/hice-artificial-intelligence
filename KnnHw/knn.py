class KNN:
    def __init__(self, k, x, y, iris_name):
        self.k = k
        self.features = x
        self.target = y
        self.iris_name = iris_name
        self.weighted_majority_vote = 0
        self.nearest_k = 0

    def calculate_distance(self, x, y):
        return x + y

    # 가장 가까운 k개의 데이터가 어떤 것인지 찾아내는 함수(k는 3, 5, 10..)
    def obtain_knearest_neighbor(self):
        return 'd'

    # 아웃풋은 MV WMV가 있기 때문에, K가 3개이므로 총 6개를 레포트에 붙혀넣기.
    def obtain_majority_vote(self):
        return 'a'

    def obtain_weighted_majority_vote(self):
        return 'k'

    def reset(self):
        return ';'

    def get_nearest_k(self):
        return 'k'
