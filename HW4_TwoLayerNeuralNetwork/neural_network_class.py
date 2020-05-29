# B511086 백성진 HW4


import numpy as np


class NeuralNetworkClass:
    # 생성자 매개변수 : 입력층의 노드 수, 은닉층의 노드 수, 출력층의 노드 수
    def __init__(self, input_size, hidden_size, output_size):
        # 가중치를 정규분포 된 랜덤 값으로 지정
        # W1: x --> hidden layer
        # W2: hidden layer --> output layer
        # b1: W1 과정에 첨가되는 bias
        # b2: W2 과정에 첨가되는 bias
        self.params = {'W1': np.random.randn(input_size, hidden_size), 'b1': np.random.randn(hidden_size),
                       'W2': np.random.randn(hidden_size, output_size), 'b2': np.random.randn(output_size)}

        # 데이터 사이즈
        self.input_size = input_size
        self.output_size = output_size

        # 학습을 위해 미니 배치한 데이터
        self.batch_x = 0
        self.batch_t = 0

        # 훈련용 데이터
        self.x_train = None
        self.y_train = None

        # 테스트용 데이터
        self.x_test = None
        self.y_test = None

        # 결과 기록 리스트
        self.loss_list = []
        self.acc_list = []

    """
    예측함수 입니다. 매개변수로 받는 x로 예측을 원하는 데이터가 들어옵니다.
    본 과제에서는 층이 2개 입니다.
    """

    def predict(self, x):

        # 입력 -> 1층 단계 입니다.
        # 모든 가중치를 입력 데이터와 행렬곱 연산한 후 편향 데이터를 더합니다.
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        # 결과 값을 활성 함수인 시그모이드 함수에 넣습니다.
        z1 = sigmoid(a1)
        # 1층 -> 2층 단계 입니다.
        # 활성 값을 가중치와 행렬곱 연산한 후 편향 데이터를 더합니다.
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        # 결과 값을 소프트맥스 함수로 계산하여 각 입력 데이터의 분류 확률을 얻습니다.
        y = softmax(a2)
        return y

    """
    손실함수 입니다.
    본 과제에서는 미니배치를 사용하기 때문에, 매번 새로운 입력 데이터가 들어옵니다.
    이 입력데이터와 정답 데이터를 매개변수로 넘기지 않고, 클래스의 멤버변수로 지정한 데이터를 가져옵니다.
    """

    def loss(self, x, t):
        # x는 입력데이터, t는 정답데이터 입니다.
        # y 에 입력데이터를 예측한 결과를 저장합니다. 결과는 batch_size, 3의 형상을 가집니다.
        y = self.predict(x)
        # 예측 결과를 교차 엔트로피 함수로 실제 값과 예측값의 오차를 계산합니다.
        res = cross_entropy_error(y, t)
        return res

    """
    정확도 계산 함수 입니다.
    훈련용 데이터, 테스트 데이터의 정확도를 계산하는데 쓰이므로, 범용성을 위해 미니배치 데이터로 예측을 하는것이 아닌
    매개변수로 들어온 값으로 예측을 하도록 하였습니다.
    """

    def accuracy(self, x, t):
        # 예측
        y = self.predict(x)
        # 예측 값의 각 행마다 가장 큰 확률을 가지는 원소의 인덱스를 저장합니다.
        y = np.argmax(y, axis=1)
        # 정답 데이터는 원-핫 인코딩 되어있으므로, 마찬가지로 0이 아닌, 1을 가지는 인덱스를 t에 저장합니다.
        t = np.argmax(t, axis=1)
        # 정확도 계산
        cnt = 0
        correct = 0
        for i in range(len(y)):
            if y[i] == t[i]:
                correct += 1
            cnt += 1

        return correct / cnt

    # 매개변수로 클래스에 배치 데이터 저장하는 함수
    def set_batch_data(self, x, t):
        self.batch_x = x
        self.batch_t = t

    # 클래스의 배치 데이터 반환하는 함수
    def get_batch_data(self):
        return self.batch_x, self.batch_t

    # 전체 훈련용 데이터를 저장합니다.
    def set_train_data(self, x, t):
        self.y_train = t
        self.x_train = x

    # 전체 테스트용 데이터를 저장합니다.
    def set_test_data(self, x, t):
        self.y_test = t
        self.x_test = x



    """
    기울기 계산 함수.
    매개변수 x로 가중치 파라미터를 전달받습니다.
    가중치 파라미터의 각 value 로 수치미분을 진행합니다.
    딕셔너리의 value 는 mutable 하므로, 함수 내부에서 변형시키면 원본 딕셔너리에 영향이 갑니다.
    따라서, value 를 수정한 후, loss 진행을 하며 그 후 원래의 값으로 되돌립니다.
    x는 가중치 파라미터이며, ix는 배치 데이터, t는 배치 타겟 데이터 입니다.
    """

    def calculate_gradient(self, x, ix, t):
        # 아주 작은 h 값.
        h = 1e-4

        # x와 형상이 같은 배열을 생성합니다.
        gradients = np.zeros_like(x)

        # 넘어온 가중치의 각 value 당 루프를 돌며, 수치미분을 합니다.
        for idx in range(len(x)):
            # 넘어온 가중치의 value 저장.
            tmp_val = x[idx]

            # f(x+h) 계산
            x[idx] = float(tmp_val) + h

            # x + h 가중치로 손실 측정
            fxh1 = self.loss(ix, t)

            # f(x-h) 계산
            x[idx] = tmp_val - h

            # x - h 가중치로 손실 측정
            fxh2 = self.loss(ix, t)

            # 기울기 계산
            gradients[idx] = (fxh1 - fxh2) / (2 * h)

            # 값을 원래대로 되돌립니다.
            x[idx] = tmp_val

        return gradients

    """
    기울기 계산의 caller 함수.
    본 과제의 모든 가중치에 대해 기울기를 최신화 합니다.
    bias 값은 1차원 이지만, W 값은 2차원 행렬 이므로 각 행의 모든 열에 대해 기울기 계산을 해줍니다.
    """

    def numerical_gradient(self, x, t):
        # 저장할 가중치 딕셔너리
        gradients = {}

        # W1 에 대해 가중치 계산 진행. input x hidden layer
        grad = np.zeros_like(self.params['W1'])
        # input 행 당 모든 열에 대해 진행
        for i in range(len(self.params['W1'])):
            grad[i] = self.calculate_gradient(self.params['W1'][i], x, t)

        gradients['W1'] = grad

        # W2 에 대해 가중치 계산 진행. hidden layer x output
        grad = np.zeros_like(self.params['W2'])
        # hidden layer 행 당 모든 열에 대해 진행
        for i in range(len(self.params['W2'])):
            grad[i] = self.calculate_gradient(self.params['W2'][i], x, t)

        gradients['W2'] = grad

        # bias 에 대해 진행. b1 은 hidden layer, b2 는 output 의 개수.
        gradients['b1'] = self.calculate_gradient(self.params['b1'], x, t)
        gradients['b2'] = self.calculate_gradient(self.params['b2'], x, t)

        return gradients

    """
    학습 함수 입니다.
    배치 데이터를 받아와, 그 데이터에 대해 가중치의 기울기 값을 구한 후, 가중치를 최신화 합니다.
    총 epoch 번 반복학습하며, 각 epoch 당 iter_num 만큼 미니 배치한 데이터로 학습합니다.
    데이터 1개만으로 학습하는 경우와 모든 데이터로 한꺼번에 학습하는 경우 각각이 가지는 장단점을 타협하기 위해
    미니배치 학습을 합니다.
    """

    def learn(self, lr, epoch, batch_size):

        # 1번의 epoch 를 수행하기 위한 반복 횟수
        iter_num = int(120 / batch_size)

        # 매개변수로 들어온 epoch 횟수만큼 반복합니다.
        for i in range(epoch):

            tmpLoss = 0.0

            # 120개의 데이터를 batch size 만큼 나누어 진행합니다.
            for j in range(iter_num):
                # 미니배치 획득 ( 120개 중 batch size 개 만큼) 합니다.
                batch_sample = np.random.choice(120, batch_size)

                # 얻어낸 샘플 데이터의 인덱스를 batch 데이터로 할당합니다.
                x_batch = self.x_train[batch_sample]
                y_batch = self.y_train[batch_sample]

                self.set_batch_data(x_batch, y_batch)

                # 각 가중치의 예측 후의 손실 값 얻음.
                gradients = self.numerical_gradient(x_batch, y_batch)

                # 최신화 된 값을 학습률에 곱하여, 원래의 가중치에서 뺌.
                self.params['W1'] -= lr * gradients['W1']
                self.params['b1'] -= lr * gradients['b1']
                self.params['W2'] -= lr * gradients['W2']
                self.params['b2'] -= lr * gradients['b2']

                # 손실값 기록
                tmpLoss += self.loss(x_batch, y_batch)

            # 훈련 데이터의 손실 평균과, 정확도 측정
            train_loss = tmpLoss / iter_num
            train_acc = self.accuracy(self.x_train, self.y_train)

            print(i, 'cost, accuracy', train_loss, train_acc)

            # 손실값 평균 저장
            self.loss_list.append(train_loss)
            # 정확도 저장
            self.acc_list.append(train_acc)

            # 최종 정확도를 출력합니다.
            if i + 1 == epoch:
                print('Training Accuracy =', train_acc)
                print('Test Accuracy =', self.accuracy(self.x_test, self.y_test))


# 시그모이드 함수
def sigmoid(z):
    eMin = -np.log(np.finfo(type(0.1)).max)
    zSafe = np.array(np.maximum(z, eMin))
    return 1.0 / (1 + np.exp(-zSafe))


# 소프트맥스 함수
def softmax(x):
    # 뺄셈 계산을 위해 (x.shape[0], )를 (x.shape[0], 3)으로 변환합니다.
    # 각 행당 가장 큰 값을 얻음
    t = np.max(x, axis=1)
    # x와의 형상 통일을 위한 append.
    t = t.reshape(x.shape[0], 1)
    t1 = np.append(t, t, axis=1)
    t = np.append(t1, t, axis=1)

    # 지수 함수의 특성으로 인한 아주 큰 값을 방지하기 위해, 가장 큰 값을, 각 행의 전체 원소에 뺄셈 합니다.
    exp_x = np.exp(x - t)

    # 계산을 위해 (x.shape[0], )를 (x.shape[0], 3)으로 변환합니다.
    # x와의 형상 통일을 위한 append.
    sum_exp_x1 = np.sum(exp_x, axis=1)
    sum_exp_x1 = sum_exp_x1.reshape(x.shape[0], 1)
    sum_exp_x2 = np.append(sum_exp_x1, sum_exp_x1, axis=1)
    sum_exp_x = np.append(sum_exp_x2, sum_exp_x1, axis=1)

    y = exp_x / sum_exp_x
    return y


# 교차 엔트로피 오류 함수.
# 매개변수로 들어온 y 는 예측값, t는 원-핫 인코딩된 정답 값 입니다.
def cross_entropy_error(y, t):
    epsilon = 1e-7

    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]

    # 들어온 데이터의 오차 값을 계산합니다.
    # 값이 클 수록, 오답일 가능성이 높다는 것을 나타냅니다.
    cee = np.sum(-t * np.log(y + epsilon)) / batch_size
    return cee
