import numpy as np


class NeuralNetworkClass:
    # 생성자 매개변수 : 입력층의 노드 수, 은닉층의 노드 수, 출력층의 노드 수
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # 정규분포 된 랜덤 값을 w 에는 input size x hidden size 행렬로 지정
        # W1: x --> hidden layer
        # W2: x --> output layer
        self.params = {'W1': np.random.randn(input_size, hidden_size), 'b1': np.random.randn(hidden_size),
                       'W2': np.random.randn(hidden_size, output_size), 'b2': np.random.randn(output_size)}

        # 사이즈 저장
        self.input_size = input_size
        self.output_size = output_size

        # 배치한 데이터
        self.batch_x = 0
        self.batch_y = 0

        # 학습률
        self.lr = learning_rate

    def predict(self, x):
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    # x 는 입력 데이터, t 는 테스트 데이터 ???
    def loss(self):
        x, t = self.get_batch_data()
        y = self.predict(x)  # (batch size, 3)
        res = cross_entropy_error(y, t)
        return res

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        t = np.argmax(t, axis=1)
        cnt = 0
        correct = 0
        for i in range(len(y)):
            if y[i] == t[i]:
                correct += 1
            cnt += 1

        return correct / cnt

    def set_batch_data(self, x, t):
        self.batch_x = x
        self.batch_y = t

    def get_batch_data(self):
        return self.batch_x, self.batch_y

    # 기울기 계산용 손실함수
    def loss_func(self, n):
        return self.loss()

    def calculate_gradient(self, f, x):
        h = 1e-4  # 0.0001
        gradients = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

        for idx in range(len(x)):
            tmp_val = x[idx]

            # f(x+h) 계산
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)

            # f(x-h) 계산
            x[idx] = tmp_val - h
            fxh2 = f(x)

            gradients[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val  # 값 복원

        return gradients

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self):
        gradients = {}

        grad = np.zeros_like(self.params['W1'])
        for i in range(len(self.params['W1'])):
            grad[i] = self.calculate_gradient(self.loss_func, self.params['W1'][i])

        gradients['W1'] = grad

        grad = np.zeros_like(self.params['W2'])
        for i in range(len(self.params['W2'])):
            grad[i] = self.calculate_gradient(self.loss_func, self.params['W2'][i])

        gradients['W2'] = grad

        gradients['b1'] = self.calculate_gradient(self.loss_func, self.params['b1'])
        gradients['b2'] = self.calculate_gradient(self.loss_func, self.params['b2'])

        return gradients

    def learn(self):
        x, t = self.get_batch_data()
        gradients = self.numerical_gradient()
        self.params['W1'] -= self.lr * gradients['W1']
        self.params['b1'] -= self.lr * gradients['b1']
        self.params['W2'] -= self.lr * gradients['W2']
        self.params['b2'] -= self.lr * gradients['b2']


# 시그모이드 함수
def sigmoid(z):
    eMin = -np.log(np.finfo(type(0.1)).max)
    zSafe = np.array(np.maximum(z, eMin))
    return 1.0 / (1 + np.exp(-zSafe))


# 소프트맥스 함수
def softmax(x):
    # exp_x = np.exp(x)   # x가 커질수록 값이 너무 커지므로 컴퓨터가 감당 불가 하므로 아래와같이 오버플로우 대비한다
    t = np.max(x, axis=1)
    t = t.reshape(x.shape[0], 1)
    t1 = np.append(t, t, axis=1)
    t = np.append(t1, t, axis=1)

    exp_x = np.exp(x - t)

    sum_exp_x1 = np.sum(exp_x, axis=1)
    sum_exp_x1 = sum_exp_x1.reshape(x.shape[0], 1)
    sum_exp_x2 = np.append(sum_exp_x1, sum_exp_x1, axis=1)
    sum_exp_x = np.append(sum_exp_x2, sum_exp_x1, axis=1)

    y = exp_x / sum_exp_x
    print(y)
    return y


def cross_entropy_error(y, t):
    epsilon = 1e-7

    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]

    cee = np.sum(-t * np.log(y + epsilon)) / batch_size
    return cee
