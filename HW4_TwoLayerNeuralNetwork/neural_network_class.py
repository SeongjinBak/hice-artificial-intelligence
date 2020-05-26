import numpy as np


class NeuralNetworkClass:
    # 생성자 매개변수 : 입력층의 노드 수, 은닉층의 노드 수, 출력층의 노드 수
    def __init__(self, input_size, hidden_size, output_size):
        # 정규분포 된 랜덤 값을 w 에는 input size x hidden size 행렬로 지정
        # W1: x --> hidden layer
        # W2: x --> output layer
        self.params = {'W1': np.random.randn(input_size, hidden_size), 'b1': np.random.randn(hidden_size),
                       'W2': np.random.randn(hidden_size, output_size), 'b2': np.random.randn(output_size)}

        # 사이즈 저장
        self.input_size = input_size
        self.output_size = output_size

    def predict(self, x):
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        # print(y)
        return y

    # x 는 입력 데이터, t 는 테스트 데이터 ???
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)

        y = np.argmax(y, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def f(self):
        x, t = 1
        f = self.loss(x, t)
        return f(self.params['W1'])

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_w = lambda n: self.loss(x, t)

        gradients = {'W1': numerical_gradient(loss_w, self.params['W1']),
                     'b1': numerical_gradient(loss_w, self.params['b1']),
                     'W2': numerical_gradient(loss_w, self.params['W2']),
                     'b2': numerical_gradient(loss_w, self.params['b2'])}

        return gradients


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    gradients = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
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


def numerical_gradient(f, t_x):
    if t_x.ndim == 1:
        return _numerical_gradient_no_batch(f, t_x)
    else:
        gradients = np.zeros_like(t_x)

        for idx in range(len(t_x)):
            gradients[idx] = _numerical_gradient_no_batch(f, t_x[idx])

        return gradients


# 시그모이드 함수
def sigmoid(z):
    eMin = -np.log(np.finfo(type(0.1)).max)
    zSafe = np.array(np.maximum(z, eMin))
    return 1.0 / (1 + np.exp(-zSafe))


# 소프트맥스 함수
def softmax(x):
    # x가 커질수록 값이 너무 커지므로 컴퓨터가 감당 불가 하므로 아래와같이 오버플로우 대비한다
    # x의 행 별로 계산한다.
    exp_x = np.exp(x - np.max(x, axis=0))
    sum_exp_x = np.sum(exp_x, axis=0)
    y = exp_x / sum_exp_x
    return y


def cross_entropy_error(y, t):
    epsilon = 1e-7
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
