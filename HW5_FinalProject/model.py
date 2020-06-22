# coding: utf-8
# 2020/인공지능/final/B511086/백성진
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


"""
활성화 계층으로, ReLU 함수를 사용했습니다.
순전파시 입력으로 들어온 값들 중 0 이하인 계층은 0으로, 그 이상은 해당 값으로 지정합니다.
역전파시에는, 입력으로 0이 들어왔던 값은 그대로 0을, 나머지는 다음 계층에서 들어온 미분값을 전달합니다.
"""


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # 0 이하의 값을 가지는 인덱스 체크
        self.mask = (x <= 0)
        # out 전달 값 입력
        out = x.copy()
        # 인덱스가 체크 되어있는 마스크의 인덱스는 0 처리
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 입력이 0 이었던 것은 역전파시 0을 그대로 전달.
        dout[self.mask] = 0
        dx = dout
        return dx


class CustomActivation:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


"""
행렬의 원소 곱을 수행하는 Affine 층 클래스 입니다.
기존에 단일 값, 혹은 단일 차원 배열로 넘겨주던 입력값을 행렬로 한번에 처리하는 구조 입니다.
"""


class Affine:
    def __init__(self, W, b):
        # 가중치와 편향 값을 초기화 합니다.
        self.W = W
        self.b = b

        self.x = None

        self.dW = None
        self.db = None

    # 순전파시 입력값과 가중치를 행렬곱 한 후, 편향을 더하여 반환합니다.
    # 이 때, 입력으로 들어온 행렬은 따로 저장합니다.
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    # 역전파시, 매개변수로 들어온 다음 클래스의 미분값을, 가중치의 전치행렬과 행렬곱 합니다.
    # 그 후, 미분된 가중치는 순전파시 저장한 입력 행렬의 전치행렬과 미분값을 행렬곱 한 후 저장합니다.
    # 마찬가지로, 미분된 편향값은 행렬의 계수를 이전 층의 행렬과 맞추기 위해 미분값의 행들을 한꺼번에 더한 후 저장합니다.
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


"""
소프트맥스와 교차 엔트로피 오류를 통합한 계층 입니다.
마지막 계층에서 그 역할을 수행합니다.
두개의 기능을 합친 이유는, 미분의 역전파시 계산을 쉽게 수행하기 위함 입니다.
또한, 이번 과제에서 결과는 6가지의 동작을 '분류'하는 것이기에 소프트 맥스 함수를 사용합니다.
"""


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    # 예측값을 이전 계층인 Affine 계층에서 받아온 값을 소프트맥스로 정규화합니다.
    # 그 후 정답 레이블과 예측된 레이블을 교차엔트로피 에러 손실함수를 이용하여 손실율을 구합니다.
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    # 역전파시, 정답 레이블이 원-핫 인코딩 형태라면 예측 레이블에서 타겟 레이블을 빼준 것을,
    # 원-핫 인코딩이 아니라면 해당하는 인덱스에만 1을 표기하고 1을 빼준 값을 반환합니다.
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 정답 레이블이 원-핫 인코딩 형태일 때
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    # 가중치를 최신화 합니다.
    def update(self, params, grads):
        params['W1'] -= self.lr * grads['W1']
        params['b1'] -= self.lr * grads['b1']
        params['W2'] -= self.lr * grads['W2']
        params['b2'] -= self.lr * grads['b2']
        params['W3'] -= self.lr * grads['W3']
        params['b3'] -= self.lr * grads['b3']
        params['W4'] -= self.lr * grads['W4']
        params['b4'] -= self.lr * grads['b4']
        params['W5'] -= self.lr * grads['W5']
        params['b5'] -= self.lr * grads['b5']
        params['W6'] -= self.lr * grads['W6']
        params['b6'] -= self.lr * grads['b6']


class CustomOptimizer:
    pass


class Model:
    """
    네트워크 모델 입니다.

    """

    def __init__(self, lr=0.01):
        """
        클래스 초기화
        """

        self.params = {}
        self.__init_weight()
        self.__init_layer()
        self.optimizer = SGD(lr)

    def __init_layer(self):
        """
        레이어를 생성하시면 됩니다.
        """
        # 계층 생성, 6층 레이어를 생성하였습니다. 
        # Affine 계층 다음, 활성 계층으로 ReLU 함수를 사용하였고 마지막 계층은 소프트맥스와 교차 엔트로피 오류 계층으로 설정합니다.
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.layers['Relu5'] = Relu()
        self.layers['Affine6'] = Affine(self.params['W6'], self.params['b6'])
        self.lastLayer = SoftmaxWithLoss()

    def __init_weight(self, ):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        # 가중치 초기화
        self.params = {}

        # ReLU를 활성함수로 사용하므로, 전파시 활성값 사라짐을 막기 위해 파라미터의 초기값으로 he 초기값을 사용합니다.
        self.params['W1'] = np.random.randn(6, 8) / (6 / 2) ** .5
        self.params['b1'] = np.zeros(8)
        self.params['W2'] = np.random.randn(8, 10) / (8 / 2) ** .5
        self.params['b2'] = np.zeros(10)
        self.params['W3'] = np.random.randn(10, 10) / (10 / 2) ** .5
        self.params['b3'] = np.zeros(10)
        self.params['W4'] = np.random.randn(10, 10) / (10 / 2) ** .5
        self.params['b4'] = np.zeros(10)
        self.params['W5'] = np.random.randn(10, 8) / (10 / 2) ** .5
        self.params['b5'] = np.zeros(8)
        self.params['W6'] = np.random.randn(8, 6) / (8 / 2) ** .5
        self.params['b6'] = np.zeros(6)

    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        # 예측을 위해 순전파 합니다.
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward
        # 데이터와 정답 레이블의 손실을 구합니다.
        self.loss(x, t)

        # backward
        # 마지막 계층의 미분값을 저장합니다.
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        # 각 층에 대한 정보를 받아옵니다.
        layers = list(self.layers.values())
        # 역전파를 위해 받아온 리스트를 역순으로 뒤집습니다.
        layers.reverse()
        # 뒤집은 리스트를 순서대로 backward 함수를 수행하여 미분합니다.
        # 반복문 내부에서 다음 계층의 미분값을 이번 계층에 넘깁니다.
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        # 반복문으로 구한 각 계층의 가중치와 편향의 gradient 를 딕셔너리에 저장합니다.
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db
        grads['W5'], grads['b5'] = self.layers['Affine5'].dW, self.layers['Affine5'].db
        grads['W6'], grads['b6'] = self.layers['Affine6'].dW, self.layers['Affine6'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        # 로드한 파라미터를 실제 네트워크 모델의 파라미터로 갱신합니다.
        self.layers['Affine1'] = Affine(params['W1'], params['b1'])
        self.layers['Affine2'] = Affine(params['W2'], params['b2'])
        self.layers['Affine3'] = Affine(params['W3'], params['b3'])
        self.layers['Affine4'] = Affine(params['W4'], params['b4'])
        self.layers['Affine5'] = Affine(params['W5'], params['b5'])
        self.layers['Affine6'] = Affine(params['W6'], params['b6'])
