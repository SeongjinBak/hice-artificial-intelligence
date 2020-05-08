"""
B511086 백성진
Logistic Regression 클래스 입니다.
"""
import numpy as np

import functions as func


class LogisticRegression:
    def __init__(self, x, xsize, fsize, y, w, learning_rate):
        self.y = y
        self.x = x
        self.m = xsize
        self.t = fsize
        self.w = w
        self.lr = learning_rate
        self.current_batch = 0

    def cost(self):
        tx = func.softmax(func.sigmoid(np.dot(self.x, self.w)))  # m x t 행렬
        tcost = np.sum((self.y * np.log(tx)) + (1 - self.y) * np.log(1 - tx)) * (-1 / self.m)
        return tcost

    def learn(self, batch_size):
        while self.current_batch < self.m:
            sample = np.random.randint(self.m - batch_size)
            to_batch = self.x[sample:sample + batch_size]
            to_batch_target = self.y[sample:sample + batch_size]
            self.gradient_decent(to_batch, to_batch_target, batch_size)
            self.current_batch += batch_size
        self.current_batch = 0

    def gradient_decent(self, batch_x, batch_y, batch_size):
        tx = func.softmax(np.dot(batch_x, self.w))  # m x t 행렬
        #print(tx)
        # iris 에서는 t가 3, mnist 에서는 t가 10
        for i in range(self.t):
            x = batch_x[:, i]
            # print(x)
            xj = x.reshape(batch_size, 1)
            # print(i, xj)
            # print(self.w[i,:].shape)
            tsum = self.lr * np.sum(tx - batch_y, axis=0) * xj
            # print(tsum)
            for j in range(batch_size):
                ttsum = tsum[j, :]
                self.w[i, :] = self.w[i, :] - ttsum
        print(self.w)
        print()

    def predict(self, t_x, t_y):
        # 예측
        t_x = np.append(t_x, np.array([1.0]))
        print(t_x)
        tx = func.softmax(np.dot(t_x, self.w))  # 1 x t 행렬
        print(tx)

        qi = np.argmax(tx)
        ai = np.argmax(t_y)

        print("예측값 : ", qi, "정답값 : ", ai)
