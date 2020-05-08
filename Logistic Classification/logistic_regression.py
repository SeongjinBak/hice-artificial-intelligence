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
        self.t = fsize  # iris 5
        self.w = w
        self.lr = learning_rate
        self.current_batch = 0
        self.ailist = [0, 0, 0]
        self.aicount = [0, 0, 0]

    def cost(self, batch_size):
        epoch_cost = 0
        local_epoch = 0
        while self.current_batch < self.m:
            local_epoch += 1
            sample = np.random.randint(self.m - batch_size)
            to_batch = self.x[sample:sample + batch_size]
            to_batch_target = self.y[sample:sample + batch_size]

            tx = func.softmax(np.dot(to_batch, self.w))  # m x t 행렬
            #print(to_batch_target.shape)
            tcost = np.sum((to_batch_target * np.log(tx)) + (1 - to_batch_target) * np.log(1 - tx), axis=0) * (-1 / batch_size)
            self.current_batch += batch_size
            epoch_cost += tcost
            tcost = 0

        self.current_batch = 0
        return epoch_cost / local_epoch

    def learn(self, batch_size):
        while self.current_batch <= self.m:
            sample = np.random.randint(self.m - batch_size)
            to_batch = self.x[sample:sample + batch_size]
            to_batch_target = self.y[sample:sample + batch_size]
            self.gradient_decent(to_batch, to_batch_target, batch_size)
            self.current_batch += batch_size
        self.current_batch = 0

    def gradient_decent(self, batch_x, batch_y, batch_size):
        tx = func.softmax(np.dot(batch_x, self.w))  # m x t 행렬
        # print(tx)
        # iris 에서는 t가 3, mnist 에서는 t가 10
        for i in range(self.t):
            # x의 i번째 행 을 다 가져온다.
            x = batch_x[:, i]
            xj = x.reshape(batch_size, 1)
            tsum = self.lr * np.sum((tx - batch_y) * xj, axis=0)
            self.w[i, :] = self.w[i, :] - tsum
        # print(self.w)

    def predict(self, t_x, t_y):
        # 예측
        t_x = np.append(t_x, np.array([1.0]))
        print(t_x)
        tx = func.softmax(np.dot(t_x, self.w))  # 1 x t 행렬
        print(tx)

        qi = np.argmax(tx)
        ai = np.argmax(t_y)

        self.aicount[ai] += 1
        if ai == qi:
            self.ailist[qi] += 1

        print("예측값 : ", qi, "정답값 : ", ai)
