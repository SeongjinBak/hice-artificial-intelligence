"""
B511086 백성진
Logistic Regression 클래스 입니다.
"""
import numpy as np
import random
import functions as func


class LogisticRegression:
    def __init__(self, x, xsize, fsize, y, w, learning_rate, issingle=False):
        self.y = y
        self.x = x
        self.m = xsize
        self.t = fsize  # iris 5 mnist 784+1
        self.w = w
        self.lr = learning_rate
        self.current_batch = 0
        self.ailist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.aicount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.issingle = issingle

    def cost(self, batch_size):
        epoch_cost = 0

        if self.m == batch_size:
            sample = 0
        else:
            # sample = np.random.randint(self.m - batch_size)
            sample = np.array(np.random.randint(0, self.m, batch_size))  # random.sample(self.m, batch_size))
        # print(sample)
        # to_batch = self.x[sample:sample + batch_size]
        # to_batch_target = self.y[sample:sample + batch_size]
        to_batch = np.array([self.x[i] for i in sample])
        to_batch_target = np.array([self.y[i] for i in sample])
        # print(to_batch_target)
        tx = func.sigmoid(np.dot(to_batch, self.w))  # m x t 행렬
        epsilon = 1e-7
        if not self.issingle:
            tcost = np.sum((to_batch_target * np.log(tx + epsilon)) + (1 - to_batch_target) * np.log(1 - tx + epsilon),
                           axis=0) * (-1 / batch_size)
        else:
            tcost = np.sum(
                (to_batch_target * np.log(tx + epsilon)) + (1 - to_batch_target) * np.log(1 - tx + epsilon)) * (
                            -1 / batch_size)
            # print(1-to_batch_target)
        epoch_cost += tcost
        return epoch_cost

    def learn(self, batch_size):
        if self.m == batch_size:
            sample = 0
        else:
            # sample = np.random.randint(self.m - batch_size)
            sample = np.array(np.random.randint(0, self.m, batch_size))  # random.sample(self.m, batch_size))

        # to_batch = self.x[sample:sample + batch_size]
        # to_batch_target = self.y[sample:sample + batch_size]
        to_batch = np.array([self.x[i] for i in sample])
        to_batch_target = np.array([self.y[i] for i in sample])
        self.gradient_decent(to_batch, to_batch_target, batch_size)
        self.current_batch = 0

    def gradient_decent(self, batch_x, batch_y, batch_size):
        tx = func.sigmoid(np.dot(batch_x, self.w))  # m x t 행렬
        # print(tx)
        # iris 에서는 t가 3, mnist 에서는 t가 10
        for i in range(self.t):
            # x의 i번째 열 을 다 가져온다.
            x = batch_x[:, i]
            xj = x.reshape(batch_size, 1)
            # multi class weights
            if not self.issingle:
                tsum = self.lr * np.sum((tx - batch_y) * xj, axis=0)
                self.w[i, :] = self.w[i, :] - tsum
            # single class weights
            else:
                tsum = self.lr * np.sum((tx - batch_y) * xj)
                self.w[i] = self.w[i] - tsum
                # print(tsum)

    def predict(self, x, y, sz, to_find_target = 0):
        single_cnt = 0
        single_try = 0
        for i in range(sz):
            t_x = x[i]
            t_y = y[i]
            # 예측
            t_x = np.append(t_x, np.array([1.0]))
            tx = func.sigmoid(np.dot(t_x, self.w))  # 1 x t 행렬
            # print(tx)

            qi = np.argmax(tx)
            ai = np.argmax(t_y)

            self.aicount[ai] += 1
            if ai == qi:
                self.ailist[qi] += 1

            if self.issingle:
                if ai == to_find_target:
                    single_try += 1
                    if tx > 0.5:
                        single_cnt += 1

            # print("예측값 : ", qi, "정답값 : ", ai)
        if not self.issingle:
            for i in range(10):
                if self.aicount[i] != 0:
                    print(i, self.ailist[i] / self.aicount[i])
        else:
            print("class : ", to_find_target, "acc : ", single_cnt / single_try)
