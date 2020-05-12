"""
B511086 백성진
Logistic Regression 클래스 입니다.
"""
import numpy as np
import random
import functions as func


class LogisticRegression:
    # 생성자에서는 데이터 x와 타겟데이터 y, 그리고 각 데이터의 크기, 정답 리스트 그리고 Single classification 여부
    def __init__(self, x, xsize, fsize, y, w, learning_rate, issingle=False):
        self.y = y
        self.x = x
        self.m = xsize
        self.t = fsize  # iris 4 + 1, mnist 784+1
        self.w = w
        self.lr = learning_rate
        self.ailist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.aicount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.issingle = issingle

    # 코스트 함수. batch_size 개 만큼의 계산을 한꺼번에 한다.
    def cost(self, batch_size):
        # 0 ~ m 중 랜덤으로 batch_size 개를 뽑는다.
        sample = np.array(np.random.randint(0, self.m, batch_size))
        # 뽑은것으로 x 데이터와 y 데이터를 뽑는다.
        to_batch = np.array([self.x[i] for i in sample])
        to_batch_target = np.array([self.y[i] for i in sample])

        # Hypothesis
        tx = func.sigmoid(np.dot(to_batch, self.w))  # m x t 행렬
        
        # 로그 내 0 을 방지하기 위한 작은 수
        epsilon = 1e-7

        # Multiple class lr 인 경우, 행별 코스트 계산을 실시한다.
        if not self.issingle:
            tcost = np.sum((to_batch_target * np.log(tx + epsilon)) + (1 - to_batch_target) * np.log(1 - tx + epsilon),
                           axis=0) * (-1 / batch_size)

        # Single class lr 인 경우, 한꺼번에 계산을 실시한다.
        else:
            tcost = np.sum(
                (to_batch_target * np.log(tx + epsilon)) + (1 - to_batch_target) * np.log(1 - tx + epsilon)) * (
                            -1 / batch_size)
        return tcost

    # 학습함수. batch_size 만큼의 데이터를 한꺼번에 학습한다.
    def learn(self, batch_size):
        # 0 ~ m 개의 데이터 중 batch_size 만큼의 데이터를 랜덤으로 선출 후 계산용 배열에 추가한다.
        sample = np.array(np.random.randint(0, self.m, batch_size))
        to_batch = np.array([self.x[i] for i in sample])
        to_batch_target = np.array([self.y[i] for i in sample])

        # Gradient decent 계산하여 weight 최신화 한다.
        self.gradient_decent(to_batch, to_batch_target, batch_size)

    # 경사 하강법 함수
    def gradient_decent(self, batch_x, batch_y, batch_size):
        tx = func.sigmoid(np.dot(batch_x, self.w))  # m x t 행렬

        # 피쳐 개수 만큼 반복문을 돌며 theta(weight) 최신화 한다.
        for i in range(self.t):     # iris 에서는 t가 3, mnist 에서는 t가 10
            # x의 i번째 열 을 다 가져온다.
            x = batch_x[:, i]
            # m * 1 사이즈의 배열로 바꾼다.
            xj = x.reshape(batch_size, 1)

            # multiple class weights 최신화
            if not self.issingle:
                tsum = self.lr * np.sum((tx - batch_y) * xj, axis=0)
                self.w[i, :] = self.w[i, :] - tsum
                
            # single class weights 최신화
            else:
                tsum = self.lr * np.sum((tx - batch_y) * xj)
                self.w[i] = self.w[i] - tsum

    # 예측함수. Single class lr일 경우 찾을 타겟의 번호를 매개변수로 받는다.
    def predict(self, x, y, sz, to_find_target = 0):
        single_cnt = 0
        single_try = 0

        # 테스트 케이스의 사이즈만큼 루프를 돈다.
        for i in range(sz):
            single_try += 1
            
            t_x = x[i]
            t_y = y[i]

            # 계산용 bias 추가.
            t_x = np.append(t_x, np.array([1.0]))

            # Hypothesis
            tx = func.sigmoid(np.dot(t_x, self.w))  # 1 x t 행렬

            # 계산된 h(x) 중 가장 큰 인덱스를 가지는 것과, 타겟의 '클래스 번호'를 얻는다.
            qi = np.argmax(tx)
            ai = np.argmax(t_y)
            
            # 각각의 클래스의 정확도 판별용 코드
            self.aicount[ai] += 1
    
            # 추정된 클래스와 진짜 클래스가 같은 경우(예측성공)    
            if ai == qi:
                self.ailist[qi] += 1
                single_cnt += 1

            # Single class lr의 경우
            if self.issingle:
                # 모든 데이터 중, 분류하려는 클래스와 정답 클래스가 일치하는 경우
                if ai == to_find_target:
                    # 0.5 이상인 경우 예측성공 처리
                    if tx > 0.5:
                        single_cnt += 1

        # Multiple class lr
        if not self.issingle:
            """
            각 클래스별 정확도
            for i in range(10):
                if self.aicount[i] != 0:
                    print(i, self.ailist[i] / self.aicount[i])
            """
            print("정확도 : ", single_cnt / single_try, "데이터 총 수 ", single_try, "맞은 데이터 수", single_cnt)

        # Single class lr
        else:
            print("분류하고자 한클래스 : ", to_find_target, "정확도 : ", single_cnt / single_try, "데이터 총 수 ", single_try,
                  "맞은 데이터 수", single_cnt)
