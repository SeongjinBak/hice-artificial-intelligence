# B511086 백성진 knn mnist
import sys
import os
# 소요시간 측정을 할 수 있도록 설정
import timeit

import numpy as np
from PIL import Image
from knn import KNN

# 부모 디렉터리에서 import 할 수 있도록 설정
sys.path.append(os.pardir)

# mnist data load
from dataset.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# 데이터를 hand-craft 하는 함수. 4또는 7로 평균 된 데이터 리스트가 반환된다.
# 예를들어, 4로 평균을 낸 경우 0번째행의 28개 원소는 4 * 7 개 원소 이므로 4개원소의 평균값이 새로운 리스트의 7개원소 중 1개가 된다.
def hand_craft_data(li):
    craftedlist_4 = []
    craftedlist_7 = []

    for i in range(len(li)):
        tmpli = li[i]
        crafted_4 = []
        crafted_7 = []
        sum_4 = 0
        sum_7 = 0
        cnt_4 = 0
        cnt_7 = 0

        for it in range(len(tmpli)):
            cnt_4 += 1
            cnt_7 += 1
            # 4개 픽셀의 평균
            if cnt_4 % 4 == 0:
                crafted_4.append(sum_4 / 4)
                sum_4 = 0
            # 7개 픽셀의 평균
            if cnt_7 % 7 == 0:
                crafted_7.append(sum_7 / 7)
                sum_7 = 0
            sum_4 += tmpli[it]
            sum_7 += tmpli[it]

        craftedlist_4.append(crafted_4)
        craftedlist_7.append(crafted_7)

    return craftedlist_4, craftedlist_7


def get_target_label(li):
    rdtarget = []
    for i in range(len(li)):
        rdtarget.append(li)


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

sample_train_image = x_train[0:60000]
sample_train_label = t_train[0:60000]

# test data 개수
size = 500

# 0~10000 사이의 랜덤한 size 개 만큼의 데이터 인덱스 선정
sample = np.random.randint(0, t_test.shape[0], size)
sample_test_image = []
sample_test_label = []

# 랜덤하게 선정된 데이터 인덱스의 실제 데이터를 리스트에 저장
for i in sample:
    sample_test_image.append(x_test[i])
    sample_test_label.append(t_test[i])

sample_train_image_4, sample_train_image_7 = hand_craft_data(sample_train_image)

sample_test_image_4, sample_test_image_7 = hand_craft_data(sample_test_image)

# knn 알고리즘의 k는 10으로 설정
k_num = 10
knn_mnist = KNN(k_num, sample_train_image, sample_train_label, label_name)
knn_mnist_4 = KNN(k_num, sample_train_image_4, sample_train_label, label_name)
knn_mnist_7 = KNN(k_num, sample_train_image_7, sample_train_label, label_name)

# 아래 옵션이 true인 경우에만 7로 평균 낸 데이터도 분류한다. 4로 평균 낸 데이터가 기본이다.
option_for_7 = True

# 아래 옵션은 데이터 feature 가공을 거치지 않은 mnist 데이터 셋을 사용하는 옵션이다.
option_for_using_original = True

# 결과 출력용 리스트
accs = [0.0, 0.0, 0.0]
corrects = [0, 0, 0]
misses = [0, 0, 0]
times = [0.0, 0.0, 0.0]

acc_4 = 0
correct_4 = 0
miss_4 = 0

# 4개로 압축한 것과 7개로 압축한 것의 정확도 비교. option 이 true 인 경우에만 7개 압축한 것도 knn 분류 한다.
start_time = timeit.default_timer()
for i in range(size):
    knn_mnist_4.get_nearest_k(sample_test_image_4[i])
    vote = knn_mnist_4.weighted_majority_vote()
    print("4-averaged Test Data: ", i, " Computed class: ", vote, ", \tTrue class: ", sample_test_label[i])

    if vote == str(sample_test_label[i]):
        acc_4 += 1
        correct_4 += 1
    else:
        miss_4 += 1
    knn_mnist_4.reset()
stop_time = timeit.default_timer()
acc_4 /= size

accs[0] = acc_4
corrects[0] = correct_4
misses[0] = miss_4
times[0] = stop_time - start_time

# 7개 분류 옵션이 켜진 경우
if option_for_7 is True:
    acc_7 = 0
    correct_7 = 0
    miss_7 = 0

    start_time = timeit.default_timer()
    for i in range(size):
        knn_mnist_7.get_nearest_k(sample_test_image_7[i])
        vote = knn_mnist_7.weighted_majority_vote()
        print("7-averaged Test Data: ", i, " Computed class: ", vote, ", \tTrue class: ", sample_test_label[i])

        if vote == str(sample_test_label[i]):
            acc_7 += 1
            correct_7 += 1
        else:
            miss_7 += 1
        knn_mnist_7.reset()

    stop_time = timeit.default_timer()
    acc_7 /= size

    accs[1] = acc_7
    corrects[1] = correct_7
    misses[1] = miss_7
    times[1] = stop_time - start_time

# feature 가공을 거치지 않은 데이터를 입력 feature 로 사용하는 경우
if option_for_using_original is True:
    acc = 0
    correct = 0
    miss = 0
    start_time = timeit.default_timer()

    for i in range(size):
        knn_mnist.get_nearest_k(sample_test_image[i])
        vote = knn_mnist.weighted_majority_vote()
        print("None-averaged Test Data: ", i, " Computed class: ", vote, ", \tTrue class: ", sample_test_label[i])

        if vote == str(sample_test_label[i]):
            acc += 1
            correct += 1
        else:
            miss += 1
        knn_mnist.reset()

    stop_time = timeit.default_timer()
    acc /= size

    accs[2] = acc
    corrects[2] = correct
    misses[2] = miss
    times[2] = stop_time - start_time

# 테스트 결과 출력
print("\nTest size :", size)
if option_for_using_original is True:
    print("None-crafted", "accuracy :", accs[2], "\tcorrect :", corrects[2], "\tmiss :", misses[2],
          "\telapsed time(sec) :", times[2])
print("4-averaged crafted", "accuracy :", accs[0], "\tcorrect :", corrects[0], "\tmiss :", misses[0],
      "\telapsed time(sec) :", times[0])
if option_for_7 is True:
    print("7-averaged crafted", "accuracy :", accs[1], "\tcorrect :", corrects[1], "\tmiss :", misses[1],
          "\telapsed time(sec) :", times[1])
