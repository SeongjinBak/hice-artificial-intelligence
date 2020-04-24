import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.pardir)
# 부모 디렉터리에서 import 할수있도록 설정

from dataset.mnist import load_mnist


# mnist data load 할 수 있는 함수 import


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

image = x_train[0]
label = t_train[0]
print(label)
print(image.shape)

image = image.reshape(28, 28)
print(image.shape)

img_show(image)
