import sys
sys.path.append('../')

from util.watermarking import Watermarking
from util.test import test_random
from util.test import encrypt_img
from util.test import eval_metrics
import numpy as np

class Testing(Watermarking):
    def __init__(self):
        pass

    def Enc(self, I, W, K):
        return I+(W*K)

    def Dec(self, D, I, K):
        return (D-I)/K

T = Testing()
assert(test_random(T))
encrypt_img(T, "../../test_images/img_1.jpeg")
W = np.random.rand(10, 10, 3)*255
print(eval_metrics(T, "../../test_images/img_1.jpeg", W))

