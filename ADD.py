"""Additive algorithm for Watermarking"""

import numpy as np
from util.watermarking import Watermarking
from util.test import test_random
from util.test import encrypt_img
from util.test import eval_metrics

class ADD(Watermarking):
    """Abstract class for watermarking for ADD algorithm."""
    def __init__(self):
        pass

    def enc(self, _i, _w, _k):
        """Encrypts an image _i with a watermark _w and a key _k."""

        alpha = _k
        i_res = _i + alpha*_w
        i_res = np.clp(i_res,0,255)
        return i_res

    def dec(self, _d, _i, _k):
        """Decrypts a watermark from an image _i with a key _k."""

        alpha = _k
        i_res = _i - _d*alpha
        i_res = np.clip(i_res, 0, 255)
        return i_res

T = ADD()
W = b"1100"
_d = (T.enc(np.eye(10,10,3).astype(np.uint8),W,10))
print(eval_metrics(T, "../test_images/img_1.jpeg", b"110000"))
encrypt_img(T, "../test_images/img_1.jpeg", b"1111101010010101010010101100100101000101010101001010")
print(test_random(T))
