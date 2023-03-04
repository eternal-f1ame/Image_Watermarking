"""DWT watermarking algorithm."""

import itertools
import numpy as np
import cv2
import pywt
from PIL import Image
from util.watermarking import Watermarking
from util.test import test_random
from util.test import encrypt_img
from util.test import eval_metrics
from util.test import eval_metrics_cv



quant = np.array([[16,11,10,16,24,40,51,61],      
                    [12,12,14,19,26,58,60,55],    
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])

class DWT(Watermarking):
    """Class for watermarking using DWT algorithm."""

    def __init__(self):
        self.message = None
        self.bit_mess = None
        self.ori_col = 0
        self.ori_row = 0
        self.num_bits = 0

    def enc(self, img, secret_msg, _k):
        """Encrypts an image _i with a watermark _w and a key _k."""
        # Convert the secret message to binary
        self.message = secret_msg
        self.bit_mess = self.to_bits()

        self.ori_row, self.ori_col = img.shape[:2]

        self.num_bits = len(self.bit_mess)
        if self.num_bits > self.ori_row * self.ori_col:
            raise ValueError("Message is too long to be encoded in the image.")

        coeffs = pywt.dwt2(img, "haar")

        _ll, (_lh, _hl, _hh) = coeffs

        ll_q = np.round(_ll / _k)
        ll_q = np.multiply(ll_q, quant)

        index = 0
        for row, col in itertools.product(range(self.ori_row), range(self.ori_col)):
            if index == self.num_bits:
                break
            bit = int(self.bit_mess[index])
            if bit == 0 and ll_q[row][col] % 2 == 1:
                ll_q[row][col] -= 1
            elif bit == 1 and ll_q[row][col] % 2 == 0:
                ll_q[row][col] += 1
            index += 1

        ll_q = np.divide(ll_q, quant)
        ll_q = np.round(ll_q * _k)

        coeffs_q = (ll_q, (_lh, _hl, _hh))
        img_q = pywt.idwt2(coeffs_q, "haar")

        return img_q.astype(np.uint8)

    def dec(self, img, _i, _k):
        """Decrypts a watermark from an image _i with a key _k."""

        coeffs = pywt.dwt2(img, "haar")

        _ll, (_lh, _hl, _hh) = coeffs
        ll_q = np.round(_ll / _k)
        ll_q = np.multiply(ll_q, quant)

        bit_mess = ""
        for row, col in itertools.product(range(self.ori_row), range(self.ori_col)):
            if len(bit_mess) == self.num_bits:
                break
            if ll_q[row][col] % 2 == 1:
                bit_mess += "0"
            else:
                bit_mess += "1"

        self.bit_mess = bit_mess

    def to_bits(self):
        """Converts a string to a list of bits"""

        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8,'0')
            bits.append(binval)
        self.num_bits = bin(len(bits))[2:].rjust(8,'0')
        return bits

def add_padd(img):
    """Adds padding to an image to make it divisible by 8."""
    col = img.shape[1]
    row = img.shape[0]
    img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))    
    return img


T = DWT()
W = "1100"
img = cv2.imread("../test_images/img_1.jpeg")
img = add_padd(img)
print(img.shape)
_d = (T.enc(img,W,10))
print(T.dec(_d, None, None))
print(eval_metrics_cv(T, img, "110000"))
