"""DWT watermarking algorithm."""

import numpy as np
import cv2
import pywt
from ..util.watermarking import Watermarking
from ..util.test import test_random
from ..util.test import encrypt_img
from ..util.test import eval_metrics
from ..util.test import eval_metrics_cv

class DWT(Watermarking):
    """Class for watermarking using DWT algorithm."""

    def __init__(self):
        self.type = "DWT"
        self.message = None
        self.bit_mess = None
        self.ori_col = 0
        self.ori_row = 0
        self.num_bits = 0

    def enc(self, img, secret_msg, _k):
        """Encrypts an image with a watermark."""

        img = self.add_padd(img)
        b_img, g_img, r_img = cv2.split(img)

        image = b_img

        ori_row, ori_col = image.shape
        self.message = str(len(secret_msg))+'*'+secret_msg
        self.bit_mess = self.to_bits()
        binary_watermark = self.bit_mess

        coeffs = pywt.dwt2(image, "haar")
        _ll, (_ch, _cv, _cd) = coeffs
        _ll = np.array(_ll, dtype=np.int32)

        binary_watermark_index = 0
        for i in range(_ll.shape[0]):
            for j in range(_ll.shape[1]):
                if binary_watermark_index >= len(binary_watermark):
                    break
                binary_pixel = np.binary_repr(_ll[i, j], width=8)
                modified_pixel = int(binary_pixel[:-1] + binary_watermark[binary_watermark_index], 2)
                _ll[i, j] = modified_pixel
                binary_watermark_index += 1
            if binary_watermark_index >= len(binary_watermark):
                break

        modified_coeffs = (_ll, (_ch, _cv, _cd))
        watermarked_image = pywt.idwt2(modified_coeffs, "haar")

        watermarked_image = cv2.resize(watermarked_image, (ori_col, ori_row))
        r_img = cv2.resize(r_img, (ori_col, ori_row))
        g_img = cv2.resize(g_img, (ori_col, ori_row))
        ret_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        ret_image[:, :, 0] = watermarked_image
        ret_image[:, :, 1] = g_img
        ret_image[:, :, 2] = r_img
        return ret_image

    def dec(self,img, _i, _k):
        """Extracts the watermark from an image."""

        coeffs = pywt.dwt2(img, "haar")
        _ll, (_, _, _) = coeffs

        binary_watermark = ""
        binary_watermark_index = 0
        for i in range(_ll.shape[0]):
            for j in range(_ll.shape[1]):
                if binary_watermark_index >= len(binary_watermark):
                    break

                binary_pixel = np.binary_repr(_ll[i, j], width=8)
                binary_watermark += binary_pixel[-1]
                binary_watermark_index += 1
            if binary_watermark_index >= len(binary_watermark):
                break

        watermark = "".join(
            chr(int(binary_watermark[i:i+8], 2)) for i in range(0, len(binary_watermark), 8))

        return watermark

    def chunks(self, _l, _n):
        """Yield successive _n-sized chunks from l."""
        m = int(_n)
        for i in range(0, len(_l), m):
            yield _l[i:i + m]

    def to_bits(self):
        """Converts a string to a list of bits"""
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8,'0')
            bits.append(binval)
        self.num_bits = bin(len(bits))[2:].rjust(8,'0')
        return bits

    def add_padd(self, img):
        """Adds padding to an image to make it divisible by 8."""
        col = img.shape[1]
        row = img.shape[0]
        img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))
        return img

T = DWT()
print(eval_metrics(T, "test_images/building/000.jpg", "Hello World"))
enc_im = encrypt_img(T, "test_images/building/000.jpg", "Hello World")
print(T.dec(enc_im, 0, 0))
