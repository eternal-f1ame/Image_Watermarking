"""DFT watermarking algorithm."""

import itertools
import numpy as np
import cv2
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

class DCT(Watermarking):
    """Class for watermarking for DCT algorithm."""

    def __init__(self):
        self.type = "DCT"
        self.message = None
        self.bit_mess = None
        self.ori_col = 0
        self.ori_row = 0
        self.num_bits = 0

    def enc(self,img,secret_msg,_k):
        """Encrypts an image _i with a watermark _w and a key _k."""

        secret=secret_msg
        self.message = str(len(secret))+'*'+secret_msg
        self.bit_mess = self.to_bits()

        img = self.add_padd(img)
        row,col = img.shape[:2]
        self.ori_row, self.ori_col = row, col

        if(col/8)*(row/8)<len(secret):
            print("Error: Message too large to encode in image")
            return False

        row,col = img.shape[:2]
        b_img,g_img,r_img = cv2.split(img)
        new_channels = [b_img, g_img, r_img]
        b_img = np.float32(b_img)
        img_blocks = [np.round(b_img[j:j+8, i:i+8]-128)
                        for (j,i) in itertools.product(range(0,row,8), range(0,col,8))]
        dct_blocks = [np.round(cv2.dct(img_Block)) for img_Block in img_blocks]
        quantized_dct = [np.round(dct_Block/quant) for dct_Block in dct_blocks]
        mess_index = 0
        letter_index = 0
        for quantized_block in quantized_dct:
            _dc = quantized_block[0][0]
            _dc = np.uint8(_dc)
            _dc = np.unpackbits(_dc)
            _dc[7] = self.bit_mess[mess_index][letter_index]
            _dc = np.packbits(_dc)
            _dc = np.float32(_dc)
            _dc= _dc-255
            quantized_block[0][0] = _dc
            letter_index = letter_index+1
            if letter_index == 8:
                letter_index = 0
                mess_index = mess_index + 1
                if mess_index == len(self.message):
                    break
        s_img_blocks = [quantized_block *quant+128 for quantized_block in quantized_dct]
        s_img=[]
        for chunk_row_blocks in self.chunks(s_img_blocks, col/8):
            for row_block_num in range(8):
                for block in chunk_row_blocks:
                    s_img.extend(block[row_block_num])
        s_img = np.array(s_img).reshape(row, col)
        s_img = np.uint8(s_img)
        f_img = cv2.merge((s_img,g_img,r_img))
        return f_img

    def dec(self,img, _i, _k):
        row,col = img.shape[:2]
        mess_size = None
        message_bits = []
        buff = 0
        b_img,g_img,r_img = cv2.split(img)
        b_img = np.float32(b_img)
        img_blocks = [b_img[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]    
        quantized_dct = [img_Block/quant for img_Block in img_blocks]
        i=0
        for quantized_block in quantized_dct:
            _dc = quantized_block[0][0]
            _dc = np.uint8(_dc)
            _dc = np.unpackbits(_dc)
            if _dc[7] == 0:
                buff+= 1 << (7-i)
            i=1+i
            if i == 8:
                message_bits.append(chr(buff))
                buff = 0
                i =0
                if message_bits[-1] == '*' and mess_size is None:
                    try:
                        mess_size = int(''.join(message_bits[:-1]))

                    except: pass

            if len(message_bits) - len(str(mess_size)) - 1 == mess_size:
                return ''.join(message_bits)[len(str(mess_size))+1:]
        s_img_blocks = [quantized_block *quant+128 for quantized_block in quantized_dct]
        s_img=[]
        for chunk_row_blocks in self.chunks(s_img_blocks, col/8):
            for row_block_num in range(8):
                for block in chunk_row_blocks:
                    s_img.extend(block[row_block_num])
        s_img = np.array(s_img).reshape(row, col)
        s_img = np.uint8(s_img)
        s_img = cv2.merge((s_img,g_img,r_img))
        return ''

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


T = DCT()
print(eval_metrics(T, "test_images/building/000.jpg", "Hello World"))
encrypt_img(T, "test_images/building/000.jpg", "Hello World")
print(test_random(T))
