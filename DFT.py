import cv2
from PIL import Image
import itertools
from util.watermarking import Watermarking
from util.test import test_random
from util.test import encrypt_img
from util.test import eval_metrics
from util.test import eval_metrics_cv
import numpy as np

quant = np.array([[16,11,10,16,24,40,51,61],      
                    [12,12,14,19,26,58,60,55],    
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])

class DCT(Watermarking):    
    def __init__(self): 
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0   

    def Enc(self,img,secret_msg,K):
        secret=secret_msg
        self.message = str(len(secret))+'*'+secret_msg
        self.bitMess = self.toBits()

        row,col = img.shape[:2]
        self.oriRow, self.oriCol = row, col  
        
        if((col/8)*(row/8)<len(secret)):
            print("Error: Message too large to encode in image")
            return False
        
        row,col = img.shape[:2]
        bImg,gImg,rImg = cv2.split(img)
        new_channels = [bImg, gImg, rImg]
        bImg = np.float32(bImg)
        imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        quantizedDCT = [np.round(dct_Block/quant) for dct_Block in dctBlocks]
        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            DC[7] = self.bitMess[messIndex][letterIndex]
            DC = np.packbits(DC)
            DC = np.float32(DC)
            DC= DC-255
            quantizedBlock[0][0] = DC
            letterIndex = letterIndex+1
            if letterIndex == 8:
                letterIndex = 0
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        fImg = cv2.merge((sImg,gImg,rImg))
        return fImg

    def Dec(self,img, I, K):
        row,col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0
        bImg,gImg,rImg = cv2.split(img)
        bImg = np.float32(bImg)
        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]    
        quantizedDCT = [img_Block/quant for img_Block in imgBlocks]
        i=0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            if DC[7] == 0:
                buff+= 1 << (7-i)
            i=1+i
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i =0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                        
                    except:
                        pass
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize))+1:]
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        return ''
      
    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]


    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8,'0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8,'0')
        return bits


def addPadd(img):
    col = img.shape[1]
    row = img.shape[0]
    img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))    
    return img


T = DCT()
W = "1100"
img = cv2.imread("../test_images/img_1.jpeg")
img = addPadd(img)
print(img.shape)
D = (T.Enc(img,W,10))
print(T.Dec(D, None, None))
print(eval_metrics_cv(T, img, "110000"))
