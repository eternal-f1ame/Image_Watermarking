import numpy as np
import cv2
from util.metrics import METRICS

def test_random(T):
    cover_image = np.random.rand(100, 100, 3)*255
    cover_image = cover_image.astype(np.uint)
    K = 10
    W = np.random.rand(100, 100, 3)*255
    W = W.astype(np.uint)
    encrypted = T.Enc(cover_image, W, K)
    return np.array_equal(W,T.Dec(encrypted, cover_image, K))

def encrypt_img(T, img_path, K=10):
   cover_image = cv2.imread(img_path)
   cover_image = cover_image.astype(np.uint8)
   W = np.random.rand(cover_image.shape[0],cover_image.shape[1],cover_image.shape[2])*2
   W = W.astype(np.uint8)
   encrypted = T.Enc(cover_image, W, K)
   cv2.imshow("enc",encrypted)
   cv2.waitKey(0)
   return T.Enc(cover_image, W, K)
   
def eval_metrics(T, img_path, W, K = 10):
    cover_img = cv2.imread(img_path)
    cover_img = cover_img.astype(np.uint8)
    W = W.astype(np.uint8)
    W = np.resize(W, cover_img.shape)
    D = T.Enc(cover_img, W, K)
    metrics = {}
    for metric in METRICS.keys():
        metrics[metric] = METRICS[metric](D, cover_img)
    return metrics
