import cv2
import os
import numpy as np

IMAGE_DIR ="/Users/ayushanand/programming/course_work/CV_project_1/Image_Watermarking/visual/coco128/images/train2017/"

for img_name in os.listdir(IMAGE_DIR):
    img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    img = cv2.resize(img, (300,300))
    watermark = cv2.imread("Watermark.png")
    watermark = cv2.resize(watermark, (300, 300))
    final = cv2.addWeighted(img, 0.8, watermark, 0.2, 0)
    cv2.imwrite(os.path.join("./outputs/",img_name), final)
