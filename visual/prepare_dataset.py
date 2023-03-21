"""Prepare Output Dataset for Visual Watermarking"""
import os
import cv2

IMAGE_DIRS  = ["COCO_Dataset/", "100_Image_Dataset"]

for IMAGE_DIR in IMAGE_DIRS:
    for img_name in os.listdir(("../"+IMAGE_DIR)):
        img = cv2.imread("../"+os.path.join(IMAGE_DIR, img_name))
        img = cv2.resize(img, (300,300))
        watermark = cv2.imread("Watermark.png")
        watermark = cv2.resize(watermark, (300, 300))
        final = cv2.addWeighted(img, 0.8, watermark, 0.2, 0)
        cv2.imwrite(os.path.join(("outputs/"+IMAGE_DIR),img_name), final)
