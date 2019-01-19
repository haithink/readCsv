# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:30:23 2019

@author: dell
"""

import imgaug as ia
from imgaug import augmenters as iaa
import pandas as pd
import cv2
import os



  
seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px={"x": 0, "y": 0},
        scale=(1, 1),
        #shear = 10.0
    ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])
    
#生成目标图片不需要遍历目标框，因为所有图片的变换是一样的，只需要变换一次即可
imgPath = "D:\\dccontest\\dataset\\train_dataset"
files = os.listdir(imgPath)

seq_det = seq.to_deterministic()

#for file in files:
#    if(file.endswith('.jpg')):
#        img = cv2.imread(imgPath + "/" + file)
#        image_aug = seq_det.augment_images([img])[0]
#        cv2.imwrite("res/"+ file,image_aug)
#        print(file)

width = 2666
height = 2000
channel = 3
label = "cyl"

boxes = []
    
filename = "NoExist"

def genImage(boxes, filename):
    bbs = ia.BoundingBoxesOnImage(boxes, shape=(height, width, channel))
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    #  测试 转换后的坐标框是否正确!
    img = cv2.imread(imgPath + "/" + filename)
    image_aug = seq_det.augment_images([img])[0]
    image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])
    cv2.imwrite("GT/" + filename,image_after)
    return 

data = pd.read_csv('train.csv')

for i in range(data.shape[0]):
    #print(data.loc[i][4])
    if(i == 0):
        filename = data.loc[i][0]
        
    if(filename != data.loc[i][0]):
        genImage(boxes, filename)
        boxes = []
        
        box = ia.BoundingBox(x1=data.loc[i][4], y1=data.loc[i][5], x2=data.loc[i][6], y2=data.loc[i][7])
        boxes.append(box)
        filename = data.loc[i][0]
    else:
        box = ia.BoundingBox(x1=data.loc[i][4], y1=data.loc[i][5], x2=data.loc[i][6], y2=data.loc[i][7])
        boxes.append(box)
        
genImage(boxes, filename)        
        
        