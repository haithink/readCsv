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

data = pd.read_csv('train.csv')


    
seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px={"x": 0, "y": 0},
        scale=(0.5, 0.5),
        shear = 10.0
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

fcsv = open("AffineTrain.csv", 'w')

for i in range(data.shape[0]):
    #print(data.loc[i][4])
    boxes = []
    box = ia.BoundingBox(x1=data.loc[i][4], y1=data.loc[i][5], x2=data.loc[i][6], y2=data.loc[i][7])
    boxes.append(box)
    
    bbs = ia.BoundingBoxesOnImage(boxes, shape=(2000, 2666, 3))
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    before = bbs.bounding_boxes[0]
    after = bbs_aug.bounding_boxes[0]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )
    
    fcsv.write(data.loc[i][0]+",")
    fcsv.write(str(int(after.x1))+","+str(int(after.y1))+","+str(int(after.x2))+","+str(int(after.y2)) + "\n")

fcsv.close()


    