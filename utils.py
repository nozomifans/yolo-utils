#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 00:03:05 2019

@author: zcgu
"""

import cv2
import pandas as pd
    
def coord_yolo_to_cor(box):
    # https://blog.goodaudience.com/part-1-preparing-data-before-training-yolo-v2-and-v3-deepfashion-dataset-3122cd7dd884
    #img_h, img_w, _ = shape
   
    #x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    #x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    x1, y1 = (box[0] + box[2]/2), (box[1] + box[3]/2)
    x2, y2 = (box[0] - box[2]/2), (box[1] - box[3]/2)
    return [x1,y1,x2,y2]

def coord_cor_to_yolo(box):
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    xmax, xmin = sorting(box[0], box[2])
    ymax, ymin = sorting(box[1], box[3])
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x
    w = w
    y = y
    h = h
  
    return [x,y,w,h]

def overlap(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    #print("DEBUG: BOX A = {}")
    boxA = coord_yolo_to_cor(boxA)
    boxB = coord_yolo_to_cor(boxB)
    
#    print("DEBUG: BOX_A = {}".format(boxA))
#    print("DEBUG: BOX_B = {}".format(boxB))
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
#    print("DEBUG: xA,yA,xB,yB = {}, {}, {}, {}".format(xA,yA,xB,yB))
    
    
	# compute the area of intersection rectangle
    interArea = max(0, xA - xB ) * max(0, yA - yB )
#    print("DEBUG: InterArea = {}".format(interArea))
    
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[0] - boxA[2]) * (boxA[1] - boxA[3])
    boxBArea = (boxB[0] - boxB[2]) * (boxB[1] - boxB[3])
#    print("DEBUG: boxAArea = {}".format(boxAArea))
#    print("DEBUG: boxBArea = {}".format(boxBArea))
    
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
#    print("IOU = {}".format(iou))
    return iou

def calScore(s_p, s_t):
    '''
    Kaggle score
    '''
    min_error_list = []
    if not s_p:
        return 1, 1, 1
    else:
        d_list = []
        f_list = []
        for prediction_box in s_p:
            max_error_list = []
            for ground_truth_box in s_t:
                if prediction_box[0] == ground_truth_box[0]:
                    d = 0
                else:
                    d = 1
                if overlap(prediction_box[1:5], ground_truth_box[1:5]) > 0.5:
                    f = 0
                else:
                    f = 1
                d_list.append(d)
                f_list.append(f)
                max_error_list.append(max(d,f))   # the first max
            min_error_list.append(min(max_error_list))  # the first min
            
    return min(min_error_list), min(d_list), min(f_list)


def normalize_pred(_input,w,h):
    res = []
    if _input is not None:
        for obj in _input:
            obj = obj[0].split()
            obj = list(map(int,list(map(float, obj))))
            obj[1] = obj[1]/w
            obj[3] = obj[3]/w
            obj[2] = obj[2]/h
            obj[4] = obj[4]/h
            res.append(obj) 
    return res

def draw_box(boxA,color,imgobj):
    shape = imgobj.shape
    boxA = coord_yolo_to_cor(boxA,shape)
    x1,y1 = int(boxA[0]*shape[1]), int(boxA[1]*shape[0])
    x2,y2 = int(boxA[2]*shape[1]), int(boxA[3]*shape[0])
    cv2.rectangle(imgobj, (x1, y1), (x2, y2), color, 2)


def visualize(s_p, s_t, imgobj):
    for obj in s_p:
        draw_box(obj[1:5],(255,0,0),imgobj)
    for obj in s_t:
        draw_box(obj[1:5],(0,0,255),imgobj)
    cv2.imshow("image",imgobj)
    cv2.waitKey(0)

def load_label(path):
    labels = pd.read_csv(path, sep = " ", header = None)
    labels.iloc[:,0] = labels.iloc[:,0]+1
    return labels.values