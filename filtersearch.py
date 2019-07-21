#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 11:05:11 2019

@author: zcgu
"""
import cv2
import pandas as pd
import darknet as dn
import os
from os.path import expanduser
import utils
from tqdm import tqdm
from tqdm import trange
import random
import itertools
import time
import numpy as np

# Setup
cfg_path=expanduser("~") + "/cuda-workspace/yolo-visualization/yolov3-ILSVRC.cfg"
weight_path=expanduser("~") + "/cuda-workspace/yolo-visualization/weights_bk/yolov3-ILSVRC_final_2.weights"
meta_path=expanduser("~") + "/cuda-workspace/yolo-visualization/ILSVRC.data"
names_path = expanduser("~") + "/workspace/data/kaggle/imglocalization/LOC_synset_mapping.txt/LOC_synset_mapping.txt"

data_path = expanduser("~") + "/cuda-workspace/yolo-visualization/ILSVRC/Data/CLS-LOC/val/images/ILSVRC2012_val_{}.JPEG"
label_path = expanduser("~") + "/cuda-workspace/yolo-visualization/ILSVRC/Data/CLS-LOC/val/labels/ILSVRC2012_val_{}.txt"

# Dump score
dump_score = 1
nREP = 3
rangeFilter = range(1,32,3)
rangeHierFilter = range(1,32,3)


# encapsulate prediction to a single function
def __predict(yolo,path,names, thresh = 0.25, hier_thresh = 0.25):
    # r = detect(net, meta, str(path).encode(), .01)
    r = yolo.detect(str(path), thresh = thresh, hier_thresh = hier_thresh)
    # format prediction
    res=str()
    pred=[[str(names.loc[names[0]==str(val[0], "utf-8")[0:9]].index[0]+1) 
        + " " + " ".join(map(str, val[2]))] for val in r]
    if len(pred) > 0:
        res= pred[:5]
    return res

def predict(yolo,names,indices,t,ht,dumpdir='',niter=0):
    # predict for all test data
    scores = [] 
    ks = []
    fs = []
    
    for idx in trange(len(indices), leave = False):
        i = indices[idx]
        #img="ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_"+str("%08d" % (i+1))+'.JPEG'
        img_p=data_path.format(str("%08d" % (i+1)))
        label_p=label_path.format(str("%08d" % (i+1)))
        
        # Predict
        predres=__predict(yolo, img_p, names, thresh = t, hier_thresh = ht)
        #print(img_p,":\n", predres)
        
        # Nomalize prediction
        ## Load img
        imgobj = cv2.imread(img_p)
        predres = utils.normalize_pred(predres,imgobj.shape[1],imgobj.shape[0])
#        print("DEBUG: Prediction = {}".format(predres))
        # Load lables
        label = utils.load_label(label_p)
#        print("DEBUG: LABEL = {}".format(label))
        # Find score
        min_score, mink, minf = utils.calScore(predres,label)
        scores.append(min_score)
        ks.append(mink)
        fs.append(minf)
    
    if (dump_score == 1):
        df = pd.DataFrame({'indices':indices,'scores':scores,'ks':ks,'fs':fs})
        df.to_csv(os.path.join(dumpdir,"score_{}_{}_{}.dump".format(t,ht,niter)), index=False)
        
    return sum(scores)/len(scores)

def grid_search(yolo,names,t_range, ht_range):
    combinations = list(itertools.product(t_range,ht_range))
    print("/n Total Search combinations = {}".format(len(combinations)))
    scores = []
    time_stamp = int(time.time()*(10e5))
    cwd = os.getcwd()
    cwd_gs = os.path.join(cwd,"gridsearch")
    if not os.path.isdir(cwd_gs):   
        os.mkdir(cwd_gs)
    
    if (dump_score == 1):
        dump_path = os.path.join(cwd_gs,"dump{}".format(time_stamp))
        os.mkdir(dump_path)
        
    for idx in trange(len(combinations)):
        item = combinations[idx]
        scores_tmp = []
        for rep in trange(nREP,leave = False):
            indices = random.sample(range(0,50000),1000)
            scores_tmp.append(predict(yolo,names,indices,item[0]/100,item[1]/100,dump_path,rep))
        scores.append(np.mean(scores_tmp))
    
    df = pd.DataFrame({'combinations':combinations,'scores':scores})
    df.to_csv("gridsearch/gs_{}.csv".format(time_stamp), index=False)
    
    
    

def init_yolo(cfg,weight,meta):
    yolo = dn.Yolo()
    yolo.set_net(cfg,weight)
#    yolo.set_meta(expanduser("~")+"/tools/yolo/darknet/cfg/coco.data")
    yolo.set_meta(meta)
    return yolo




def main():
    yolo = init_yolo(cfg_path,weight_path,meta_path)
    
    names=pd.read_csv(names_path, sep='\t', header=None)
    names[1]=names[0].str[10:]
    names[0]=names[0].str[0:9]
    
    grid_search(yolo, names,rangeFilter, rangeHierFilter)

#    grid_search(yolo,names,range(1,2),range(1,2))
    return 1


if __name__ == "__main__":
    main()
    
