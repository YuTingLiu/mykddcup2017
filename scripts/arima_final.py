# -*- coding: utf-8 -*-
"""
Created on Sat May 27 23:10:03 2017

@author: L
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from data_util import *
import sys
import os

#时间平移
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Week
from pandas.tseries.offsets import Day
from pandas.tseries.offsets import Minute

import matplotlib.pyplot as plt
import json
import pickle

from main_time import *
    
GLOBAL = {}
train_f = r'../train/'
arima_path = r'../arima/'
this = 'arima'
bp_train_iter = 50000
if __name__ == '__main__':
    GLOBAL = cache(GLOBAL,this=this)
    print(GLOBAL)
    traints = ['10/8/2016 06:00','10/8/2016 15:00']
    testts = ['10/25/2016 06:00','10/25/2016 15:00']
    trainlist = []
    testlist=[]
    bp_list = []
####    #arima train and test
#    for i in range(len(traints)):
#        GLOBAL,traindf = train(GLOBAL,start=traints[i],freq='20Min',days=17,periods=12,p=10,this=this)
#        pred4hour,pred2hour = arima_pred(GLOBAL,start=testts[i],freq='20Min',periods=12,days=7,this=this)
#        trainlist.append(traindf)
#        testlist.append(pred2hour)
#        bp_list.append(pred4hour)
#    df = pd.concat(testlist)
#    df.index.name = 'time_window_s'
#    df.to_csv(r'arima_time_final.csv')#预测集结果
#    df = pd.concat(bp_list)
#    df.to_csv(r'arima_time_final_bp.csv')#预测集4小时结果
#    df = pd.concat(trainlist)
#    df.to_csv(r'arima_time_total.csv')#训练集预测结果
#    GLOBAL = cache(GLOBAL,this=this)
#    
##    
##    
    #evalute
    l = []
    df = load_volume(fdir=r'arima_time_final_bp.csv')
    df = df[pd.notnull(df['pred'])].set_index('time_window_s')
    for i in range(len(testts)):
        _,seq = gen_df(fdir=r'arima_time_final_bp.csv',start=testts[i],periods=6,days=7)
        print('seq ',len(seq))
        l.append(df.loc[seq,:])
    df = pd.concat(l)
    final_agg(fdir=r'arima_time_final_bp.csv')
#    
    
        
        
    '''
    输入点为当前时间t
    输出为需要预测的时间t+T
    
    所以预测8：20的流量 ，需要将时间点设置为8:00
    
    '''
