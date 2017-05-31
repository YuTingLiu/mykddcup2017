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
this = 'arima_bp'
bp_train_iter = 50000
if __name__ == '__main__':
    GLOBAL = cache(GLOBAL,this=this)
    print(GLOBAL)
    traints = ['10/8/2016 06:00','10/8/2016 15:00']
    testts = ['10/25/2016 06:00','10/25/2016 15:00']
    trainlist = []
    testlist=[]
    bp_list = []
#####    #arima train and test
#    for i in range(len(traints)):
#        GLOBAL,traindf = train(GLOBAL,start=traints[i],freq='20Min',days=17,periods=12,this=this)
#        pred4hour,pred2hour = arima_pred(GLOBAL,start=testts[i],freq='20Min',periods=12,days=7,this=this)
#        trainlist.append(traindf)
#        testlist.append(pred2hour)
#        bp_list.append(pred4hour)
#    df = pd.concat(testlist)
#    df.index.name = 'time_window_s'
#    df.to_csv(r'time_final.csv')#预测集结果
#    df = pd.concat(bp_list)
#    df.to_csv(r'time_final_bp.csv')#预测集4小时结果
#    df = pd.concat(trainlist)
#    df.to_csv(r'time_total.csv')#训练集预测结果
#    GLOBAL = cache(GLOBAL,this=this)
#    final_agg()
#    
#    for i in range(len(traints)):
#        bp_tt(start=traints[i],days=17,train=True,this=this,bp_train_iter=bp_train_iter)

##############################################################
#    '''
#        在测试集上，使用BP预训练模型测试
#        测试集BP修正
#    '''
    predict = False
    test = True
    day = 7
    reg = True
    if test:
        ll = testts
    else:
        ll = traints
    dflist = []
#    for i in range(len(ll)):
#        df = bp_tt(start=ll[i],test=test,predict=predict,days=day,this=this,bp_train_iter=bp_train_iter,periods=6)
#        dflist.append(df)
#    df = pd.concat(dflist)
#    df.to_csv(r'bp_test_result.csv')
    
        ##线性回归
    if reg:
        feature = ['pred','bp']
        df,paramlist = linear_reg(fdir = r'bp_test_result.csv',feature =feature,predict=predict)
        print(paramlist)
        print(paramlist['pred'])
        print(paramlist['bp'])
        print(df.dtypes)
        df['bp']=paramlist['pred']*df.loc[:,'pred'] + paramlist['bp']*df.loc[:,'bp']
        GLOBAL['linear'] = paramlist
        GLOBAL = cache(GLOBAL,this=this)
    else:
        paramlist = GLOBAL['linear']
        print(paramlist)
        df['bp']=paramlist['pred']*df.loc[:,'pred'] + paramlist['bp']*df.loc[:,'bp'] +paramlist['gbdt']*df.loc[:,'gbdt']
    df.to_csv(''.join([this,r'after_linear.csv']))
    
    
    modifyed = []
    df = load_volume(fdir=''.join([this,r'after_linear.csv']))
    df = df[pd.notnull(df['bp'])]
    for i in range(len(ll)):
        df1 = bp_modification(df,start=ll[i],predict=predict,days=day,method=3)   
        modifyed.append(df1)
    df = pd.concat(modifyed)
    df.index.name = 'time_window_s'
    df.to_csv(r'bp_modifyed.csv')
    
#    
    #evalute
    df = load_volume(fdir=''.join([this,r'after_linear.csv']))
    df = df[pd.notnull(df['bp'])]
    for i in range(len(testts)):
        evalute(df,start=testts[i],predict=predict,days=7)
#        
#        
    bp_final_agg(fdir=r'bp_modifyed.csv')
#    
    
        
        
    '''
    输入点为当前时间t
    输出为需要预测的时间t+T
    
    所以预测8：20的流量 ，需要将时间点设置为8:00
    
    '''
