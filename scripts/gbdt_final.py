# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:24:34 2017

@author: Administrator
"""

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
this = 'gbdt'

def gg(start='10/18/2016 06:00',test_size=12,freq='20Min',periods=12,
          days=7,predict=False,test=False,train=False,this=''):
    
    outputlist = []
    
    df,train_seq = load(start=start,freq=freq,normalize = False,periods=periods,days=days,pat=True)

    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
#    df.loc[:,'residual'] = np.log(df['travel_time'])-np.log(df['pred'])

    hour = train_seq.hour[0]#param
    if predict:
        train_seq = train_seq + Hour(2)
#    df.to_csv(r'bp.csv',index=False)

    inter = df.groupby('intersection_id')
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            batch_x = dgroup.set_index('time_window_s').loc[train_seq,:]
            print(batch_x['travel_time'])
            batch_x['travel_time'] = modification(batch_x['travel_time'],method=3,show=False)
            batch_x.loc[:,'residual'] = np.log(batch_x['travel_time'])
            while np.any(pd.isnull(batch_x)):
                print('nan found ',t,'',d)
                print(batch_x.head())
                batch_x = batch_x.bfill()
                batch_x = batch_x.ffill()
#                print(batch_x.head())
#                sys.exit()
            columns = ['Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday','midnight','morning',\
            'noon','afternoon','evening','w0','w20','w40','000','100','102','103','104','105','107','108','109','110',\
            '111','112','115','116','119','120','122','123','out_link_cross_count','in_link_cross_conut','length',\
            'link_count','1_length','2_length','3_length','4_length','pressure','sea_pressure','wind_direction',\
            'wind_speed','temperature','rel_humidity','precipitation','pattern']
#            print(batch_x.dtypes)
#            print(batch_x[columns])
            
            x_data = bp_pca(np.array(batch_x[columns]))
#    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
#                                'temperature','rel_humidity','precipitation',
#                                'dayofweek','hour','minute','direction','pattern']])
                   
            y_data = np.array(batch_x.loc[:,'residual'])#修改为残差
            y_data = y_data.reshape((len(y_data),1))
    
            if train:
                gbdt = kdd_gbdt()
                gbdt.save_name = ''.join([this,str(t),str(d),str(0)])
                gbdt.train_times = 1
                
                best_params = gbdt.choose(x_data,y_data)
                print(best_params)
                gbdt.params = best_params
                gbdt.train(x_data,y_data)
                gbdt.save_model()
            else:
                gbdt = kdd_gbdt()
                gbdt.save_name = ''.join([this,str(t),str(d),str(hour)])
                gbdt.load_model()
                pred = gbdt.predict(x_data)
                print('pred length',len(pred))
                print('true length ',len(y_data))
                pred = pd.Series(data=np.array(pred).flatten(),index = train_seq)
                pred = modification(pred,method=3,show=True)
                batch_x.loc[:,'bp'] = np.exp(pred)
                outputlist.append(batch_x[['intersection_id','tollgate_id','travel_time','gbdt']]) 
    if len(outputlist)>0:
        print('ouputlist',len(outputlist))
        df = pd.concat(outputlist)
        df.index.name = 'time_window_s'
        return df
    
    
if __name__ == '__main__':
#    GLOBAL = cache(GLOBAL)
#    print(GLOBAL)
    traints = ['10/8/2016 06:00','10/8/2016 15:00']
    testts = ['10/25/2016 06:00','10/25/2016 15:00']
    trainlist = []
    testlist=[]
    bp_list = []
    
    #GBDT训练
    gg(start='9/19/2016 00:00',days=30,train=True,this = this,periods=72)
        
    ####训练集BP修正
    
    ##GBDT 测试
##############################################################
    predict = False
    test = True
    day = 7
    if test:
        ll = testts
    else:
        ll = traints
    dflist = []
    for i in range(len(ll)):
        df = gg(start=ll[i],test=test,predict=predict,days=day,this=this,periods=6)
        dflist.append(df)
    df = pd.concat(dflist)
    df.to_csv(''.join([this,r'_test_result.csv']))
    
    modifyed = []
    df = load_volume(fdir=''.join([this,r'_test_result.csv']))
    df = df[pd.notnull(df['bp'])]
    for i in range(len(ll)):
        df1 = bp_modification(df,start=ll[i],predict=predict,days=day,method=3)   
        modifyed.append(df1)
    df = pd.concat(modifyed)
    df.index.name = 'time_window_s'
    df.to_csv(''.join([this,r'_test_result.csv']))
#    
    #evalute
    df = load_volume(fdir=''.join([this,r'_test_result.csv']))
    df = df[pd.notnull(df['bp'])]
    for i in range(len(ll)):
        evalute(df,start=ll[i],predict=predict,days=7)
        
    '''
    输入点为当前时间t
    输出为需要预测的时间t+T
    
    所以预测8：20的流量 ，需要将时间点设置为8:00
    
    '''
