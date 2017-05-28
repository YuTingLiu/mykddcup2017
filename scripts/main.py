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

from arima_bp import bp_net
from rbf_test import rbf_net
from time_series_analysis_p1 import *

def gen_df(start='10/8/2016 06:00',freq='T',normalize = False,test = False,pat = False,periods=72):
    '''
    help fun
    '''
    if freq is '20Min':
        df = load_volume(fdir='train_union1.csv')
        train_seq = produce_seq(start=start,periods=periods,freq='20Min',days = 7)
        if len(train_seq) != 72*20:
            print('train_seq len ',len(train_seq))
#            sys.exit()
        if test is True:
            df = load_test(fdir='test_union1.csv')
            test_seq = produce_seq(start=start,periods=periods,freq='20Min',days = 6)
            df = prep(df,pat=pat,normalize=normalize)
        else:
            test_seq = produce_seq(start='10/16/2016 06:00',periods=periods,freq='20Min',days = 1)
            df = prep(df,pat=pat,normalize=normalize)
    else:
        sys.exit(0)
#    print(df)
    return df,train_seq,test_seq

def gen_test(start='10/18/2016 08:00',freq='T',normalize = False,test = True,pat = False,periods=6):
    
    test_seq = produce_seq(start=start,periods=periods,freq='20Min',days = 7)
    return test_seq
    
def produce_seq(start='09/19/2016 00:20',periods=20,freq='T',days = 7):
    '''
    help fun to produce time sequence
    '''
    seq = pd.date_range(start=start,periods=periods,freq=freq)
    seq1 = seq
    for day in  range(days-1):
        seq = seq + Day()
        seq1 += seq
    return seq1

def next_20min(seq, m=20):
    '''
    add constant time to seq
    '''
    return seq + Minute(m)

def plot_pred(pred,y_true):
    '''
    input :pred,true
    datashape : [[1,2],[1,2]]
    
    '''    
    plt.plot(pred[:,0].flatten(),'blue')
    plt.plot(y_true[:,0].flatten(),'*')
    plt.show()
    

def next_seq(seq,offset="20Min"):
    return seq  + Minute(offset)
    
		
def train(GLOBAL,start='10/8/2016 06:00',freq = '20Min',tollgate = 1):
    '''
    主要的内容是:数据按照时间分割,送入ARIMA训练,得到结果1
	      整合结果1,送入BPNN训练,得到训练好的网络
         输出:每个tollgate一个模型
    '''
    df,train_seq,test_seq = gen_df(start=start,freq=freq,normalize = False,test=False,periods=12)

    df.loc[:,'day'] = df['time_window_s'].dt.dayofyear
    tolls = df.groupby('tollgate')
    outputlist = []
    for t,tgroup in tolls:
        ds = tgroup.groupby('direction')
        
        train_seq1 = train_seq.copy()
        for d,dgroup in ds:
            if t==2 and d==1:
                continue
            group = dgroup.set_index('time_window_s')[['volume']]
#            data_check()
            print('ARIMA模型,训练X天的数据,默认时间序列周期为一天,通过建立模型,保存模型')
            print('time seq',train_seq1[0],train_seq1[-1])
#            sys.exit()
            for i in range(1):
                day = train_seq1.day[0]
                hour = train_seq1.hour[0]
                ts = group.loc[train_seq1,:]
                from time_series_analysis_p2 import run_aram
                model,MRSE,pred,diffn = run_aram(ts,1,1)
                param = ''.join([str(t),str(d),str(hour)])
                GLOBAL[param] = str(diffn)
                f = open(''.join([r'../arima','//',str(t),str(d),str(hour),'-',str(diffn)]),'wb')
                pickle.dump(model,f)
                f.close()
                #还原
                ts.loc[:,'pred'] = pred
                plot_compare(ts['volume'],ts['pred'],0)
                ts.loc[:,'direction'] = d
                ts.loc[:,'tollgate'] = t
                ts.index.name = 'time_window_s'
                outputlist.append(ts)
#        sys.exit(0)
    output = pd.concat(outputlist)
    output.to_csv('output1.csv')
    df = df.set_index(['tollgate','direction','time_window_s'])
    df = df.join(output.reset_index().set_index(['tollgate','direction','time_window_s'])[['pred']])
    df.to_csv('total_arima_bp_log1.csv',index = True)
    return GLOBAL,df
#        sys.exit(0)

def arima_pred(GLOBAL,start='10/18/2016 06:00',freq = '20Min',tollgate = 1):
    '''
    '''
    print(GLOBAL)
#    sys.exit()
    outputlist=[]
    df,train_seq,test_seq = gen_df(start=start,freq=freq,normalize = False,test=True,periods=12,pat=False)
    _,_,seq = gen_df(start=start,freq=freq,normalize = False,test=True,periods=6,pat=False)
    seq = seq + Hour(2)
    df.loc[:,'day'] = df['time_window_s'].dt.dayofyear
    print(df.head())
    tolls = df.groupby('tollgate')
    for t,tgroup in tolls:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            if t==2 and d==1:
                continue
            group = dgroup.set_index('time_window_s')[['volume']]
            ts = group.loc[test_seq,:]
            hour = train_seq.hour[0]
            param = ''.join([str(t),str(d),str(hour)])
            diffn = GLOBAL[param]
            f = open(''.join([r'../arima','//',str(t),str(d),str(hour),'-',str(diffn)]),'rb')
            model = pickle.load(f)
            f.close()
            print(model)
            pred = model.predict(len(ts))
            print(len(test_seq))
            print(len(pred))
            print(len(ts))
#            print(pred)
            from time_series_analysis_p2 import predict_recover
            df = predict_recover(pred,ts['volume'],0)
#            print(df)
            temp = pd.Series(data=df['Series'].values,index=ts.index)
            ts.loc[:,'pred'] = temp
            ts.loc[:,'tollgate'] = t
            ts.loc[:,'direction'] = d
            ts.loc[:,'time_window_e'] = ts.index + Minute(20)
            ts.index.name = 'time_window_s'
#            plot_compare(ts['volume'],ts['pred'],0)
            print(ts.loc[seq,:].head(12))
#            sys.exit()
            outputlist.append(ts.loc[seq,:])
    df = pd.concat(outputlist)
    df.to_csv(r'pred_finnal.csv')
    return df
               
def bp_train(tollgate=1):
    _,train_seq,test_seq = gen_df(freq,normalize = False,test=False,periods=12)
    df = pd.read_csv('total_arima_bp_log1.csv')
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    df.loc[:,'residual'] = df['volume']-df['pred']
#    df.loc[:,'residual'] = np.log(df['volume'])-np.log(df['pred'])
    df.to_csv(r'bp.csv',index=False)
    batch_x = df[pd.notnull(df['pred'])]
    print(batch_x)
    batch_x = batch_x[batch_x['tollgate']==tollgate]
    x_data = np.array(batch_x[['pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute','direction','pattern']])
#    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
#                                'temperature','rel_humidity','precipitation',
#                                'dayofweek','hour','minute','direction','pattern']])
    if np.sum(np.isnan(x_data)) > 0:
        print('nan found ')
        sys.exit(0)
    y_data = np.array(batch_x.loc[:,'residual'])#修改为残差
    y_data = y_data.reshape((len(y_data),1))
#    print(x_data)
#    print(y_data)
    
    #begin bp train
    bp = bp_net()
    N,D = x_data.shape
    K = len(y_data[0])
    bp.middle = 30
    bp.modelname = str(tollgate)
    bp.training_iters = 25000
    bp.build(D,K)
    bp.fit(x_data,y_data,x_data,y_data)
#    print(train_seq)
    print('train',x_data.shape,y_data.shape)
    print('train seq ',train_seq[0],train_seq[-1])
    print('test seq',test_seq[0],test_seq[-1])
                    
def rbf_train(tollgate=1):
    _,train_seq,test_seq = gen_df(freq,normalize = False,test=False)
    df = pd.read_csv('total_arima_bp_log.csv')
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    df.loc[:,'residual'] = np.log(df['volume'])-np.log(df['pred'])
    df.to_csv(r'bp.csv',index=False)
    batch_x = df[pd.notnull(df['pred'])]
    batch_x = batch_x[batch_x['tollgate']==tollgate]
    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute','direction','pattern']])
    if np.any(np.isnan(x_data)):
        print('nan found ')
        sys.exit(0)
    y_data = np.array(batch_x.loc[:,'residual'])#修改为残差
    y_data = y_data.reshape((len(y_data),1))
#    print(x_data)
#    print(y_data)
    
    #begin bp train
    bp = rbf_net()
    N,D = x_data.shape
    K = len(y_data[0])
    bp.middle = 150
    bp.modelname = str(tollgate)
    bp.training_iters = 25000
    bp.build(D,K)
    bp.fit(x_data,y_data,x_data,y_data)
#    print(train_seq)
    print('train',x_data.shape,y_data.shape)
    print('train seq ',train_seq[0],train_seq[-1])
    print('test seq',test_seq[0],test_seq[-1])


def cache(config,fdir='global_config.txt'):
    if len(config)>0:
        f = open(fdir,'w')
        json.dump(config,f)
        f.close()
        return config
    else:
        f = open(fdir,'r')
        config = json.load(f)
        f.close()
        return config
def final_agg(fdir=r'final.csv'):
    df = pd.read_csv(fdir)
    df.loc[:,'time_window'] = '[' + df['time_window_s'] +','+ df['time_window_e'] +')'
    df.loc[:,'volume'] = df.loc[:,'pred']
    df = df[['tollgate','time_window','direction','volume']]
    df.to_csv(r'../submit_volume.csv',index=False)
    
GLOBAL = {}
train_f = '../train'
if __name__ == '__main__':
    GLOBAL = cache(GLOBAL)
    print(GLOBAL)
    traints = ['10/18/2016 06:00','10/18/2016 15:00']
    testts = ['10/25/2016 06:00','10/25/2016 15:00']
    trainlist = []
    testlist=[]
    for i in range(len(traints)):
        GLOBAL,traindf = train(GLOBAL,start=traints[i],freq='20Min',tollgate=1)
        df = arima_pred(GLOBAL,start=testts[i],freq='20Min',tollgate=1)
        trainlist.append(traindf)
        testlist.append(df)
    df = pd.concat(testlist)
    df.index.name = 'time_window_s'
    print(df.head())
    df.to_csv(r'final.csv')
    df = pd.concat(trainlist)
    df.to_csv(r'total.csv')
    GLOBAL = cache(GLOBAL)
    final_agg()
#    bp_train()
#    rbf_train()
#    test(freq='20Min',tollgate=1)






    '''
    输入点为当前时间t
    输出为需要预测的时间t+T
    
    所以预测8：20的流量 ，需要将时间点设置为8:00
    
    '''
