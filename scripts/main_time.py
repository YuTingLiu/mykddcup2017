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
        df = load_volume(fdir='time_train_union1.csv')
        train_seq = produce_seq(start=start,periods=periods,freq='20Min',days = 7)
        if len(train_seq) != 72*20:
            print('train_seq len ',len(train_seq))
#            sys.exit()
        if test is True:
            df = load_test(fdir='time_test_union1.csv')
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
    
		
def train(GLOBAL,start='10/8/2016 06:00',freq = '20Min',tollgate = 1,p=4,q=2):
    '''
    主要的内容是:数据按照时间分割,送入ARIMA训练,得到结果1
	      整合结果1,送入BPNN训练,得到训练好的网络
         输出:每个tollgate一个模型
    '''
    df,train_seq,test_seq = gen_df(start=start,freq=freq,normalize = False,test=False,periods=12)

    df.loc[:,'day'] = df['time_window_s'].dt.dayofyear
    inter = df.groupby('intersection_id')
    outputlist = []
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        
        train_seq1 = train_seq.copy()
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')[['travel_time']]
#            data_check()
            print('ARIMA模型,训练X天的数据,默认时间序列周期为一天,通过建立模型,保存模型')
            print('time seq',train_seq1[0],train_seq1[-1])
#            sys.exit()
            for i in range(1):
                day = train_seq1.day[0]
                hour = train_seq1.hour[0]
                ts = group.loc[train_seq1,:]
                from time_series_analysis_p2 import run_aram
                model,MRSE,pred,diffn = run_aram(ts,p,q)
                param = ''.join([str(t),str(d),str(hour)])
                GLOBAL[param] = str(diffn)
                f = open(''.join([r'../arima','//',str(t),str(d),str(hour),'-',str(diffn)]),'wb')
                pickle.dump(model,f)
                f.close()
                #还原
                ts.loc[:,'pred'] = pred
                plot_compare(ts['travel_time'],ts['pred'],0)
                ts.loc[:,'tollgate_id'] = d
                ts.loc[:,'intersection_id'] = t
                ts.index.name = 'time_window_s'
                outputlist.append(ts)
#        sys.exit(0)
    output = pd.concat(outputlist)
    output.to_csv('output1.csv')
    df = df.set_index(['intersection_id','tollgate_id','time_window_s'])
    df = df.join(output.reset_index().set_index(['intersection_id','tollgate_id','time_window_s'])[['pred']])
    df.to_csv('total_arima_bp_log1.csv',index = True)
    return GLOBAL,df
#        sys.exit(0)

def arima_pred(GLOBAL,start='10/18/2016 06:00',freq = '20Min',tollgate = 1):
    '''
    '''
    print(GLOBAL)
#    sys.exit()
    pred2hour=[]
    pred4hour=[]
    df,train_seq,test_seq = gen_df(start=start,freq=freq,normalize = False,test=True,periods=12,pat=False)
    _,_,seq = gen_df(start=start,freq=freq,normalize = False,test=True,periods=6,pat=False)
    seq = seq + Hour(2)
    df.loc[:,'day'] = df['time_window_s'].dt.dayofyear
    print(df.head())
    tolls = df.groupby('intersection_id')
    for t,tgroup in tolls:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')
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
            df = predict_recover(pred,ts['travel_time'],0)
#            print(df)
            temp = pd.Series(data=df['Series'].values,index=ts.index)
            ts.loc[:,'pred'] = temp
            ts.loc[:,'intersection_id'] = t
            ts.loc[:,'tollgate_id'] = d
            ts.loc[:,'time_window_e'] = ts.index + Minute(20)
            ts.index.name = 'time_window_s'
#            plot_compare(ts['volume'],ts['pred'],0)
            print(ts.loc[seq,:].head(12))
#            sys.exit()
            pred4hour.append(ts)
            pred2hour.append(ts.loc[seq,:])
            
    return pd.concat(pred4hour),pd.concat(pred2hour)
               
def bp_train(start='10/18/2016 06:00',test_size=12,freq='20Min'):
    _,train_seq,test_seq = gen_df(start=start,freq=freq,normalize = False,test=False,periods=12)
    df = pd.read_csv(r'time_total.csv')
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    df.loc[:,'residual'] = df['travel_time']-df['pred']
    df.loc[:,'residual'] = np.log(df['travel_time'])-np.log(df['pred'])

    hour = train_seq.hour[0]    #param
#    df.to_csv(r'bp.csv',index=False)
    df = df[pd.notnull(df['pred'])]
    inter = df.groupby('intersection_id')
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            batch_x = dgroup.set_index('time_window_s').loc[train_seq,:]
            x_data = np.array(batch_x[['pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute']])
#    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
#                                'temperature','rel_humidity','precipitation',
#                                'dayofweek','hour','minute','direction','pattern']])
            if np.sum(np.isnan(x_data)) > 0:
                print('nan found ')
                sys.exit(0)
            y_data = np.array(batch_x.loc[:,'residual'])#修改为残差
            y_data = y_data.reshape((len(y_data),1))
    
            train_size = len(x_data)-int(test_size)
            X=x_data[:train_size]
            Y=y_data[:train_size]
            Xtest=x_data[train_size:]
            Ytest=y_data[train_size:]
            
            #begin bp train
            bp = bp_net()
            N,D = x_data.shape
            K = len(y_data[0])
            bp.middle = 60
            bp.modelname = ''.join([str(t),str(d),str(hour)])
            bp.training_iters = bp_train_iter
            bp.build(D,K)
            bp.fit(X,Y,Xtest,Ytest)
        #    print(train_seq)
            print('train',X.shape,Y.shape)
            print('train seq ',train_seq[0],train_seq[-1])
            
def bp_test(start='10/18/2016 06:00',freq='20Min',test=False): 
    
    outputlist = []
    
    _,train_seq,test_seq = gen_df(start=start,freq=freq,normalize = False,test=test,periods=6)
    if test:
        train_seq = test_seq.copy()
        df = pd.read_csv(r'time_final_bp.csv')
#        print(df)
#        sys.exit()
    else:
        df = pd.read_csv(r'time_total.csv')
        
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
#    df.loc[:,'residual'] = df['travel_time']-df['pred']
    df.loc[:,'residual'] = np.log(df['travel_time'])-np.log(df['pred'])
#    df.to_csv(r'bp.csv',index=False)
    hour = train_seq.hour[0]
#    train_seq = train_seq + Hour(2)
    print('test seq ',train_seq[0],train_seq[-1])
    df = df[pd.notnull(df['pred'])]
    inter = df.groupby('intersection_id')
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            batch_x = dgroup.set_index('time_window_s')
            batch_x = batch_x.loc[train_seq,:]
            if np.any(pd.isnull(batch_x)):
                print('nan found ',t,'',d)
                batch_x = batch_x.bfill()
                print(batch_x)
                
            x_data = np.array(batch_x[['pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute']])
#    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
#                                'temperature','rel_humidity','precipitation',
#                                'dayofweek','hour','minute','direction','pattern']])
            y_data = np.array(batch_x.loc[:,'residual'])#修改为残差
            y_data = y_data.reshape((len(y_data),1))
            X=x_data
            batch_y=y_data
            #begin bp test
            bp = bp_net()
            bp.modelname = ''.join([str(t),str(d),str(hour)])
            bp.training_iters = bp_train_iter
            pred = bp.predict(X)
            print(np.sum(pred,axis=0))
            print(np.sum(batch_y,axis=0))
            #预测结束后
            print('pred length',len(pred))
            print('true length ',len(batch_y))
            pred = pd.Series(data=np.array(pred).flatten(),index = train_seq)
            
            batch_x.loc[:,'residual'] = pred
            batch_x.loc[:,'bp'] = np.exp((batch_x['residual']+np.log(batch_x['pred'])))
            outputlist.append(batch_x)
    print('ouputlist',len(outputlist))
    df = pd.concat(outputlist)
    df.index.name = 'time_window_s'
    return df
         
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

def evalute(df,start='10/18/2016 06:00',freq = '20Min',tollgate = 1):
    '''
    主要的内容是:数据按照时间分割,送入ARIMA训练,得到结果1
	      整合结果1,送入BPNN训练,得到训练好的网络
         输出:每个tollgate一个模型
    '''
    _,train_seq,test_seq = gen_df(start=start,freq=freq,normalize = False,test=False,periods=6)
#    train_seq = train_seq + Hour(2)
    inter = df.groupby('intersection_id')
    result = []
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')[['travel_time','bp']].loc[train_seq,:]
            
            #计算MAPE
            import MAPE
            result.append(MAPE.mape(group))
    result = np.sum(result)/len(result)
    print('final',result)
    return result
            
            
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
def final_agg(fdir=r'time_final.csv'):
    df = pd.read_csv(fdir)
    df.loc[:,'time_window'] = '[' + df['time_window_s'] +','+ df['time_window_e'] +')'
    df.loc[:,'travel_time'] = df.loc[:,'pred']
    df = df[['intersection_id','tollgate_id','time_window','travel_time']]
    df.to_csv(r'../submit_time.csv',index=False)
    
GLOBAL = {}
train_f = '../train'
this = 'time'
bp_train_iter = 50000
if __name__ == '__main__':
    GLOBAL = cache(GLOBAL)
    print(GLOBAL)
    traints = ['10/18/2016 06:00','10/18/2016 15:00']
    testts = ['10/25/2016 06:00','10/25/2016 15:00']
    trainlist = []
    testlist=[]
    bp_list = []
#    #arima train and test
#    for i in range(len(traints)):
#        GLOBAL,traindf = train(GLOBAL,start=traints[i],freq='20Min',tollgate=1)
#        pred4hour,pred2hour = arima_pred(GLOBAL,start=testts[i],freq='20Min',tollgate=1)
#        trainlist.append(traindf)
#        testlist.append(pred2hour)
#        bp_list.append(pred4hour)
#    df = pd.concat(testlist)
#    df.index.name = 'time_window_s'
#    print(df.head())
#    df.to_csv(r'time_final.csv')
#    df = pd.concat(trainlist)
#    df.to_csv(r'time_total.csv')
#    df = pd.concat(bp_list)
#    df.to_csv(r'time_final_bp.csv')
#    GLOBAL = cache(GLOBAL)
    final_agg()
#    
#    
#    for i in range(len(traints)):
#        bp_train(start=traints[i])
#        
#    dflist = []
#    for i in range(len(traints)):
#        df = bp_test(start=traints[i])
#        dflist.append(df)
#    df = pd.concat(dflist)
#    df.to_csv(r'bp_result.csv')
#        
#
#
#
#    #evalute
#    df = load_volume(fdir=r'bp_result.csv')
#    df = df[pd.notnull(df['bp'])]
#    for i in range(len(traints)):
#        evalute(df,start=traints[i])

#############################################################
#    for i in range(len(traints)):
#        bp_train(start=traints[i])
        
    dflist = []
    for i in range(len(testts)):
        df = bp_test(start=testts[i],test=True)
        dflist.append(df)
    df = pd.concat(dflist)
    df.to_csv(r'bp_test_result.csv')
        
#    rbf_train()
#    test(freq='20Min',tollgate=1)


    #evalute
    df = load_volume(fdir=r'bp_test_result.csv')
    df = df[pd.notnull(df['bp'])]
    for i in range(len(testts)):
        evalute(df,start=testts[i])

    '''
    输入点为当前时间t
    输出为需要预测的时间t+T
    
    所以预测8：20的流量 ，需要将时间点设置为8:00
    
    '''
