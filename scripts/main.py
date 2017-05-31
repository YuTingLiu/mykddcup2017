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
from time_series_analysis_p2 import *

def load(fdir='volume_union.csv',start='10/8/2016 06:00',freq='T',normalize = False,pat = False,periods=72,days=7):
    return gen_df(fdir=fdir,start=start,normalize = normalize,pat = pat,periods=periods,days=days)
		
def train(GLOBAL,start='10/8/2016 06:00',freq = '20Min',p=10,q=4,days=7,periods=12):
    '''
    主要的内容是:数据按照时间分割,送入ARIMA训练,得到结果1
	      整合结果1,送入BPNN训练,得到训练好的网络
         输出:每个tollgate一个模型
    '''
    df,train_seq = load(start=start,freq=freq,normalize = False,periods=periods,days=days,pat=False)

    df.loc[:,'day'] = df['time_window_s'].dt.dayofyear
    inter = df.groupby('tollgate')
    outputlist = []
    for t,tgroup in inter:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')[['volume']]
            group = group.rolling(12).mean() #add a rolling prepro
#            data_check()

            print('time seq',train_seq[0],train_seq[-1])
#            sys.exit()
            day = train_seq.day[0]
            hour = train_seq.hour[0]
            ts = group.loc[train_seq,:]
            param = ''.join(['volume',str(t),str(d),str(hour)])
            while np.any(pd.isnull(ts)):
                print(param,'train nan')
                print(ts[pd.isnull(ts)])
                sys.exit()
            print(param,'ARIMA模型,训练X天的数据,默认时间序列周期为一天,通过建立模型,保存模型')
            
            model = ARIMA_V1(name=param)
            model.build(maxar=p,maxma=q,maxdiff=6,test_size=3,save_path=arima_path)
            pred = model.train(ts)
            GLOBAL[param] = str(model.diffn)
            model.save_model()
            
            #还原
            ts.loc[:,'pred'] = pred
            plot_compare(ts['volume'],ts['pred'],0)
#            sys.exit()
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

def arima_pred(GLOBAL,start='10/18/2016 06:00',freq = '20Min',periods=6,days=7):
    '''
    '''
#    sys.exit()
    pred2hour=[]
    pred4hour=[]
    df,train_seq = load(start=start,freq=freq,normalize = False,periods=12,days=days)
    _,seq = load(start=start,freq=freq,normalize = False,periods=6,pat=False,days=days)
    seq = seq + Hour(2)
    df.loc[:,'day'] = df['time_window_s'].dt.dayofyear
    print(df.head())
    tolls = df.groupby('tollgate')
    for t,tgroup in tolls:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')
            ts = group.loc[train_seq,:]
            print('predict step ',len(ts))
            hour = train_seq.hour[0]
            param = ''.join(['volume',str(t),str(d),str(hour)])
            diffn = GLOBAL[param]
            
            model = ARIMA_V1(name=param)
            model.build(save_path=arima_path)
            model.diffn = diffn
            pred = model.predict(load=True,step=len(ts))
            
            print(len(train_seq))
            print(len(pred))
            print(len(ts))
            df = predict_recover(pred,ts['volume'],0)
#            print(df)
            temp = pd.Series(data=df['Series'].values,index=ts.index)
            ts.loc[:,'pred'] = temp
            ts.loc[:,'direction'] = d
            ts.loc[:,'tollgate'] = t
            ts.loc[:,'time_window_e'] = ts.index + Minute(20)
            ts.index.name = 'time_window_s'
#            plot_compare(ts['volume'],ts['pred'],0)
            print(ts.loc[seq,:].head(12))
#            sys.exit()
            pred4hour.append(ts)
            pred2hour.append(ts.loc[seq,:])
            
    return pd.concat(pred4hour),pd.concat(pred2hour)
    
def bp_train(fdir='volume_total.csv',start='10/18/2016 06:00',test_size=12,freq='20Min',days=7):
    _,train_seq = load(start=start,freq=freq,normalize = False,periods=12,days=days,pat=False)
    df = pd.read_csv(fdir)
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    df.loc[:,'residual'] = np.log(df['volume'])-np.log(df['pred'])
#    df.loc[:,'residual'] = df.loc[:,'residual'].rolling(12).mean().values
#    train_seq = train_seq[12:]
    hour = train_seq.hour[0]#param
#    df.to_csv(r'bp.csv',index=False)
    df = df[pd.notnull(df['pred'])]
    inter = df.groupby('tollgate')
    for t,tgroup in inter:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            if int(t)==2 and int(d)==0:
                print(dgroup.set_index('time_window_s').loc[train_seq,:])
            else:
                print('lueguo')
                continue
            batch_x = dgroup.set_index('time_window_s').loc[train_seq,:]
            while np.any(pd.isnull(batch_x)):
                print('nan found ',t,'',d)
                print(batch_x[pd.isnull(batch_x)])
                print(len(batch_x[pd.isnull(batch_x)]))
                sys.exit()
                
            x_data = np.array(batch_x[['pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute','pattern']])
#    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
#                                'temperature','rel_humidity','precipitation',
#                                'dayofweek','hour','minute','direction','pattern']])
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
            bp.modelname = ''.join(['volume',str(t),str(d),str(hour)])
            bp.training_iters = bp_train_iter
            bp.build(D,K)
            bp.fit(X,Y,Xtest,Ytest)
        #    print(train_seq)
            print('train',X.shape,Y.shape)
            print('train seq ',train_seq[0],train_seq[-1])
            
def bp_test(fdir='volume_total.csv',start='10/18/2016 06:00',freq='20Min',predict=False,test=False,days=7,periods=6): 
    
    outputlist = []
    
    _,train_seq = load(start=start,freq=freq,normalize = False,periods=periods,days=days)
    df = pd.read_csv(fdir)
        
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
#    df.loc[:,'residual'] = df['travel_time']-df['pred']
    df.loc[:,'residual'] = np.log(df['volume'])-np.log(df['pred'])
#    df.to_csv(r'bp.csv',index=False)
    hour = train_seq.hour[0]
    if predict:
        train_seq = train_seq + Hour(2)
    print('test seq ',train_seq[0],train_seq[-1])
    df = df[pd.notnull(df['pred'])]
    inter = df.groupby('tollgate')
    for t,tgroup in inter:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            batch_x = dgroup.set_index('time_window_s')
            batch_x = batch_x.loc[train_seq,:]
            while np.any(pd.isnull(batch_x)):
                print('由于nan found ',t,'',d)
                batch_x = batch_x.bfill()
                batch_x = batch_x.ffill()
                
            x_data = np.array(batch_x[['pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute','pattern']])
#    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
#                                'temperature','rel_humidity','precipitation',
#                                'dayofweek','hour','minute','direction','pattern']])
            y_data = np.array(batch_x.loc[:,'residual'])#修改为残差
            y_data = y_data.reshape((len(y_data),1))
            X=x_data
            batch_y=y_data
            #begin bp test
            bp = bp_net()
            bp.modelname = ''.join(['volume',str(t),str(d),str(hour)])
            itertemp = {'volume106':195000,'volume116':185000,'volume206':50000,'volume306':90000,
                        'volume316':50000,'volume1015':60000,'volume1115':50000,'volume2015':50000,
                        'volume3015':50000,'volume3115':50000}
            bp.training_iters = itertemp[bp.modelname]
            pred = bp.predict(X)
            print(np.sum(pred,axis=0))
            print(np.sum(batch_y,axis=0))
            #预测结束后
            print('pred length',len(pred))
            print('true length ',len(batch_y))
            pred = pd.Series(data=np.array(pred).flatten(),index = train_seq)
            
            
            pred = modification(pred,3,show=True)
            
            batch_x.loc[:,'residual'] = pred
            batch_x.loc[:,'bp'] = np.exp((batch_x['residual']+np.log(batch_x['pred'])))
            outputlist.append(batch_x)
    print('ouputlist',len(outputlist))
    df = pd.concat(outputlist)
    df.index.name = 'time_window_s'
    return df
def bp_next_train(fdir='volume_total.csv',start='10/18/2016 06:00',test_size=12,freq='20Min',days=7):
    
    outputlist = []
    
    _,train_seq = load(start=start,freq=freq,normalize = False,periods=6,days=days,pat=False)
    df = pd.read_csv(fdir)
    df.loc[:,'time_window_s']=pd.to_datetime(df.loc[:,'time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    df.loc[:,'residual'] = np.log(df['volume'])-np.log(df['pred'])
    
    hour = train_seq.hour[0]#param
#    df.to_csv(r'bp.csv',index=False)
    df = df[pd.notnull(df['pred'])]
    inter = df.groupby('tollgate')
    for t,tgroup in inter:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            batch_x = dgroup.set_index('time_window_s').loc[train_seq,:]
            while np.any(pd.isnull(batch_x)):
                print('nan found ',t,'',d)
                print(batch_x[pd.isnull(batch_x)])
                print(len(batch_x[pd.isnull(batch_x)]))
                batch_x = batch_x.bfill()
                batch_x = batch_x.ffill()
                
            x_data = np.array(batch_x[['pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute','pattern']])
#    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
#                                'temperature','rel_humidity','precipitation',
#                                'dayofweek','hour','minute','direction','pattern']])
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
            bp.modelname = ''.join(['volume',str(t),str(d),str(hour)])
            bp.training_iters = bp_train_iter
            bp.train1(X,Y,Xtest,Ytest)
        #    print(train_seq)
            print('train',X.shape,Y.shape)
            print('train seq ',train_seq[0],train_seq[-1])   
            
            pred = bp.predict(x_data)
            print(np.sum(pred,axis=0))
            print(np.sum(y_data,axis=0))
            #预测结束后
            print('pred length',len(pred))
            print('true length ',len(y_data))
            pred = pd.Series(data=np.array(pred).flatten(),index = train_seq)
            
            batch_x.loc[:,'residual'] = pred
            batch_x.loc[:,'bp'] = np.exp((batch_x['residual']+np.log(batch_x['pred'])))
            outputlist.append(batch_x)
    print('ouputlist',len(outputlist))
    df = pd.concat(outputlist)
    df.index.name = 'time_window_s'
    return df
def rbf_train(tollgate=1):
    _,train_seq = load(freq,normalize = False)
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

def evalute(df,start='10/18/2016 06:00',freq = '20Min',predict=False,days=7):
    '''
    主要的内容是:数据按照时间分割,送入ARIMA训练,得到结果1
	      整合结果1,送入BPNN训练,得到训练好的网络
         输出:每个tollgate一个模型
    '''
    
    bp_result = []
    arima_result=[]
    
    origin,train_seq = load(start=start,freq=freq,normalize = False,periods=6,days=days)
    if predict:
        train_seq = train_seq + Hour(2)
    #生成这个时间段上的前1天序列
    plot_seq = pd.date_range(start=train_seq[0]-Day(1)-Hour(2),freq=freq,periods=12)
    plot_seq = plot_seq + train_seq
    inter = df.groupby('tollgate')
    for t,tgroup in inter:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            #计算MAPE
            import MAPE
            group = dgroup.set_index('time_window_s')[['volume','bp','pred']].loc[train_seq,:]
            plot_group = origin[(origin['tollgate']==t) & (origin['direction']==d)]
            plot_group = plot_group.set_index('time_window_s')[['volume']].loc[plot_seq,:]
            plot_group.loc[:,'bp'] = group['bp']
            plot_group.loc[:,'pred'] = group['pred']
            bp_result.append(MAPE.mape(group))
            #plot
#            print(plot_group)
#            sys.exit()
            plot_compare(plot_group['volume'],plot_group['bp'],0)
            plot_compare(plot_group['volume'],plot_group['pred'],0)
            
            
            group = dgroup.set_index('time_window_s')[['volume','pred']].loc[train_seq,:]
            arima_result.append(MAPE.mape(group))
            
            
    result = np.sum(bp_result)/len(bp_result)
    print('final',result)
    result = np.sum(arima_result)/len(arima_result)
    print('arima',result)
    return result

def bp_modification(df,start='10/18/2016 06:00',freq = '20Min',predict=False,days=7,method=2):
    '''
     数据修正
    '''
    modifyed = []
    origin,train_seq = load(start=start,freq=freq,normalize = False,periods=6,days=days)
    if predict:
        train_seq = train_seq + Hour(2)
    #生成这个时间段上的前1天序列
    plot_seq = pd.date_range(start=train_seq[0]-Day(1)-Hour(2),freq=freq,periods=12)
    plot_seq = plot_seq + train_seq
    inter = df.groupby('tollgate')
    for t,tgroup in inter:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')[['volume','bp','pred']].loc[train_seq,:]
            
            #这里可以添加一个权值处理,即线性回归
            print(group)
            group.loc[:,'bp'] = 0.5*group.loc[:,'bp'] + 0.5*group.loc[:,'pred']
            print(group['bp'])
#            sys.exit()
#            if t=='B' and d == 1:
#                print(group)
            ts = modification(group['bp'],method=method)
            if ts.sum() < 10:
                print(group)
                sys.exit()
            group.loc[:,'bp'] = ts
            group.loc[:,'tollgate'] = t
            group.loc[:,'direction'] = d
            modifyed.append(group)
    return pd.concat(modifyed)
            
def cache(config,fdir='volume_global_config.txt'):
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
    df.loc[:,'volume'] = df.loc[:,'pred']
    df = df[['tollgate','direction','time_window','volume']]
    df.to_csv(r'../volume_submit_time.csv',index=False)
    
def bp_final_agg(fdir=r'bp_test_result.csv'):
    df = load_volume(fdir)
    df.loc[:,'time_window_e'] = df.time_window_s + Minute(20)
    df.to_csv(fdir,index = False)
    df =  pd.read_csv(fdir)
    df.loc[:,'time_window'] = '[' + df['time_window_s'] +','+ df['time_window_e'] +')'
    df.loc[:,'volume'] = df.loc[:,'bp']
    df = df[['tollgate','time_window','direction','volume']]
    df.columns = ['tollgate_id','time_window','direction','volume']
    df.to_csv(r'../bp_submit_volume.csv',index=False)
    
GLOBAL = {}
train_f = r'../train/'
arima_path = r'../arima/'
this = 'time'
bp_train_iter = 50000
if __name__ == '__main__':
    GLOBAL = cache(GLOBAL)
#    print(GLOBAL)
    traints = ['9/19/2016 06:00','9/19/2016 15:00']
    middle = ['10/18/2016 06:00','10/18/2016 15:00']
    testts = ['10/25/2016 06:00','10/25/2016 15:00']
    trainlist = []
    testlist=[]
    bp_list = []
#####    #arima train and test
#    for i in range(len(traints)):
#        GLOBAL,traindf = train(GLOBAL,start=traints[i],freq='20Min',days=36,periods=12)
#        pred4hour,pred2hour = arima_pred(GLOBAL,start=testts[i],freq='20Min',periods=12,days=7)
#        trainlist.append(traindf)
#        testlist.append(pred2hour)
#        bp_list.append(pred4hour)
#    df = pd.concat(testlist)
#    df.index.name = 'time_window_s'
#    df.to_csv(r'volume_final.csv')#预测集结果
#    df = pd.concat(bp_list)
#    df.to_csv(r'volume_final_bp.csv')#预测集4小时结果
#    df = pd.concat(trainlist)
#    df.to_csv(r'volume_total.csv')#训练集预测结果
#    GLOBAL = cache(GLOBAL)
#    
#    for i in range(len(traints)):
#        bp_train(fdir=r'volume_total.csv',start=traints[i],days=36)
#        
#    #训练集BP修正
#    dflist = []
#    for i in range(len(middle)):
#        df = bp_test(fdir=r'volume_total.csv',start=middle[i],predict=False,days=7)
#        dflist.append(df)
#    df = pd.concat(dflist)
#    df.to_csv(r'bp_result.csv')
#        
##    使用model继续训练
##    print('将模型中没有的时间 段继续训练，可以看到收敛程度')
##    dflist = []
##    for i in range(len(testts)):
##        df = bp_next_train(fdir=r'bp_result.csv',start=testts[i],days=7)
##        dflist.append(df)
##    df = pd.concat(dflist)
##    df.to_csv(r'bp_result.csv')
#
#    modifyed = []
#    df = load_volume(fdir=r'bp_result.csv')
#    df = df[pd.notnull(df['bp'])]
#    for i in range(len(middle)):
#        df1 = bp_modification(df,start=middle[i],predict=False,days=7,method=3)   
#        modifyed.append(df1)
#    df = pd.concat(modifyed)
#    df.index.name = 'time_window_s'
#    df.to_csv(r'bp_modifyed.csv')
##    evalute
#    df = load_volume(fdir=r'bp_modifyed.csv')
#    df = df[pd.notnull(df['bp'])]
#    for i in range(len(middle)):
#        evalute(df,start=middle[i],predict=False,days=7)
###
###############################################################
##    '''
##        在测试集上，使用BP预训练模型测试
##        测试集BP修正
##    '''
##    
##      
    predict = True  
    dflist = []
    for i in range(len(testts)):
        df = bp_test(fdir='volume_final_bp.csv',start=testts[i],test=True,predict=predict,days=7)
        dflist.append(df)
    df = pd.concat(dflist)
    df.to_csv(r'bp_test_result.csv')
    modifyed = []
    df = load_volume(fdir=r'bp_test_result.csv')
    df = df[pd.notnull(df['bp'])]
    for i in range(len(testts)):
        df1 = bp_modification(df,start=testts[i],predict=predict,days=7,method=3)   
        modifyed.append(df1)
    df = pd.concat(modifyed)
    df.index.name = 'time_window_s'
    df.to_csv(r'bp_modifyed.csv')
    #evalute
    df = load_volume(fdir=r'bp_modifyed.csv')
    df = df[pd.notnull(df['bp'])]
    for i in range(len(testts)):
        evalute(df,start=testts[i],predict=predict,days=7)
    bp_final_agg(fdir=r'bp_modifyed.csv')
    '''
    输入点为当前时间t
    输出为需要预测的时间t+T
    
    所以预测8：20的流量 ，需要将时间点设置为8:00
    
    '''
