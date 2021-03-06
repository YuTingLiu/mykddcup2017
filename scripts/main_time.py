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
from sklearn.decomposition import PCA
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
from gbdt import *

train_f = r'../train/'
arima_path = r'../arima/'

def bp_pca(_x):
    pca = PCA(n_components=10)
    pca.fit(_x)
    print(pca.explained_variance_ratio_)
    return pca.transform(_x)

def load(fdir='time_union.csv',start='10/8/2016 06:00',freq='T',normalize = False,pat = False,periods=72,days=7):
    return gen_df(fdir=fdir,start=start,normalize = normalize,pat = pat,periods=periods,days=days)
		
def train(GLOBAL,start='10/8/2016 06:00',freq = '20Min',p=4,q=4,days=7,periods=12,this=''):
    '''
    主要的内容是:数据按照时间分割,送入ARIMA训练,得到结果1
	      整合结果1,送入BPNN训练,得到训练好的网络
         输出:每个tollgate一个模型
    '''
    df,train_seq = load(start=start,freq=freq,normalize = False,periods=periods,days=days,pat=True)

    df.loc[:,'day'] = df['time_window_s'].dt.dayofyear
    inter = df.groupby('intersection_id')
    outputlist = []
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')[['travel_time']]
#            data_check()

            print('time seq',train_seq[0],train_seq[-1])
#            sys.exit()
            day = train_seq.day[0]
            hour = train_seq.hour[0]
            ts = group.loc[train_seq,:]
            param = ''.join([this,str(t),str(d),str(hour)])
            while np.any(pd.isnull(ts)):
                print(param,'train nan')
                print(ts[pd.isnull(ts)])
                sys.exit()
            print(param,'ARIMA模型,训练X天的数据,默认时间序列周期为一天,通过建立模型,保存模型')
            
            ts['travel_time'] = modification(ts['travel_time'],method=3,show=False)
#            sys.exit()
#            
            
            ####以下为各个模型训练部分
            
            
            model = ARIMA_V1(name=param)
            model.build(maxar=p,maxma=q,maxdiff=6,test_size=6,save_path=arima_path)
            pred = model.train(ts)
            GLOBAL[param] = str(model.diffn)
            model.save_model()
            
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

def arima_pred(GLOBAL,start='10/18/2016 06:00',freq = '20Min',periods=6,days=7,this=''):
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
    tolls = df.groupby('intersection_id')
    for t,tgroup in tolls:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')
            ts = group.loc[train_seq,:]
            hour = train_seq.hour[0]
            param = ''.join([this,str(t),str(d),str(hour)])
            diffn = GLOBAL[param]
            
            model = ARIMA_V1(name=param)
            model.build(save_path=arima_path)
            model.diffn = diffn
            pred = model.predict(load=True,step=len(ts))
            
            print(len(train_seq))
            print(len(pred))
            print(len(ts))
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
    
def bp_tt(start='10/18/2016 06:00',test_size=12,freq='20Min',periods=12,
          days=7,predict=False,test=False,train=False,this='',bp_train_iter=1000):
    
    outputlist = []
    
    _,train_seq = load(start=start,freq=freq,normalize = False,periods=periods,days=days,pat=True)
    if test:
        df = pd.read_csv(r'time_final_bp.csv')
    else:
        df = pd.read_csv(r'time_total.csv')
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
#    df.loc[:,'residual'] = np.log(df['travel_time'])-np.log(df['pred'])

    hour = train_seq.hour[0]#param
    if predict:
        train_seq = train_seq + Hour(2)
#    df.to_csv(r'bp.csv',index=False)
    df = df[pd.notnull(df['pred'])]
    inter = df.groupby('intersection_id')
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            batch_x = dgroup.set_index('time_window_s').loc[train_seq,:]
#            print(batch_x['travel_time'])
            batch_x['travel_time'] = modification(batch_x['travel_time'],method=3,show=False)
            batch_x.loc[:,'residual'] = np.log(batch_x['travel_time'])-np.log(batch_x['pred'])
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
    
            train_size = len(x_data)-int(test_size)
            X=x_data[:train_size]
            Y=y_data[:train_size]
            Xtest=x_data[train_size:]
            Ytest=y_data[train_size:]
            
            if train:
                #begin bp train
                bp = bp_net()
                N,D = x_data.shape
                K = len(y_data[0])
                bp.middle = 60
                bp.modelname = ''.join([this,str(t),str(d),str(hour)])
                bp.training_iters = bp_train_iter
                bp.build(D,K)
                bp.fit(X,Y,Xtest,Ytest)
            #    print(train_seq)
                print('train',X.shape,Y.shape)
                print('train seq ',train_seq[0],train_seq[-1])
            else:
                #begin bp test
                bp = bp_net()
                bp.modelname = ''.join([this,str(t),str(d),str(hour)])
                bp.training_iters = bp_train_iter
                pred = bp.predict(x_data)
                print(np.sum(pred,axis=0))
                print(np.sum(y_data,axis=0))
                #预测结束后
                print('pred length',len(pred))
                print('true length ',len(y_data))
                pred = pd.Series(data=np.array(pred).flatten(),index = train_seq)
                
#                pred = filter(pred)
                pred = modification(pred,3)
                
                
                plot_compare(batch_x['residual'],pred,0)
#                sys.exit()
                batch_x.loc[:,'residual'] = pred
                batch_x.loc[:,'bp'] = np.exp((batch_x['residual']+np.log(batch_x['pred'])))
                outputlist.append(batch_x)
    if len(outputlist)>0:
        print('ouputlist',len(outputlist))
        df = pd.concat(outputlist)
        df.index.name = 'time_window_s'
        return df
            
def gbdt_tt(start='10/18/2016 06:00',freq='20Min',days=7,periods=12,train=False,test=False,predict=False,this=''): 
    
    outputlist=[]
    
    _,train_seq = load(start=start,freq=freq,normalize = False,periods=periods,days=days,pat=True)
    if test:
        df = pd.read_csv(r'time_final_bp.csv')
    else:
        df = pd.read_csv(r'time_total.csv')
        
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    df.loc[:,'residual'] = df['travel_time']-df['pred']
    df.loc[:,'residual'] = np.log(df['travel_time'])-np.log(df['pred'])

    hour = train_seq.hour[0]#param
    if predict:
        train_seq = train_seq + Hour(2)
#    df.to_csv(r'bp.csv',index=False)
    df = df[pd.notnull(df['pred'])]
    inter = df.groupby('intersection_id')
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            batch_x = dgroup.set_index('time_window_s').loc[train_seq,:]
            while np.any(pd.isnull(batch_x)):
                print('nan found ',t,'',d)
                print(batch_x[pd.isnull(batch_x)])
                batch_x = batch_x.bfill()
                batch_x = batch_x.ffill()
                
            batch_x['wind_direction'] = batch_x['wind_direction'].astype('category')
            batch_x['dayofweek'] = batch_x['dayofweek'].astype('category')
            batch_x['hour'] = batch_x['hour'].astype('category')
            batch_x['minute'] = batch_x['minute'].astype('category')
            batch_x['pattern'] = batch_x['pattern'].astype('category')   
            
            feature = ['pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute','pattern']
            x_data = np.array(batch_x[feature])
    
#    x_data = np.array(batch_x[['pressure','sea_pressure','wind_direction','wind_speed',
#                                'temperature','rel_humidity','precipitation',
#                                'dayofweek','hour','minute','direction','pattern']])
            train_size = len(x_data)-12            
            y_data = np.log(np.array(batch_x.loc[:,'travel_time']))#修改为残差
            y_data = y_data.reshape((len(y_data),1))
            
            if train:
                gbdt = kdd_gbdt()
                gbdt.save_name = ''.join([this,str(t),str(d),str(hour)])
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
                batch_x.loc[:,'gbdt'] = np.exp(pred)
                outputlist.append(batch_x[['intersection_id','tollgate_id','travel_time','gbdt']])  
    if len(outputlist)>0:
        print('ouputlist',len(outputlist))
        df = pd.concat(outputlist)
        df.index.name = 'time_window_s'
        return df

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
    inter = df.groupby('intersection_id')
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            #计算MAPE
            import MAPE
            group = dgroup.set_index('time_window_s')[['travel_time','bp']].loc[train_seq,:]
            plot_group = origin[(origin['intersection_id']==t) & (origin['tollgate_id']==d)]
            plot_group = plot_group.set_index('time_window_s')[['travel_time']].loc[plot_seq,:]
            plot_group.loc[:,'bp'] = group['bp']
            bp_result.append(MAPE.mape(group))
            #plot
#            print(plot_group)
#            sys.exit()
            plot_compare(plot_group['travel_time'],plot_group['bp'],0)
            
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
    inter = df.groupby('intersection_id')
    for t,tgroup in inter:
        ds = tgroup.groupby('tollgate_id')
        for d,dgroup in ds:
            group = dgroup.set_index('time_window_s')[['travel_time','bp']].loc[train_seq,:]
            
            #这里可以添加一个权值处理,即线性回归
            print(group)
#            group.loc[:,'bp'] = 0.4*group.loc[:,'bp'] + 0.6*group.loc[:,'pred']
            print(group['bp'])
#            sys.exit()
#            if t=='B' and d == 1:
#                print(group)
            ts = modification(group['bp'],method=method)
            if ts.sum() < 10:
                print(group)
                sys.exit()
            group.loc[:,'bp'] = ts
            group.loc[:,'intersection_id'] = t
            group.loc[:,'tollgate_id'] = d
            modifyed.append(group)
    return pd.concat(modifyed)
            
def linear_reg(fdir = r'bp_test_result.csv',feature = ['pred','bp','gbdt'],predict=False):
    df = load_volume(fdir)
    print('线性回归 travel_time pred bp 这三列')
    feature = feature
    X = df[feature]
    y = df['travel_time']
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print(linreg.intercept_)
    print(linreg.coef_)
    print(zip(feature, linreg.coef_))
    y_pred = linreg.predict(X_test)
    from sklearn import metrics
    print( "MAE:",metrics.mean_absolute_error(y_test, y_pred))
    param = dict(zip(feature, linreg.coef_))
    return df,param

            
def cache(config,fdir='global_config.txt',this=''):
    if len(config)>0:
        f = open(''.join([this,fdir]),'w')
        json.dump(config,f)
        f.close()
        return config
    else:
        f = open(''.join([this,fdir]),'r')
        config = json.load(f)
        f.close()
        return config
def final_agg(fdir=r'time_final.csv'):
    df = pd.read_csv(fdir)
    df.loc[:,'time_window'] = '[' + df['time_window_s'] +','+ df['time_window_e'] +')'
    df.loc[:,'travel_time'] = df.loc[:,'pred']
    df = df[['intersection_id','tollgate_id','time_window','travel_time']]
    df.columns = ['intersection_id','tollgate_id','time_window','avg_travel_time']
    if len(df) != 504:
        print(len(df))
        df.to_csv('temp.csv')
        sys.exit()
    df.to_csv(r'..result/arima_time.csv',index=False)
    
def bp_final_agg(fdir=r'bp_test_result.csv',this=''):
    df = load_volume(fdir)
    df.loc[:,'time_window_e'] = df.time_window_s + Minute(20)
    df.to_csv(fdir,index = False)
    df =  pd.read_csv(fdir)
    df.loc[:,'time_window'] = '[' + df['time_window_s'] +','+ df['time_window_e'] +')'
    df.loc[:,'travel_time'] = df.loc[:,'bp']
    df = df[['intersection_id','tollgate_id','time_window','travel_time']]
    df.columns = ['intersection_id','tollgate_id','time_window','avg_travel_time']
    df.to_csv(''.join([r'..result/',this,'bp_submit_time.csv']),index=False)
    

#    
#GLOBAL = {}
#train_f = r'../train/'
#arima_path = r'../arima/'
#this = 'time'
#bp_train_iter = 50000
#if __name__ == '__main__':
#    GLOBAL = cache(GLOBAL)
#    print(GLOBAL)
#    traints = ['10/8/2016 06:00','10/8/2016 15:00']
#    testts = ['10/25/2016 06:00','10/25/2016 15:00']
#    trainlist = []
#    testlist=[]
#    bp_list = []
#####    #arima train and test
#    for i in range(len(traints)):
#        GLOBAL,traindf = train(GLOBAL,start=traints[i],freq='20Min',days=17,periods=12)
#        pred4hour,pred2hour = arima_pred(GLOBAL,start=testts[i],freq='20Min',periods=12,days=7)
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
#    GLOBAL = cache(GLOBAL)
#    final_agg()
#    
##    for i in range(len(traints)):
##        bp_tt(start=traints[i],days=17,train=True)
###        
##    #GBDT训练
##    for i in range(len(traints)):
##        gbdt_tt(start=traints[i],days=17)
#        
#    #####训练集BP修正
##    dflist = []
##    for i in range(len(traints)):
##        df = bp_tt(start=traints[i],predict=False,days=17)
##        dflist.append(df)
##    df = pd.concat(dflist)
##    df.to_csv(r'bp_result.csv')
##    
##    ###GBDT 测试
###    dflist = []
###    for i in range(len(traints)):
###        df = bp_test(start=traints[i],predict=False,days=17)
###        dflist.append(df)
###    df = pd.concat(dflist)
###    df.to_csv(r'bp_result.csv')
##        
##    ###使用model继续训练
##    
##    ###evalute
##    df = load_volume(fdir=r'bp_result.csv')
##    df = df[pd.notnull(df['bp'])]
##    for i in range(len(traints)):
##        evalute(df,start=traints[i],predict=False,days=17)
##
###############################################################
##    '''
##        在测试集上，使用BP预训练模型测试
##        测试集BP修正
##    '''
#    predict = True
#    test = True
#    day = 7
#    reg = False
#    if test:
#        ll = testts
#    else:
#        ll = traints
#    dflist = []
#    for i in range(len(ll)):
#        df = bp_tt(start=ll[i],test=test,predict=predict,days=day)
#        dflist.append(df)
#    df = pd.concat(dflist)
#    df.to_csv(r'bp_test_result.csv')
#    
#    ###GBDT 测试
#    dflist = []
#    for i in range(len(ll)):
#        gbdt = gbdt_tt(start=ll[i],train=False,test=test,predict=predict,days=day)
#        dflist.append(gbdt)
#    print(len(dflist))
#    gbdt = pd.concat(dflist)
#    gbdt.to_csv(r'gbdt_test_result.csv')
#    
#    ##线性回归
#    df = df.reset_index().set_index(['intersection_id','tollgate_id','time_window_s'])
#    gbdt = gbdt.reset_index().set_index(['intersection_id','tollgate_id','time_window_s'])
#    df = df.join(gbdt[['gbdt']])
#    df.to_csv('bp_gbdt.csv')
#    if reg:
#        feature = ['pred','bp','gbdt']
#        paramlist = linear_reg(fdir = r'bp_gbdt.csv',feature =feature,predict=predict)
#        print(paramlist)
#        print(paramlist['pred'])
#        print(paramlist['bp'])
#        print(paramlist['gbdt'])
#        print(df.dtypes)
#        df['bp']=paramlist['pred']*df.loc[:,'pred'] + paramlist['bp']*df.loc[:,'bp'] +paramlist['gbdt']*df.loc[:,'gbdt']
#        GLOBAL['linear'] = paramlist
#        GLOBAL = cache(GLOBAL)
#    else:
#        paramlist = GLOBAL['linear']
#        print(paramlist)
#        df['bp']=paramlist['pred']*df.loc[:,'pred'] + paramlist['bp']*df.loc[:,'bp'] +paramlist['gbdt']*df.loc[:,'gbdt']
#    df.to_csv(r'after_linear.csv')
#    
#    modifyed = []
#    df = load_volume(fdir=r'after_linear.csv')
#    df = df[pd.notnull(df['bp'])]
#    for i in range(len(ll)):
#        df1 = bp_modification(df,start=ll[i],predict=predict,days=day,method=3)   
#        modifyed.append(df1)
#    df = pd.concat(modifyed)
#    df.index.name = 'time_window_s'
#    df.to_csv(r'bp_modifyed.csv')
##    
##    
#    #evalute
#    df = load_volume(fdir=r'bp_modifyed.csv')
#    df = df[pd.notnull(df['bp'])]
#    for i in range(len(testts)):
#        evalute(df,start=testts[i],predict=predict,days=7)
##        
##        
#    bp_final_agg(fdir=r'bp_modifyed.csv')
##    
#    
#        
#        
#    '''
#    输入点为当前时间t
#    输出为需要预测的时间t+T
#    
#    所以预测8：20的流量 ，需要将时间点设置为8:00
#    
#    '''
