# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:13:58 2017
@author: 竹间为简
@published in: 简书
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as st
import numpy as np
import pyflux as pf
import pickle
import os

#######################################
'''help func'''  
def test_stationarity(ts):
    dftest = adfuller(ts, autolag='AIC')
    #返回的是p值
    return dftest[1]


def best_diff(ts, maxdiff = 3):
    d_set = []
    p_set = []
    for i in range(1, maxdiff):
        temp = ts.copy() #每次循环前，重置
        temp = temp.diff(i)
        temp = temp.dropna() #差分后，前几行的数据会变成nan，所以删掉
        pvalue = test_stationarity(temp)
        d_set.append(i)
        p_set.append(pvalue)
    d = dict(zip(d_set,p_set))
    if d != None:
        mindiff = min(d,key=d.get)
    return mindiff


def produce_diffed_timeseries(ts, diffn):
    if diffn != 0:
        temp = ts.diff(diffn)
    else:
        temp = ts
    temp.dropna(inplace=True) #差分之后的nan去掉
    return temp

def choose_order(ts, maxar, maxma):
    order = st.arma_order_select_ic(ts, maxar, maxma, ic=['aic', 'bic', 'hqic'])
    return order.bic_min_order


def predict_recover(ts, train, diffn):
    if diffn != 0:
        ts.iloc[0] = ts.iloc[0]+train[-diffn]
        ts = ts.cumsum()
    ts = np.exp(ts)
#    ts.dropna(inplace=True)
    print('还原完成')
    return ts

'''arima model'''
class ARIMA_V1:
    def __init__(self,name='should be the name of saved model'):
        self.name = name
        self.model = ''
        self.diffn = 0
    
    def build(self,maxar=4,maxma=4,maxdiff=6,test_size=6,save_path=''):
        self.maxar = maxar
        self.maxma = maxma
        self.maxdiff = maxdiff
        self.test_size = test_size
        self.save_path=save_path
        
        
    def train(self,df):
        '''输入必须是DataFrame型
            索引为时间
            对应第一列为需要预测的数据
            返回原序列预测序列
        '''
        data = df.dropna()
        diffn = 0
        data.loc[:,'log'] = np.log(data[data.columns[0]])
        #    test_size = int(len(data) * 0.33)
        train_size = len(data)-int(self.test_size)
        ts, test = data['log'][:train_size], data['log'][train_size:]
        if test_stationarity(ts) < 0.01:
            print(len(ts),'平稳，不需要差分')
        else:
            diffn = best_diff(ts, maxdiff = self.maxdiff)
            ts = produce_diffed_timeseries(ts, diffn)
            print('差分阶数为'+str(diffn)+'，已完成差分')
        print('开始进行ARMA拟合')
        order = choose_order(ts, self.maxar, self.maxma)
        print('模型的阶数为：'+str(order))
        _ar = order[0]
        _ma = order[1]
    #    print(ts)
        print(type(ts))
        model = pf.ARIMA(data=ts.values, ar=_ar, ma=_ma,family=pf.Normal())
        model.fit("MLE")
        test_predict = model.predict(int(self.test_size))
        mu,Y=model._model(model.latent_variables.get_z_values())
        fitted_values = model.link(mu)
        temp = np.ones((len(data)-self.test_size-len(fitted_values)))*np.mean(fitted_values)
        fitted_values = np.concatenate((temp,fitted_values))
        print(len(fitted_values),len(data))
        if self.test_size > 0:
            fitted_values = np.concatenate((fitted_values,np.array(test_predict).flatten()))
        temp = pd.Series(data=fitted_values,index=data.index)
        #re
        temp = predict_recover(temp,ts,diffn)
        
        test_predict1 = predict_recover(test_predict, ts, diffn)
        RMSE = np.sqrt(((np.array(test_predict1)-np.array(test))**2).sum()/test.size)
        print(len(test_predict),"测试集的RMSE为："+str(RMSE))
        
        self.model = model
        self.diffn = diffn
        return temp
    
    def save_model(self):
        if os.path.exists(self.save_path):
            f = open(''.join([self.save_path,self.name,'-',str(self.diffn),'.pkl']),'wb')
            pickle.dump(self.model,f)
            f.close()
        else:
            print('path does not exist')
            
    def load_model(self):
        if os.path.exists(self.save_path):
            f = open(''.join([self.save_path,self.name,'-',self.diffn,'.pkl']),'rb')
            self.model = pickle.load(f)
            f.close()
        else:
            print('path does not exist')
        
    def predict(self,load=False,step=1):
        if load:
            self.load_model()
            print(self.model)
            
        else:
            print(self.model)
        return self.model.predict(step)

