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


def run_aram(df, maxar, maxma, test_size = 14):
    data = df.dropna()
    data = np.log(data[data.columns[0]])
    #    test_size = int(len(data) * 0.33)
    train_size = len(data)-int(test_size)
    train, test = data[:train_size], data[train_size:]
    if test_stationarity(train[train.columns[1]]) < 0.01:
        print('平稳，不需要差分')
    else:
        diffn = best_diff(train, maxdiff = 8)
        train = produce_diffed_timeseries(train, diffn)
        print('差分阶数为'+str(diffn)+'，已完成差分')
    print('开始进行ARMA拟合')
    order = choose_order(train[train.columns[2]], maxar, maxma)
    print('模型的阶数为：'+str(order))
    _ar = order[0]
    _ma = order[1]
    model = pf.ARIMA(data=train, ar=_ar, ma=_ma, family=pf.Normal())
    model.fit("MLE")
    test = test['payment_times']
    test_predict = model.predict(int(test_size))
    test_predict = predict_recover(test_predict, train, diffn)
    RMSE = np.sqrt(((np.array(test_predict)-np.array(test))**2).sum()/test.size)
    print("测试集的RMSE为："+str(RMSE))
