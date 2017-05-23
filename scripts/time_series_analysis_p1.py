#coding:utf-8
import pandas as pd
from pandas import Series,DataFrame
import string
import re
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #测试作3d图
import pandas.io.date_converters as conv #日期转换接口
import time
import math
import os
from statsmodels.tsa.stattools import adfuller#Python的统计建模和计量经济学工具包
from statsmodels.tsa.seasonal import seasonal_decompose
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from numpy.linalg import LinAlgError
from statsmodels.graphics.api import qqplot#qq图
from scipy import stats
import statsmodels.api as sm
import sys


#时间平移
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Week
from pandas.tseries.offsets import Day
from pandas.tseries.offsets import Minute

#C:\Anaconda3\lib\site-packages\statsmodels\base\model.py:466: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
import warnings


'''
-----------------分析时间序列的平稳性。通过 均值、方差
@输入： DataFrame

时间序列只有两列，时间、data
'''
def test_stationarity(timeseries,window):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    #Plot rolling statistics:
    figure(window)
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    return float(dfoutput[0])


def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):  
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            #if s == "xxx":
                #continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)  
    return fileList



def train(resultARIMA,df_log):
##    plt.plot(df_log_diff)
##    plt.plot(results_ARIMA.fittedvalues, color='red')
##    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-df_log_diff)**2))
##    plt.show()
    predictions_ARIMA_diff = pd.Series(resultARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(df_log['song_daily_count'], index=df_log.index)
    
##    #：1）选择最末期数据；2）选择近三期数据的平均；3）选择近三期的移动平均
    predictions_ARIMA_log = predictions_ARIMA_diff_cumsum.add(predictions_ARIMA_log.mean(),fill_value=0)    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    print('len :%',len(predictions_ARIMA))
##    plt.plot(df)
##    plt.plot(predictions_ARIMA, color='red')
##    plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-np.exp(df_log))**2)/len(np.exp(df_log))))
##    plt.show()
    return predictions_ARIMA


def train_for_weka(result):
##    plt.plot(df_log_diff)
##    plt.plot(results_ARIMA.fittedvalues, color='red')
##    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-df_log_diff)**2))
##    plt.show()
##    predictions_ARIMA_diff = pd.Series(result, copy=True)
##    print(predictions_ARIMA_diff.head())
    predictions_ARIMA_diff_cumsum = result
    print( predictions_ARIMA_diff_cumsum.head())
    predictions_ARIMA_log = pd.Series(df_log['song_daily_count'], index=df_log.index)
    print('predict len:%', len(predictions_ARIMA_log.index))
    print(predictions_ARIMA_log.head())
    
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum['train'],fill_value=0)
    print(df_log.head())
    print(predictions_ARIMA_log.head())
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    print('len :%',len(predictions_ARIMA))
##    plt.plot(df)
##    plt.plot(predictions_ARIMA, color='red')
##    plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-np.exp(df_log))**2)/len(np.exp(df_log))))
##    plt.show()
    return predictions_ARIMA

  
def train_for_predict(predict_series,df_log):
    predictions_ARIMA_diff_cumsum = predict_series
##    print( predictions_ARIMA_diff_cumsum.head())
    predictions_ARIMA_log = pd.Series(df_log, index=df_log.index)
##    print('predict len:%', len(predictions_ARIMA_log.index))
##    print(predictions_ARIMA_log.head())

    #：1）选择最末期数据；2）选择近三期数据的平均；3）选择近三期的移动平均
##    predictions_ARIMA_log = predictions_ARIMA_diff_cumsum.add(predictions_ARIMA_log.mean(),fill_value=0)    

    #每月一分组，大概吧，这个可以再看
    #没有进入到分组-----》按照月份分组有什么不对
    grouped = predictions_ARIMA_diff_cumsum.groupby(lambda x:x.month)
    result = pd.Series()
    for name,group in grouped:
        print('before')
        print(group)
        plt.plot(predictions_ARIMA_log)
##        mean = predictions_ARIMA_log.rolling(30).mean().shift(len(group.index),freq='D')
        #使用最近一个月的平均值作为基础值
        #存在问题：所以会存在月预测出现相同，与上个月相同。
        mean = predictions_ARIMA_log.shift(len(group.index),freq='D').mean()
        
        #使用最近三个月的移动平均使用基值
##        mean = predictions_ARIMA_log.shift(len(group.index),freq='D')
##        mean = mean.tail(90).rolling(30).mean()
##        mean = mean.mean()
        
        #使用最近三个月的移动平均使用基值
##        mean = predictions_ARIMA_log.shift(len(group.index),freq='D')#时间一致
##        mean = mean.rolling(2).mean()
##        mean = pd.Series(mean.tail(len(group.index)).values,index = group.index)
##        plt.plot(mean, color='red')
##        plt.show()
        print('mean')
        print(mean)
        group = group.add(mean,fill_value=0)
        print('after')
        print(group)
        result = result.append(group)


    print(result.head())
    predictions_ARIMA = np.exp(result)
    print('predictions_ARIMA len :%',len(predictions_ARIMA))
##    plt.plot(df)
##    plt.plot(predictions_ARIMA, color='red')
##    plt.title('RMSE: %.4f'%
##              np.sqrt(sum((predictions_ARIMA-np.exp(df_log['song_daily_count']))**2)/
##                      len(np.exp(df_log['song_daily_count']))))
##    plt.show()
    return predictions_ARIMA

'''
#------------------------不同的预处理
#对时间序列取对数,做差分
'''
def df_log_func(df):
    
    x = 29#window
    #取对数
    df_log = np.log(df)
    return df_log

def df_log_diff_func(df_log,x):
    #取平均线
##    moving_avg = df_log.rolling(x).mean()
    #取差分
    df_log_diff =  df_log - df_log.shift(x)
    df_log_diff.dropna(inplace=True)
##    print('df_log_diff len:')
##    print(len(df_log_diff.index))
    return df_log_diff
    
def undiff_unlog_df(df_log_diff,x):
    '''
    由序列还原
    '''
    print(df_log_diff)
    df = df_log_diff + df_log_diff.shift(-x)
    print(df)
    df = np.exp(df)
    print('exp',df)
    return df

def df_log_ewma_func(df_log,x):
    ###使用ewma来平稳序列
    expwighted_avg = pd.ewma(df_log, halflife=x)
    plt.plot(df_log)
    plt.plot(expwighted_avg, color='red')
    plt.show(block=False)
    df_log_ewma_diff = df_log - expwighted_avg
    temp = test_stationarity(df_log_ewma_diff,x)

def plot_acf_pacf(ts_log_diff,ts_log,nlags=60):
    ts_log_diff = ts_log
    figure(2)
    lag_acf = acf(ts_log_diff, nlags=nlags)
    lag_pacf = pacf(ts_log_diff, nlags=nlags, method='ols')
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
'''
#分解序列，没有实现 
'''
def plot_residual(df_log):

##    decomposition = seasonal_decompose(df_log, pd.DatetimeIndex(start='3/1/2015',
##                                                             periods=len(df_log.index),freq='D'))
    print(df_log['song_daily_count'].values)
    decomposition = seasonal_decompose(df_log['song_daily_count'].values,freq='D')

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(df_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    
def plot_compare(src,trans,window):
    figure(1)
    if src is not None:
        plt.plot(src)
    if trans is not None:
        plt.plot(trans,color='red')
    plt.show(block=False)
##    ts_log_moving_avg_diff = src-trans
##    ts_log_moving_avg_diff.dropna(inplace=True)
##    temp = test_stationarity(df_log_diff,x)

def mse(true,pred):
    return np.sum(np.abs(true-pred)/true)

#model
'''
#找到最后的最优模型
'''
def fing_best_pqd(df_log,pqr):
    p=pqr[0]
    q=pqr[2]
    d=pqr[1]
    #在测试中已经确定，1阶差分便可以应对这个序列
#    df_log_diff = df_log_diff_func(df_log,1)
#    param = [[10, 1, 8],
#        [17, 1, 11],
#        [2, 1, 10],
#        [1, 1, 2],
#        [1, 1, 4]]
#    for p,d,q in param:
#        try:
#            results_ARMA=ARMA(df_log_diff,(p,q)).fit(disp=-1,method='css')
#            return [p,d,q]
#        except Exception as e:
#            print(e)
#            pass
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for p in range(10):
            for d in range(2):
                for q in range(6):
                    results_ARIMA=ARIMA(df_log,(p,d,q))
                    try:
                        fit1=results_ARIMA.fit(transparams=True,disp=-1,method='css')##
#                        dw_test.append(np.abs(model_observe(fit1)-2))
#                        fitted.append(fit1)
#                    except Exception  as e:
#                        print(e,'error fit')
#                    try:
#                        for i in range(5):
#                            init = [np.array([1.] * d + [1./(p+1)] * p + [1./(q+1)] * q),
#                                np.array([np.mean(df_log)] * d + [1./(p+1)] * p + [1./(q+1)] * q),
#                                np.array([.1] * d + [1./(p+1)] * p + [1./(q+1)] * q),
#                                np.array([np.mean(df_log)] * d + [1./(p+1)] * p + [1./(q+1)] * q),
#                                [0.1 for j in range(p+q+2)]]
#                            fit1=results_ARIMA.fit(start_params=init[i],disp=-1,transparams=True,method='mle')#
#                            dw_test.append(np.abs(model_observe(fit1)-2))
#                            fitted.append(fit1)
#                    except Exception  as e:
#                            print(e,'error fit    11111')
#                            pass
#                    try:
#                        fit1=results_ARIMA.fit(transparams=True,disp=-1,method = 'mle')##
#                        dw_test.append(np.abs(model_observe(fit1)-2))
#                        fitted.append(fit1)
#                    except Exception  as e:
#                        print(e,'error fit    3444444')
#                        pass
#                    try:
#                        fit1=results_ARIMA.fit(disp=-1)##
#                        dw_test.append(np.abs(model_observe(fit1)-2))
#                        fitted.append(fit1)
#                    except Exception  as e:
#                        print(e,'error fit 45555')
#                        pass
#                    try:
#                        fit1=results_ARIMA.fit(transparams=True,disp=-1,method='css-mle')
                        x=fit1.aic
                        x1= p,d,q
    #                    print (x1,x)
                        if isnan(x):
        ##                        print('find nan ',x1,x)
                            continue
                        aic.append(x)
                        pdq.append(x1)
                    except Exception  as e:
#                        print(e,'there is no way to fix it')
                        pass
    keys = pdq
    values = aic
    d = dict(zip(keys, values))
    print (d)
    #得到最小值----------------------这个方法还不会
    if d != None:
        minaicFind=min(d,key=d.get)
    return minaicFind


'''
修改的地方 ：  预测序列的平移
'''
def model_observe(model):
    print('D.W统计量')
    print(sm.stats.durbin_watson(model.resid.values))
##    #残差图
##    figure(1)
##    model.resid.plot().get_figure().show()
##    #QQ图
##    resid = model.resid
##    print(stats.normaltest(resid))#检验模型是否与正太分布不同
##    print(resid)
##    fig = plt.figure(figsize=(12,8))
##    ax = fig.add_subplot(111)
##    fig = qqplot(resid, line='q', ax=ax, fit=True)
##    fig.show()
    return sm.stats.durbin_watson(model.resid.values)
def ARIMA_predictBynum(df,pqr):
    df_log = df_log_func(df)
    #findIng----------
    
    minaic = fing_best_pqd(df_log,pqr)
    return minaic


def predict(df,step,p,d,q):
    '''
    输入：log后的数据，注意不需要差分
    '''
    dw_test = []
    fitted = []
    df_log = df_log_func(df)
#    df_log_diff = df_log_diff_func(df_log,d).dropna(inplace=True)#选择X阶差分
#    nan = pd.isnull(df_log_diff)
#    print(nan[nan.values==True])
    print('train time zone is ',df_log.index[0],df_log.index[-1])
    print(p,d,q)
    #分析时，使用了差分，这里并不需要进行差分
    results_ARIMA=ARIMA(df_log,order=[p,d,q])
    try:
        fit1=results_ARIMA.fit(transparams=True,disp=-1,method='css')
        dw_test.append(np.abs(model_observe(fit1)-2))
        fitted.append(fit1)
    except Exception  as e:
        print(e,'there is no way to fix it')
        return None,None,0
#    try:
#        fit1=results_ARIMA.fit(transparams=True,disp=-1,method='mle')
#        dw_test.append(np.abs(model_observe(fit1)-2))
#        fitted.append(fit1)
#    except Exception  as e:
#        print(e,'there is no way to fix it')
#    try:
#        fit1=results_ARIMA.fit(transparams=True,disp=-1,method='css-mle')
#        dw_test.append(np.abs(model_observe(fit1)-2))
#        fitted.append(fit1)
#    except Exception  as e:
#        print(e,'there is no way to fix it')
    d = dict(zip(fitted,dw_test))
    #得到最小值----------------------这个方法还不会
    if d is not None:
        fit1=min(d,key=d.get)
    
    pred = fit1.predict(end = len(df))
    result = fit1.forecast(step)
    return pred,result,model_observe(fit1)


def main_1(df,tollgate_id,direction,trainning_seq,step,pqr):
    '''
    input trainning seq ,
    output pred seq
    '''
    print(len(df))
    ts = df[trainning_seq]
    if (len(ts[ts==0])>0):
        print('for log ,contain zero',len(ts[ts==0]))
        sys.exit()
    print('series time zone',ts.index[0],df.index[-1])
    print('excepted train zone',trainning_seq[0],trainning_seq[-1])
    print('begin arima ,',tollgate_id ,'d',direction)
    
    #由于是1阶差分，训练序列少一个节点，所以，不需要—+Minute(20)
    pred_seq = pd.date_range(start = (trainning_seq[-1]),periods = step, freq='20Min')
    true = df[pred_seq].fillna(0)
    
    #predict
    #pqr = [10,1,8]
    pqr = ARIMA_predictBynum(ts,pqr)
    #tollgate_dir = ''.join([str(tollgate_id),'-',str(direction)])
    #aicList.append([tollgate_dir,pqr])
    
    #按照步数来预测，结果叠加到已知序列
    pred,result,dw = predict(ts,step,pqr[0],pqr[1],pqr[2])
    if dw<1 or dw>3:
        pqr = ARIMA_predictBynum(ts,pqr)
        print(pqr)
        pred,result,_ = predict(ts,step,pqr[0],pqr[1],pqr[2])
        plot_compare(ts,None,0)
        if result is None:
            result = []
            result.append(np.log((ts-ts.mean())/ts.std()))
#    if len(pred)<len(train_seq):
#	pred = pred.fillna(pred.mean())
#    print(len(pred),len(result[0]))
#    pred = np.array(pred.values)
#    pred = pred.reshape((len(pred),1))
    #print('拼接序列')
    #result = np.concatenate((pred,result[0]))
    result = result[0]
    print('arima','result',result)
    print('强制对齐预测数据时间')
#    if len(result) != len(trainning_seq):
#        print('Length of ARIMA output is ',len(result))
#        sys.exit(0)
    pred_seq = pd.date_range(start = (trainning_seq[-1]+Minute(20)),periods = len(result), freq='20Min')
    result = pd.Series(data=result,name='pred',index=pred_seq)
    result = np.exp(result)
    ###去除不可能的点
    
    true = df[pred_seq].fillna(0)
#    print(result)
    print('true seq len and pred seq len is :',len(true),len(result))
#    plot_compare(true,result,0)
    
    
    df = pd.concat([true,result],axis=1)
    df = df.reset_index()
    df.columns = ['time_window_s','volume','pred']
    df.loc[:,'tollgate_id'] = tollgate_id
    df.loc[:,'direction']=direction
    return df,pqr    
    
#所有测试代码，最后放到这个地方 0602
def test_code(ts,window):
    ts_log = df_log_func(ts)
    test_stationarity(ts,window)
    plot_acf_pacf(df_log_diff_func(ts,2),ts_log)
    
    
singerCode=[]
dw_test = []
pdqlist = []
commList = []
window =1
pdq=[]
aic=[]
dwtest = []
aicList = []
total = []
if __name__ == '__main__':
    from data_util import *
    in_file=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\training\volume(table 6)_training.csv'
    model = model1()
    df = model.load_volume_20Min(fdir = in_file)
    print(df.head())
    start = '10/15/2016 06:00'
    end = '10/16/2016 06:00'
    periods = 12 #4 hours
    freq = '20Min'
    t_seq = pd.date_range(start=start,end=end,freq=freq)
    #test
#    ts = df[(df['tollgate_id']==1) & (df['direction'] ==0)].set_index('time_window_s')['volume'][t_seq]
#    test_code(ts,72)
#    sys.exit(0)
    
    
    pred_seq = (t_seq + Day(len(set(t_seq.day))))[0:12]
    test = DataFrame()
    fileName=''
    for i in range(1,4):
        df1 = df[df['tollgate_id']==i]
        #按照direction不同来预测
        grouped = df1.groupby('direction')
        for direction,group in grouped:
            print(direction,group.head())
            pred = group.set_index('time_window_s')['volume'][t_seq].fillna(0)
            pred_y = group.set_index('time_window_s')['volume'][t_seq].fillna(0)
            true = group.set_index('time_window_s')[['volume']].loc[pred_seq,:].fillna(0)
            #具体选择预测时间段
            pqr = ARIMA_predictBynum(pred,[0,0,0])
            tollgate_d = ''.join([str(i),'-',str(direction)])
            aicList.append([tollgate_d,pqr])
            
            #按照步数来预测，结果叠加到已知序列
            pred,result = predict(pred,pred_seq,pqr[0],pqr[1],pqr[2])
#            pred.name = 'pred'
            print(pred)
            print(result)
            result = pd.Series(data=result[0],name='pred',index=pred_seq)
            
            true_log = df_log_func(true)
            plot_compare(true_log,result,0)
            print(true_log.join(result))
#            sys.exit()
            total.append(true_log.join(result))
    output = pd.concat(total)
    output.to_csv(r'arima_result.csv')
    df = pd.DataFrame(aicList,columns=['tollgate_d','pqr'])
    df.to_csv('param_list.csv',index=False)
#        plot_residual(df_log_func(df))
