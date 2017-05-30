# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:34:08 2017

@author: L
@数据处理脚本

"""

import pandas as pd 
import numpy as np
from datetime import timedelta,datetime
from pandas.tseries.offsets import *

import matplotlib.pylab as plt
import sys 
from pandas.tseries.offsets import Hour


def gen_df(fdir='union.csv',start='10/8/2016 06:00',freq='20Min',normalize = False,pat = False,periods=72,days=7):
    '''
    help fun
    '''
    if freq is '20Min':
        df = load_volume(fdir=fdir)
        train_seq = produce_seq(start=start,periods=periods,freq='20Min',days = days)
        if len(train_seq) != 72*20:
            print('train_seq len ',len(train_seq))
#            sys.exit()
    else:
        sys.exit(0)
#    print(df)
    return df,train_seq

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

def datetime2str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") 


def check_df(df):
    return 0


def load_volume(fdir='training_20min_avg_volume.csv'):
    df = pd.read_csv(fdir,date_parser=lambda x : pd.to_datetime(x,format=r'%Y/%m/%d %H:%M'),infer_datetime_format=True)
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    return df
def load_test(fdir='test1_20min_avg_volume.csv'):
    df = pd.read_csv(fdir,date_parser=lambda x : pd.to_datetime(x,format=r'%Y/%m/%d %H:%M'),infer_datetime_format=True)
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    return df
    
def prep(df,pat = True,normalize=False,weekday=True):
    #add some patten here     
        
    #Adaptive and Natural Computing Algorithms: 10th International Conference ..
    #normalize the traffic flow data to daily average
    if normalize is True:
        key = lambda x : x.dayofyear
        zscore = lambda x : (x-x.mean())/x.std()
#        zscore = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x))
        zscore = lambda x : np.log(x)
        df1 = df.set_index('time_window_s')['volume'].groupby(key).transform(zscore)
#        print(df1.groupby(key).mean())
#        print(df1.groupby(key).std())
        df.loc[:,'volume'] = df1.values
#    print(df.dtypes)
    if weekday is True:
        s = pd.Series(df['time_window_s'].dt.weekday,index=df.index,name='dayofweek')/7
        df = df.join(s)
        s = pd.Series(df['time_window_s'].dt.hour,index=df.index,name='hour')/24
        df = df.join(s)
        s = pd.Series(df['time_window_s'].dt.minute,index=df.index,name='minute')/60
        df = df.join(s)
    
    df.loc[:,'pattern'] = 0
    if pat is True:
        df = df.set_index('time_window_s')
        df.loc[:,'pattern'] = 0
        seq = pd.date_range(start='10/1/2016',end='10/7/2016',freq='20Min')        
        df.loc[seq,'pattern'] = 1
        df=df.reset_index()
        
        #添加是否上班
#        df.loc[:,'is_work'] = 1
#        df[]
    print(df.head())
    return df

def load_weather(fdir=r'E:\大数据实践\天池大赛KDD CUP\data\weather (table 7)_training_update.csv'):
    df = pd.read_csv(fdir,date_parser=[0],infer_datetime_format=True)
    df.loc[:,'date']=pd.to_datetime(df['date'],format=r'%Y/%m/%d %H:%M:%S')    
    t_seq = []
    for i in range(len(df)):
        t_seq.append(pd.to_datetime(df.loc[i,'date']) + Hour(df.loc[i,'hour']))
    df.loc[:,'time_window_s'] = t_seq
    df = df[['time_window_s','pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation']]
    df = df.set_index('time_window_s').resample('20Min').ffill()
    
    #normalize
    df = df.apply(lambda x : (x-x.mean())/x.std())
    return df.reset_index()

def fun(df1):
    '''
    按照要求填充数据
    '''
    df2 = pd.DataFrame(df1).reindex()
    if len(df2)>0:
        direction = list(df2['direction'].drop_duplicates().values)
#        print('\r\n',direction)
        if len(direction) != 2:
#            print(df2[['direction','volume']])
            if direction[0]==0:
                df2.loc[:,'direction'] = 1
                df2.loc[:,'volume'] = 0
            elif direction[0]==1:
                df2.loc[:,'direction'] = 0
                df2.loc[:,'volume'] = 0
            df1 = pd.concat([df1,df2]).sort_values(by='direction')
        else:
            df1 = df1.sort_values(by='direction')
    else:
        #empty df2
        df2.loc[:,'direction']=[[0],[1]]
        df2.loc[:,'volume']=[[0],[0]]
        df2.loc[:,'pattern']=[[0],[0]]
        df1 = pd.concat([df1,df2]).sort_values(by='direction')
    return df1


def next_batch(df,t='2016/9/19 00:00',k=1):
    '''
    get the values of input nodes
    input df is dataset
         t : time.interval is 20min  datetime64[ns]
         k : tollgate no.
    output nodelist = [v(t-20,k-1),v(t-20,k-2),v(t,k),v(t,k-1),v(t,k-2)]
    '''
    t = pd.to_datetime(t,format=r'%Y/%m/%d %H:%M')
#    print(t)
#    print(df[df['time_window_s']==t].head())
    t = datetime(t.year, t.month, t.day,t.hour, t.minute, 0)
    t_20 = t-Minute(20)
    t_a20 = t+Minute(20)
    
    #tollgate list is 1,2,3 , get tollgate expect k
    batch_list = []
    toll_list = [1,2,3]
    toll_list.remove(k)
    #1.append k
    df1 = df[(df['tollgate']==k) & (df['time_window_s']==t)]
    batch_list.append(fun(df1))
    #2. append other
    for k in toll_list:
        df1 = df[(df['tollgate']==k) & (df['time_window_s']==t)]
        batch_list.append(fun(df1))
        df1 = df[(df['tollgate']==k) & (df['time_window_s']==t_20)]
        batch_list.append(fun(df1))
    
    #3. y
    df1 = df[(df['tollgate']==k) & (df['time_window_s']==t_a20)]
    df1 = fun(df1)
    pattern = df1['pattern'].tolist()[0]
    dayofweek = df1['dayofweek'].tolist()[0]
    hour = df1['hour'].tolist()[0]
    minute = df1['minute'].tolist()[0]
    return pd.concat(batch_list),df1,pattern,dayofweek,hour,minute

def get_all_batch(df,t_seq,k):
    x_list = []
    y_list = []
    for t in t_seq:
        batch_x,batch_y,pattern,dayofweek,hour,minute = next_batch(df,t,k)
        batch_x = batch_x['volume'].tolist()
#        batch_x.append(pattern)
        batch_x.append(dayofweek)
        batch_x.append(hour)
        batch_x.append(minute)
        
        batch_y = batch_y['volume'].tolist()
#        batch_y.append(pattern)
        
        x_list.append(batch_x)
        y_list.append(batch_y)
#    print(x_list)
#    print('y_list',y_list)
    return x_list,y_list


def plot_tollgate_volume(df,t_seq,k=1):
    print(t_seq[0],t_seq[-1])
    df = df[df['tollgate']==k][['time_window_s','direction','volume']].set_index('time_window_s')
    print('direction 0',len(df[df['direction']==0]))
#    print(df[df['direction']==0].loc[t_seq,:]['volume'])
    plt.plot(df[df['direction']==0].loc[t_seq,:]['volume'])
    plt.show()
    print('direction 1',len(df[df['direction']==1]))
    plt.plot(df[df['direction']==1].loc[t_seq,:]['volume'])
    plt.show()
    
class model1:
    '''
    data producer
    '''
    def __init__(self):
        self.name = 'modelT'
        
    def load_volume_hour(self,fdir='volume(table 6)_training.csv'):
        '''
        20170518
        new model for this task
        imput history record for vehicle
        output tollgate,direction,time_window_s,volume
        '''
        df = pd.read_csv(fdir,infer_datetime_format=True)
        df.loc[:,'time_window_s']=pd.to_datetime(df['time'],format=r'%Y/%m/%d %H:%M:%S')
    #    print(df.head())
        df.loc[:,'volume'] = 1
        aggregate = lambda x : x.set_index('time_window_s').resample('T').agg(sum).fillna(0)
        df = df.groupby(['tollgate','direction'])[['time_window_s','volume']].apply(aggregate)
    #    print(df.head(100))
        return df.reset_index()
    
    def load_volume_20Min(self,fdir='volume(table 6)_training.csv'):
        '''
        20170521
        new model for this task
        imput history record for vehicle
        output tollgate,direction,time_window_s,volume
        '''
        df = pd.read_csv(fdir,infer_datetime_format=True)
        df.loc[:,'time_window_s']=pd.to_datetime(df['date_time'],format=r'%Y/%m/%d %H:%M:%S')
    #    print(df.head())
        df.loc[:,'volume'] = 1
        aggregate = lambda x : x.set_index('time_window_s').resample('20Min').agg(np.sum).fillna(np.mean(x))# avoid nan and inf
        df = df.groupby(['tollgate','direction'])[['time_window_s','volume']].apply(aggregate)
    #    print(df.head(100))
        return df.reset_index()
    
    def data_union(self,data,weather):
        in_file=data
        src = self.load_volume_20Min(in_file)
        df = load_weather(fdir=weather)
        src = src.set_index('time_window_s')
        df = df.set_index('time_window_s')
        df = src.join(df,how='left')
        print(len(src),len(df))
        if len(src) != len(df):
            sys.exit()
        df.to_csv(r'volume_train_union.csv')
        return df
    
    def union(self,dflist):
        for df in dflist:
            print(df.head())
        df = pd.concat(dflist)
        df = df.reset_index()
        df = prep(df)
        df.to_csv(r'volume_union.csv')

def modification(ts , method=2):
    step = 1
    #规则
    if len(ts[ts>500]):
        if ts.mean() > 500:
            print(ts.describe())
            ts[ts>500] = abs(ts-ts.median())/ts.mad()
            if len(ts[ts>500]):
                sys.exit('ts>1000')
        else:
            ts[ts>500] = ts.mean()
    print(ts)
    print(ts.describe())
    #参考 ： 东方金工《选股因子数据的异常值处理和正态转换》
    print('#1. 标准差3倍的数据归纳为异常值')
    print(ts.std()*3,':',ts[ts>ts.std()*3])
    print('#2. 用样本中位数MAD代替标准差')
    print(ts.mad()*3,':',ts[ts>ts.mad()*3])
    print('#3. 使用Hubert& Vandervieren （2007） Boxplot 改进方法')
    from statsmodels.stats.stattools import medcouple
    mc = medcouple(ts)
    q1 = ts.quantile(0.25)
    q3 = ts.quantile(0.75)
    median = ts.median()
    IQR=q3-q1
    if mc >= 0:
        l = q1 - 1.5*np.exp(-3.5*mc)*IQR
        u = q3 + 1.5*np.exp(mc)*IQR
    if mc < 0:
        l = q1 - 1.5*np.exp(-4*mc)*IQR
        u = q3 + 1.5*np.exp(mc)*IQR
    print('low',l,'high',u)
    print(ts[(ts<l)&(ts>u)])
    
    ts1 = ts.copy()
    if method == 1:
        while len(ts1[ts1>ts1.std()*3]) > 0:
            print ('modi ',ts1.describe())
            ts1[ts1>ts1.std()*3] = ts1.std()*3
    if method == 2:
        while len(ts1[abs(ts1-median)/ts.mad() >3]) > 0 and step <10:
            print ('modi ',ts1.describe())
            ts1[abs(ts1-median)/ts.mad() >3] = abs(ts1-median)/ts.mad() # 剔除偏离中位数x倍以上的数据
            step += 1
    if method == 3:
        while len(ts1[ts1<l]) > 0 and len(ts1[ts1>u]) > 0:
            print ('modi ',ts1.describe())
            ts1[ts1<l] = l
            ts1[ts1>u] = u
    plt.plot(ts1.index,ts1.values,'g')
    plt.plot(ts1.index,ts.values,'r')
    plt.show()
    return ts1

def main():

    in_file = 'training_20min_avg_volume.csv'
    src = load_volume(in_file)
##    df = prep(df)
#    t_seq = pd.date_range(start='09/19/2016 06:00',end='9/19/2016 10:00',freq='20Min')
#    t_seq1 = pd.date_range(start='10/1/2016',end='10/7/2016',freq='20Min')
#    t_seq2 = pd.date_range(start='10/8/2016',end='10/17/2016',freq='20Min')
#    plot_tollgate_volume(df,t_seq,k=1)
#    plot_tollgate_volume(df,t_seq1,k=1)
#    plot_tollgate_volume(df,t_seq2,k=1)
#    get_all_batch(df,t_seq,k=1)

#    in_file=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\training\volume(table 6)_training.csv'
#    model = model1()
#    df = model.load_volume_hour(fdir=in_file)
#    df = prep(df)
#    print(df.dtypes)
#    print(df.head(100))
#    t_seq = pd.date_range(start='09/19/2016',periods=20,freq='T')
#    print(len(t_seq))
#    x_list,y_list = get_all_batch(df,t_seq,k=1)
#    print(np.array(x_list).shape)
#    print(np.array(y_list).shape)

    dflist = []
    model = model1()
    in_file=r'E:\大数据实践\天池大赛KDD CUP\dataSet_phase2\volume(table 6)_training2.csv'
    fdir=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\testing_phase1\weather (table 7)_test1.csv'
    dflist.append(model.data_union(in_file,fdir))
    
    in_file=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\training\volume(table 6)_training.csv'
    fdir=r'E:\大数据实践\天池大赛KDD CUP\data\weather (table 7)_training_update.csv'
    dflist.append(model.data_union(in_file,fdir))
    
    
    in_file=r'E:\大数据实践\天池大赛KDD CUP\dataSet_phase2\volume(table 6)_test2.csv'
    fdir=r'E:\大数据实践\天池大赛KDD CUP\dataSet_phase2\weather (table 7)_2.csv'
    dflist.append(model.data_union(in_file,fdir))
    
    
#    in_file=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\testing_phase1\trajectories(table 5)_test1.csv'
#    fdir=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\testing_phase1\weather (table 7)_test1.csv'
#    dflist.append(model.data_union(in_file,fdir))

    model.union(dflist)
    
if __name__ == '__main__':
    this = ''
    main()