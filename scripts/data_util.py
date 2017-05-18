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

def load_volume(fdir='training_20min_avg_volume.csv'):
    df = pd.read_csv(fdir,date_parser=lambda x : pd.to_datetime(x,format=r'%Y/%m/%d %H:%M'),infer_datetime_format=True)
    df.loc[:,'time_window_s']=pd.to_datetime(df['time_window_s'],format=r'%Y/%m/%d %H:%M:%S')
    return df
    
def prep(df,pat = True,normalize=True,weekday=True):
    #add some patten here
    if pat is True:
        different_volume = [0,1,2]
        
        df = df.set_index('time_window_s')
        print(len(df))
        df1 = df['09/19/2016':'9/30/2016']
        df1.loc[:,'pattern'] = 0
        df2 = df['10/1/2016':'10/7/2016']
        df2.loc[:,'pattern'] = 1
        df3 = df['10/8/2016':'10/17/2016']
        df3.loc[:,'pattern'] = 2
        df = pd.concat([df1,df2,df3])
        df = df.reset_index()
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
        s = pd.Series(df['time_window_s'].dt.weekday,index=df.index,name='dayofweek')
        df = df.join(s)
    print(df.head())
    return df

def load_weather(fdir='weather (table 7)_training_update.csv'):
    df = pd.read_csv(fdir,date_parser=[0],infer_datetime_format=True)

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
    df1 = df[(df['tollgate_id']==k) & (df['time_window_s']==t)]
    batch_list.append(fun(df1))
    #2. append other
    for k in toll_list:
        df1 = df[(df['tollgate_id']==k) & (df['time_window_s']==t)]
        batch_list.append(fun(df1))
        df1 = df[(df['tollgate_id']==k) & (df['time_window_s']==t_20)]
        batch_list.append(fun(df1))
    
    #3. y
    df1 = df[(df['tollgate_id']==k) & (df['time_window_s']==t_a20)]
    df1 = fun(df1)
    pattern = df1['pattern'].tolist()[0]
    dayofweek = df1['dayofweek'].tolist()[0]
    return pd.concat(batch_list),df1,pattern,dayofweek

def get_all_batch(df,t_seq,k):
    x_list = []
    y_list = []
    for t in t_seq:
        batch_x,batch_y,pattern,dayofweek = next_batch(df,t,k)
        batch_x = batch_x['volume'].tolist()
#        batch_x.append(pattern)
        batch_x.append(dayofweek)
        
        batch_y = batch_y['volume'].tolist()
#        batch_y.append(pattern)
        
        x_list.append(batch_x)
        y_list.append(batch_y)
#    print(x_list)
#    print('y_list',y_list)
    return x_list,y_list


def plot_tollgate_volume(df,t_seq,k=1):
    print(t_seq[0],t_seq[-1])
    df = df[df['tollgate_id']==k][['time_window_s','direction','volume']].set_index('time_window_s')
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
        output tollgate_id,direction,time_window_s,volume
        '''
        df = pd.read_csv(fdir,infer_datetime_format=True)
        df.loc[:,'time_window_s']=pd.to_datetime(df['time'],format=r'%Y/%m/%d %H:%M:%S')
    #    print(df.head())
        df.loc[:,'volume'] = 1
        aggregate = lambda x : x.set_index('time_window_s').resample('T').agg(sum).fillna(0)
        df = df.groupby(['tollgate_id','direction'])[['time_window_s','volume']].apply(aggregate)
    #    print(df.head(100))
        return df.reset_index()
    
    

def main():

    in_file = 'training_20min_avg_volume.csv'
    df = load_volume(in_file)
#    df = prep(df)
    t_seq = pd.date_range(start='09/19/2016 06:00',end='9/19/2016 10:00',freq='20Min')
    t_seq1 = pd.date_range(start='10/1/2016',end='10/7/2016',freq='20Min')
    t_seq2 = pd.date_range(start='10/8/2016',end='10/17/2016',freq='20Min')
    plot_tollgate_volume(df,t_seq,k=1)
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
if __name__ == '__main__':
    main()