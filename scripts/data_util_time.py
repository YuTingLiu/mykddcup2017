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
from data_util import *
    
class model2:
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
        df.loc[:,'time_window_s']=pd.to_datetime(df['starting_time'],format=r'%Y/%m/%d %H:%M:%S')
    #    print(df.head())
        aggregate = lambda x : x.set_index('time_window_s').resample('20Min').agg(np.mean).fillna(np.mean(x))# avoid nan and inf
        df = df.groupby(['intersection_id','tollgate_id'])[['time_window_s','travel_time']].apply(aggregate)
#        print(df.head(100))
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
        df.to_csv(r'time_train_union.csv')
        return df
    
    def union(self,dflist):
        for df in dflist:
            print(df.head())
        df = pd.concat(dflist)
        df = df.reset_index()
        df = prep(df)
        df.to_csv(r'time_union.csv')

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
    model = model2()
    in_file=r'E:\大数据实践\天池大赛KDD CUP\dataSet_phase2\trajectories(table_5)_training2.csv'
    fdir=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\testing_phase1\weather (table 7)_test1.csv'
    dflist.append(model.data_union(in_file,fdir))
    
    in_file=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\training\trajectories(table 5)_training.csv'
    fdir=r'E:\大数据实践\天池大赛KDD CUP\data\weather (table 7)_training_update.csv'
    dflist.append(model.data_union(in_file,fdir))
    
    
    in_file=r'E:\大数据实践\天池大赛KDD CUP\dataSet_phase2\trajectories(table 5)_test2.csv'
    fdir=r'E:\大数据实践\天池大赛KDD CUP\dataSet_phase2\weather (table 7)_2.csv'
    dflist.append(model.data_union(in_file,fdir))
    
    
#    in_file=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\testing_phase1\trajectories(table 5)_test1.csv'
#    fdir=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\testing_phase1\weather (table 7)_test1.csv'
#    dflist.append(model.data_union(in_file,fdir))

    model.union(dflist)

    
if __name__ == '__main__':
    main()