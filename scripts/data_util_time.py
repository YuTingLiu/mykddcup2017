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
#        
#    def load_volume_hour(self,fdir='volume(table 6)_training.csv'):
#        '''
#        20170518
#        new model for this task
#        imput history record for vehicle
#        output tollgate,direction,time_window_s,volume
#        '''
#        df = pd.read_csv(fdir,infer_datetime_format=True)
#        df.loc[:,'time_window_s']=pd.to_datetime(df['time'],format=r'%Y/%m/%d %H:%M:%S')
#    #    print(df.head())
#        df.loc[:,'volume'] = 1
#        aggregate = lambda x : x.set_index('time_window_s').resample('T').agg(sum).fillna(0)
#        df = df.groupby(['tollgate','direction'])[['time_window_s','volume']].apply(aggregate)
#    #    print(df.head(100))
#        return df.reset_index()
    
    def path_pre(self):
        link=r'../data/dataSets/training/links (table 3).csv'
        route=r'../data/dataSets/training/routes (table 4).csv'
        route = pd.read_csv(route)
        link = pd.read_csv(link)
        # 将110，123变成[110, 123]
        def split(str): return str.split(',')
        route.link_seq = route.link_seq.apply(split)
        # 将数据按各link展开
        rows = []
        _ = route.apply(lambda row: [rows.append([row['intersection_id'], row['tollgate_id'], nn]) 
                                 for nn in row.link_seq], axis=1)
        col = ['intersection_id', 'tollgate_id', 'link_id']
        route = pd.DataFrame(rows, columns=col)
        print(route.head())
        
        #将route展开
        route.loc[:,'count'] = 1
        road = pd.pivot_table(route,index=['intersection_id','tollgate_id'],values='count',columns=['link_id'],aggfunc=np.sum,fill_value=0)
        road = road.reset_index()
        print(road)
        
        # 判断路段是否为交叉入口或者出口
        link['cross_in'] = 0
        link['cross_out'] = 0
        for i, row in link.iterrows():
            if ',' in str(row['in_top']):
                link.loc[i, 'cross_in'] = 1
            if ',' in str(row['out_top']):
                link.loc[i, 'cross_out'] = 1
        print(link)
        # 将整数型数据转换成string类型
        link['link_id'] = link['link_id'].astype(str)
        route['link_id'] = route['link_id'].astype(str)
        route = pd.merge(route, link, on='link_id', how='left')
        route.drop(['in_top', 'out_top'], axis=1, inplace=True)
        print(route)
        # 对交叉出入口进行聚合
        b1 = route[['intersection_id', 'tollgate_id', 'cross_in']]\
                .groupby(['intersection_id', 'tollgate_id'])\
                .cross_in.sum().reset_index().rename(columns={'cross_in':'in_link_cross_conut'})
        b2 = route[['intersection_id', 'tollgate_id', 'cross_out']]\
                .groupby(['intersection_id', 'tollgate_id'])\
                .cross_out.sum().reset_index().rename(columns={'cross_out':'out_link_cross_count'})
        road = pd.merge(road,b2,on=['intersection_id', 'tollgate_id'],how='left')
        road = pd.merge(road,b1,on=['intersection_id', 'tollgate_id'],how='left')
        print(road.head())
        
        # 路程
        b1 = route[['intersection_id', 'tollgate_id', 'length']].groupby(['intersection_id', 'tollgate_id']).length.sum().reset_index()
        road = pd.merge(road, b1, on=['intersection_id', 'tollgate_id'], how='left')
        print(road.head())
        # 车道数1车道2车道3车道4车道的link数，后期考虑：占总路程的比率
        # 各个路径的link总数
        b1 = route[['intersection_id', 'tollgate_id']]\
                .groupby(['intersection_id', 'tollgate_id']).size()\
                .reset_index().rename(columns={0:'link_count'})
        road = pd.merge(road, b1, on=['intersection_id', 'tollgate_id'], how='left')
        print(road.head())
        # 1,2,3,4车道道路长度
        # 测试1
        b1 = route[route.lanes==1][['intersection_id', 'tollgate_id', 'length']]\
                .groupby(['intersection_id', 'tollgate_id']).length.sum()\
                .reset_index().rename(columns={'length':'1_length'})
        road = pd.merge(road, b1, on=['intersection_id', 'tollgate_id'],how='left')
        b1 = route[route.lanes==2][['intersection_id', 'tollgate_id', 'length']]\
        .groupby(['intersection_id', 'tollgate_id']).length.sum()\
        .reset_index().rename(columns={'length':'2_length'})
        road = pd.merge(road, b1, on=['intersection_id', 'tollgate_id'],how='left')
        b1 = route[route.lanes==3][['intersection_id', 'tollgate_id', 'length']]\
                .groupby(['intersection_id', 'tollgate_id']).length.sum()\
                .reset_index().rename(columns={'length':'3_length'})
        road = pd.merge(road, b1, on=['intersection_id', 'tollgate_id'],how='left')
        b1 = route[route.lanes==4][['intersection_id', 'tollgate_id', 'length']]\
                .groupby(['intersection_id', 'tollgate_id']).length.sum()\
                .reset_index().rename(columns={'length':'4_length'})
        road = pd.merge(road, b1, on=['intersection_id', 'tollgate_id'],how='left')
        print(road.head())
        return road.fillna(0)
        
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
        #检查
        df = df.reset_index()
        ####制造特征
        #添加时间特征
        s = pd.Series(df['time_window_s'].dt.weekday_name,index=df.index,name='dayofweek')
        df = df.join(s)
        df.loc[:,'count'] = 1
        weekday = pd.pivot_table(df,index=['intersection_id','tollgate_id','time_window_s'],values='count',columns=['dayofweek'],aggfunc=np.sum,fill_value=0)
        print(weekday.head())
#        sys.exit()
        s = pd.Series(df['time_window_s'].dt.hour,index=df.index,name='hour')
        df = df.join(s)
        cut_points = [6,12,18,21]
        labels = ["midnight","morning","noon","afternoon","evening"]
        df["hour"] = binning(df["hour"], cut_points, labels)
        print(df.head())
        df.loc[:,'count'] = 1
        hour = pd.pivot_table(df,index=['intersection_id','tollgate_id','time_window_s'],values='count',columns=['hour'],aggfunc=np.sum,fill_value=0)
        print(hour.head())
        hour.to_csv(r'temp.csv')
#        sys.exit()
        s = pd.Series(df['time_window_s'].dt.minute,index=df.index,name='minute')
        df['minute'] = 'w'+s.astype(str)
        df.loc[:,'count'] = 1
        minute = pd.pivot_table(df,index=['intersection_id','tollgate_id','time_window_s'],values='count',columns=['minute'],aggfunc=np.sum,fill_value=0)
        print(minute.head())
        minute.to_csv(r'temp.csv')
#        sys.exit()
        
        df1 = weekday.join(hour).join(minute)
        print(df1.head())
        df1.to_csv(r'temp.csv')
        df = df.set_index(['intersection_id','tollgate_id','time_window_s']).join(df1)
        columns = ['travel_time','Friday','Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday','midnight','morning','noon','afternoon','evening','w0','w20','w40']
        df[columns].to_csv(r'temp.csv')
        print(df.head())
        print(df.columns)
        
        road = self.path_pre()
        df = df[columns].reset_index().set_index(['intersection_id','tollgate_id']).join(road.set_index(['intersection_id','tollgate_id']))
        df.to_csv(r'temp.csv')
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
        df.to_csv(r'time_union.csv',index=False)

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
    road = model.path_pre()
#    sys.exit()
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