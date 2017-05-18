# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:25:30 2017

@author: L
@模型为什么使用匝道编号 ，而不使用车编号 ，简化问题
@最终模型的输入只有IT组合与其路线，有利于统计。
"""

import pandas as pd
import numpy as np
from igraph import *
from datetime import datetime,timedelta
def graph():
    '''
    @节点类型：交叉口intersection_id,收费站tollgate_id,匝道编号link_id
    @intersection_id:str
    @tollgate_id:int
    @link_id:[]
    @边属性：车流方向、车编号、时间、时长、天气
    '''
    path = r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\training'
    linkf = ''.join([path,'\\','links (table 3).csv'])
    routes = ''.join([path,'\\','routes (table 4).csv'])
    vec_pathf = ''.join([path,'\\','trajectories(table 5)_training.csv'])
    volume = ''.join([path,'\\','volume(table 6)_training.csv'])
    weather = ''.join([path,'\\','weather (table 7)_training.csv'])
    g = Graph()
    print(g.summary())
    linkdf = pd.DataFrame()
    linkdf = pd.read_csv(linkf)
    print(linkdf.head())
    print(linkdf.dtypes)
    print(len(linkdf))
    
    g.add_vertices(len(linkdf))
    for n in range(len(linkdf)):
        for i, col in enumerate(linkdf.columns):
            g.vs[n][col] = linkdf.loc[n,:][col]
        print(g.vs[n])
    
    
    #intersection id and tollgate id
    n = len(g.vs)
    insec = ['A','B','C']
    tollgates = [1,2,3]
    g.add_vertices(len(insec))
    for i in range(len(insec)):
        g.vs[n+i]['intersection_id'] = insec[i]
        print(g.vs[n+i])
    n = len(g.vs)
    g.add_vertices(len(tollgates))
    for i in range(len(tollgates)):
        g.vs[n+i]['tollgate_id'] = tollgates[i]
        print(g.vs[n+i])
    
    #vec history parse
    parse_vec_travel_seq(vec_pathf)
    
    
    #add vec path
#    vec_pathdf = pd.read_csv(vec_pathf)
#    print(vec_pathdf.head())
#    #extra each path slice for every vec
#    for i in range(len(vec_pathdf)):
#        path = vec_pathdf.loc[i,:]['travel_seq']
#        path = path.split(';')
#        startInsec = g.vs.select(intersection_id=vec_pathdf.loc[i,:]['intersection_id'])[0]
#        endToll = g.vs.select(tollgate_id=vec_pathdf.loc[i,:]['tollgate_id'])[0]
#        for j in range(len(path)):
#            linkid = int(path[j].split('#')[0])
#            param = {}
#            param['intersection_id'] = vec_pathdf.loc[i,:]['intersection_id']
#            param['tollgate_id'] = vec_pathdf.loc[i,:]['tollgate_id']
#            param['time'] = path[j].split('#')[1]
#            param['timeP'] = path[j].split('#')[2]
##            print(param)
#            #add a edge
##            print('link',linkid,len(g.vs.select(link_id=linkid)))
#            v = g.vs.select(link_id=linkid)[0]
#            if j < len(path)-1:
#                edgeId = g.ecount()
#                g.add_edge(startInsec,v)
#                edge = g.es[edgeId]
#                for key, value in param.items():
#                    edge[key] = value
#                startInsec = v
#            if j == len(path)-1:
#                edgeId = g.ecount()
#                g.add_edge(v,endToll)
#                edge = g.es[edgeId]
#                for key, value in param.items():
#                    edge[key] = value
##            print(edge)   
##        print(g.es.select(intersection_id='B'))
#    g.save(''.join([path,'\\','model.gml']))

def parse_vec_travel_seq(vec_pathf):
    '''
    @解析车辆行驶历史路径
    @输出DataFrame为：time|viecid|intersection|tollgate|link_id|tl(timelast)
    '''
    dflist = []
    #add vec path
    vec_pathdf = pd.read_csv(vec_pathf)
    print(vec_pathdf.head())
    #extra each path slice for every vec
    for i in range(len(vec_pathdf)):
        path = vec_pathdf.loc[i,:]['travel_seq']
        path = path.split(';')
        startInsec = vec_pathdf.loc[i,:]['intersection_id']
        endToll = vec_pathdf.loc[i,:]['tollgate_id']
        vehicle_id = vec_pathdf.loc[i,:]['vehicle_id']
        for j in range(len(path)):
            linkid = int(path[j].split('#')[0])
            time = path[j].split('#')[1]
            timeP = path[j].split('#')[2]
            dflist.append([time,str(vehicle_id),startInsec,str(endToll),str(linkid),str(timeP)])
    df = pd.DataFrame(dflist,columns=['time','vehicle_id','intersection_id','tollgate_id','link_id','tl'])
    print(df.head())
    df.to_csv(r'e:\vec_history.csv')

def cal_vec_routs_time():
    '''
    @按照vec取表里数据，得到这个vec的routs，判断routs是否正确，得到行驶时间
    @预测输出两张表，一张每车在时间段内通过routs的时间，一张不正确的路径
    '''
    path = r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\training'
    routes = ''.join([path,'\\','routes (table 4).csv'])
    vec_pathf = ''.join([path,'\\','trajectories(table 5)_training.csv'])
    
    routedf = pd.read_csv(routes)
    vec = pd.read_csv(vec_pathf,parse_dates=[3])
    parselist = []
    print(vec.describe())
    for i in range(len(vec)):
        travel_seq = vec.loc[i,:]['travel_seq'].split(';')
        travel_seq1 = set([])
        time_seq1 = set([])
        for seq in travel_seq:
            travel_seq1.add(seq.split('#')[0])
            time_seq1.add(seq.split('#')[2])
        i_id = vec.loc[i,:]['intersection_id']
        t_id = vec.loc[i,:]['tollgate_id']
        v_id = vec.loc[i,:]['vehicle_id']
        t_time = vec.loc[i,:]['travel_time']
        (day,hour,m,sec) = time_to_period(vec.loc[i,:]['starting_time'])
        #
        travel_seq1 = ','.join([str(x) for x in travel_seq1])
        time_seq1 = ','.join([str(x) for x in time_seq1])
        parselist.append([day,hour,m,sec,i_id,t_id,v_id,t_time,travel_seq1,time_seq1])
    df = pd.DataFrame(parselist,columns=\
        ['day','hour','min','sec','intersection_id','tollgate_id','vehicle_id','tl','pathseq','tl'])
    df.to_csv(r'e:\task20160424.csv',index=False)


def time_to_period(time):
    '''
    @解析时间，分解为某一天与时间段
    '''
    day = time.dayofyear
    hour = time.hour
    mmt = time.minute
    sec = time.second
    return ((day,hour,mmt,sec))


def plot_test(df,index_list,plot_col):
    '''
    @输入表，按照index分组，每组画图
    '''
    for index in index_list:
        print(index)
        if df[index].count()>0:
            continue
        else:
            print('columns error')
            exit(0)
    def plot_group(group):
        import matplotlib.pyplot as plt
        group[[plot_col]].plot().get_figure().show()
    groups = df.groupby(index_list).apply(plot_group)
    
#cal_vec_routs_time()   
    
#graph()
    
df = pd.read_csv(r'e:\task20160424.csv')
print(df.head())
plot_test(df,['day','hour','intersection_id','tollgate_id'],'tl')
