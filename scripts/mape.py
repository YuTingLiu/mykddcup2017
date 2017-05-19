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


def mape(df):
    T = 84
    C = 5
    unit_p = lambda x : np.abs((x['volume']-x['pred'])/x['volume'])
    df = df.groupby(['tollgate_id','direction','time_window_s'])[['volume','pred']].apply(unit_p)

    df = df.reset_index().groupby(['tollgate_id','direction']).agg(sum)
    df = df/T
    return df.sum()/C



def main():
    df = load_volume()
    result = mape(df)
    print(result)    
