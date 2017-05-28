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
    df.loc[:,'mape'] = np.abs((df[df.columns[0]]-df[df.columns[1]])/df[df.columns[0]])
    print(df[df['mape']>0.5][[df.columns[0],df.columns[1]]])
    return df['mape'].sum()/len(df.index)



def main():
    df = load_volume(fdir=r'../arima_result.csv')
    df = df.set_index(['tollgate_id','direction','time_window_s'])
    df.loc[:,'mape'] = np.abs((df[df.columns[0]]-df[df.columns[1]])/df[df.columns[0]])
    print(df[df['mape']>0.5].reset_index()[['time_window_s','direction','volume','pred']])
    result = df['mape'].sum()/len(df.index)
    print('MAPE for this model is ',result)    


if  __name__ == '__main__':
    main()