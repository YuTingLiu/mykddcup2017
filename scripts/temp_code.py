
        def aggregate(group):
            #rename model's values
            group.loc[:,'model'] = 'model_'+group.loc[:,'model']
            #rename etc's values
            group.loc[:,'is_etc'] = 'etc_'+group.loc[:,'model']
            #pivot_table first
            group.loc[:,'count'] = 1
            group = pd.pivot_table(group,index=['tollgate','direction','time_window_s'],values='count',columns=['model','is_etc'],aggfunc=np.sum,fill_value=0)
            print(group)
            sys.exit()
            group = group.set_index(['tollgate','direction','time_window_s']).resample('20Min').agg(np.sum).fillna(0)
            print(group)
            sys.exit()
        df = df.groupby(['tollgate','direction']).apply(aggregate)
    #    print(df.head(100))
        return df.reset_index()
        
        
        
def filter(ts,method=1,periods=12):
    '''
    添加新的滤波器
    '''
    from scipy.signal import medfilt
    from scipy.signal import wiener
    from scipy.signal import detrend
    
    data = ts.values
    time = ts.index
    
    plt.plot(time,data ,label='origin')
    plt.plot(time,medfilt(data,periods),lw=2,label="Median")
    plt.plot(time,wiener(data,periods),'--',lw=2,label="Wiener")
    plt.plot(time,detrend(data), lw=3 ,label="Detrend")
    plt.xlabel("year")
    plt.grid(True)
    plt.legend()
    plt.show()
