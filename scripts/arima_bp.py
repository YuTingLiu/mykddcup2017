# -*- coding: utf-8 -*-
"""
Created on Mon May 22 23:02:20 2017

@author: L
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 16 22:57:10 2017
@author: L
@BP neural network
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from data_util import *
import sys
import os

#时间平移
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Week
from pandas.tseries.offsets import Day
from pandas.tseries.offsets import Minute

import matplotlib.pyplot as plt


class bp_net:
    '''
    '''
    
    def __init__(self,D=None,K=None):
        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()
        self.middle = 30
        self.training_iters = 1000
        self.modeldir = os.path.join('../model/')
        self.modelname = 'arima'
        
    def build(self,D,K):
        with self.graph.as_default():
            #1.create pleaceholder
            self.x = tf.placeholder(tf.float32,shape=[None,D],name='x')
            self.y_true = tf.placeholder(tf.float32,shape=[None,K],name='y')
            self.batch_size = tf.placeholder(tf.float32,name='batch_size')
            
            self.global_step = tf.Variable(initial_value=0,name='global_step', trainable=False)
             #定义隐层参数，层数为middle
            self.w_1 = tf.Variable(tf.truncated_normal([D, self.middle]),name='W1')
            self.b_1 = tf.Variable(tf.truncated_normal([1, self.middle]),name='B1')
            #定义输出层参数
            self.w_2 = tf.Variable(tf.truncated_normal([self.middle, K]),name='W2')
            self.b_2 = tf.Variable(tf.truncated_normal([1, K]),name='B2')
            
            self.saver = tf.train.Saver()
            # Create a TensorFlow session for executing the graph.
            
            self.session = tf.Session(graph=self.graph)
        
    def fit(self,_X,_Y,Xtest,Ytest):
        N,D = _X.shape
        K = len(_Y[0])
        
        self.build(D,K)
        training_iters = self.training_iters
        
        #build net       
        with self.graph.as_default():####默认图与自定义图的关系
            def sigma(input_x):
                return tf.div(tf.constant(1.0),
                              tf.add(tf.constant(1.0), tf.exp(tf.negative(input_x))))
                
            z_1 = tf.add(tf.matmul(self.x, self.w_1), self.b_1)
            a_1 = sigma(z_1)
            a_2 = tf.add(tf.matmul(a_1, self.w_2), self.b_2,name='output') #linear output !!!!
            
            diff = tf.subtract(a_2, self.y_true)
            
            #backprogation and update the network
            #如何修改误差公式？   仍然需要优化
            #https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
            #使用均方误差函数
#            loss = tf.reduce_sum(tf.pow(a_2 - self.y_true,2))/(self.batch_size)
#            loss = tf.reduce_mean(tf.squared_difference(a_2, self.y_true))
            loss = tf.nn.l2_loss(a_2 - self.y_true)
            
            #other loss
#            loss = -tf.reduce_sum(self.y_true * tf.log(a_2))
#            loss = tf.Print(loss, [loss], "cost") #print to the console 
            #step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss,self.global_step)
            
            #计算训练样本准确率
            acct_mat = tf.equal(self.y_true,tf.round(a_2))
            acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
            init = tf.global_variables_initializer()
        
            self.session.run(init)
        
            step = 1
            while step  < training_iters:
                # Run the optimizer using this batch of training data.
                # TensorFlow assigns the variables in feed_dict_train
                # to the placeholder variables and then runs the optimizer.
                # We also want to retrieve the global_step counter.
                self.session.run(optimizer , feed_dict={self.x: _X, 
                                                self.y_true: _Y,
                                                self.batch_size : N})
                if step % 100 == 0:
                    # 计算精度
                    acc = self.session.run(acct_res, feed_dict={self.x: _X, 
                                                        self.y_true: _Y,
                                                        self.batch_size : N})
                    # 计算损失值
                    cost = self.session.run(loss, feed_dict={self.x: _X, 
                                                     self.y_true: _Y,
                                                     self.batch_size : N})
                    print("Iter " + str(step) + ", Minibatch Loss= " +\
                    "{:.6f}".format(cost) + ", Training Accuracy= " + "{:.5f}".format(acc))
                step += 1
            # 如果准确率大于50%,保存模型,完成训练
            self.saver.save(self.session,save_path=os.path.join(self.modeldir,self.modelname), global_step=step)
            #test
            pred = self.session.run(a_2 , feed_dict={self.x:Xtest})
            print(pred)
            print(Ytest)
            print('mse is ',np.mean((pred-Ytest)**2))


    def predict(self,_X):
#        N,D = _X.shape
#        self.build(D,10,64)####wrong 与模型中tensor冲突
        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)
        
#        if self.session is 
        with self.graph.as_default():####默认图与自定义图的关系
#            self.session.run(tf.global_variables_initializer())#>0.11rc 更新了模型保存方式
            ckpt = tf.train.get_checkpoint_state(self.modeldir)
            if ckpt and ckpt.model_checkpoint_path:
                print(''.join([ckpt.model_checkpoint_path,'.meta']))
                self.saver = tf.train.import_meta_graph(''.join([ckpt.model_checkpoint_path,'.meta']))
                self.saver.restore(self.session,ckpt.model_checkpoint_path)
    #        self.saver.restore(self.session,self.savefile)
            #print all variable
#            for op in self.graph.get_operations():
#                print(op.name, " " ,op.type)
            #返回模型中的tensor
#            layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
#            layers = [op.name for op in self.graph.get_operations()]
#            feature_nums = [int(self.graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
#            for feature in feature_nums:
#                print(feature)
            
            '''restore tensor from model'''
            w_out = self.graph.get_tensor_by_name('W2:0')
            b_out = self.graph.get_tensor_by_name('B2:0')
            _input = self.graph.get_tensor_by_name('x:0')
            _input1 = self.graph.get_tensor_by_name('x_1:0')
            _out = self.graph.get_tensor_by_name('y:0')
            y_pre_cls = self.graph.get_tensor_by_name('output:0')
#            keep_prob = self.graph.get_tensor_by_name('Placeholder:0')#找到这个未命名的tensor
#            print(y_pre_cls)
#            print(_input)
#            print(keep_prob)
#            print(dsdfs)
#            self.session.run(tf.global_variables_initializer())   ####非常重要，不能添加这一句
            pred = self.session.run(y_pre_cls,feed_dict={_input:_X,
                                                         _input1 : _X})
            return pred
    

    def train1(self,_X,_Y,Xtest,Ytest):
        N,D = _X.shape
#        self.build(D,10,64)####wrong 与模型中tensor冲突
        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)
        
#        if self.session is 
        with self.graph.as_default():####默认图与自定义图的关系
#            self.session.run(tf.global_variables_initializer())#>0.11rc 更新了模型保存方式
            ckpt = tf.train.get_checkpoint_state(self.modeldir)
            if ckpt and ckpt.model_checkpoint_path:
                print(''.join([ckpt.model_checkpoint_path,'.meta']))
                self.saver = tf.train.import_meta_graph(''.join([ckpt.model_checkpoint_path,'.meta']))
                self.saver.restore(self.session,ckpt.model_checkpoint_path)
            '''restore tensor from model'''
            _input = self.graph.get_tensor_by_name('x:0')
            _input1 = self.graph.get_tensor_by_name('x_1:0')
            _out = self.graph.get_tensor_by_name('y:0')
            _out1 = self.graph.get_tensor_by_name('y_1:0')
            y_pre_cls = self.graph.get_tensor_by_name('output:0')
            batch_size = self.graph.get_tensor_by_name('batch_size:0')
            global_step = self.graph.get_tensor_by_name('global_step:0')
            
            #define optimizer
            loss = tf.nn.l2_loss(y_pre_cls - _out1)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss,global_step)
            #计算训练样本准确率
            acct_mat = tf.equal(_out1,tf.round(y_pre_cls))
            acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
            init = tf.global_variables_initializer()
        
            self.session.run(init)
            
            step = 1
            while step  < self.training_iters:
                # Run the optimizer using this batch of training data.
                # TensorFlow assigns the variables in feed_dict_train
                # to the placeholder variables and then runs the optimizer.
                # We also want to retrieve the global_step counter.
                self.session.run(optimizer , feed_dict={_input: _X, 
                                                        _input1:_X,
                                                        _out : _Y,
                                                        _out1 : _Y})
                if step % 100 == 0:
                    # 计算精度
                    acc = self.session.run(acct_res, feed_dict={_input: _X, 
                                                        _input1:_X,
                                                        _out : _Y,
                                                        _out1 : _Y})
                    # 计算损失值
                    cost = self.session.run(loss, feed_dict={_input: _X, 
                                                        _input1:_X,
                                                        _out : _Y,
                                                        _out1 : _Y})
                    print("Iter " + str(step) + ", Minibatch Loss= " +\
                    "{:.6f}".format(cost) + ", Training Accuracy= " + "{:.5f}".format(acc))
                step += 1
            # 如果准确率大于50%,保存模型,完成训练
            self.saver.save(self.session,save_path=os.path.join(self.modeldir,'bp.model'), global_step=step)
            #test
            pred = self.session.run(y_pre_cls,feed_dict={_input:_X,
                                                         _input1 : _X})
            print(np.sum(pred,axis=0))
            print(np.sum(Ytest,axis=0))
    
    
    def score(self ,_X,_Y):
        return 1-error_rate(self.predict(_X),_Y)
################################################################################    
    
def gen_df(freq='T',normalize = False,test = False,pat = True,periods=72):
    '''
    help fun
    '''
    if freq is '20Min':
        df = load_volume(fdir='train_union.csv')
        train_seq = produce_seq(start='10/8/2016 00:00',periods=periods,freq='20Min',days = 1)
        if len(train_seq) != 72*20:
            print('train_seq len ',len(train_seq))
#            sys.exit()
        if test is True:
            df = load_test(fdir='test_union.csv')
            test_seq = produce_seq(start='10/18/2016 07:20',periods=1,freq='20Min',days = 1)
            df = prep(df,pat=pat,normalize=normalize)
        else:
            test_seq = produce_seq(start='10/16/2016 06:00',periods=periods,freq='20Min',days = 1)
            df = prep(df,pat=pat,normalize=normalize)
    else:
        sys.exit(0)
    return df,train_seq,test_seq


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

def plot_pred(pred,y_true):
    '''
    input :pred,true
    datashape : [[1,2],[1,2]]
    
    '''    
    plt.plot(pred[:,0].flatten(),'blue')
    plt.plot(y_true[:,0].flatten(),'*')
    plt.show()
    

def next_seq(seq,offset="20Min"):
    return seq  + Minute(offset)
    
###############################################################################
def pre_train(freq = '20Min'):
    df,train_seq,test_seq = gen_df(freq,normalize = False,test=False)
    tolls = df.groupby('tollgate_id')
    for t,tgroup in tolls:
        ds = tgroup.groupby('direction')
        for d,dgroup in ds:
            dgroup.loc[:,'day'] = dgroup['time_window_s'].dt.dayofyear
            days = dgroup.groupby('day')
            for day,group in days:
                group = group.set_index('time_window_s')['volume']
                param = ''.join([str(t),'-',str(d),'-',str(day)])
                print('pre train param for ',param)
                from time_series_analysis_p1 import ARIMA_predictBynum
                pqr = ARIMA_predictBynum(group,[0,1,0])
                GLOBAL[param] = pqr
    cache(GLOBAL)
		
def train(freq = '20Min',tollgate_id = 1):
    '''
    主要的内容是:数据按照时间分割,送入ARIMA训练,得到结果1
	      整合结果1,送入BPNN训练,得到训练好的网络
         输出:每个tollgate一个模型
    '''
    df,train_seq,test_seq = gen_df(freq,normalize = False,test=False)
    print(df.dtypes)
    step = len(train_seq) #预测前进X步,使得预测长度与输入长度相等
    batch_x = df[df['tollgate_id']==tollgate_id]
    batch_x = batch_x.set_index(['tollgate_id','direction','time_window_s'])
    df1 = df[df['tollgate_id']==tollgate_id].set_index('time_window_s')
    df1 = df1.groupby('direction')
    batch = []
    for direction,group in df1:
        arima_x = group['volume']
        if len(arima_x[arima_x.isnull()])>0 or len(arima_x[arima_x == np.nan])>0 or len(arima_x[arima_x == np.inf]):
            print(arima_x[arima_x.isnull()])
            print(arima_x[arima == np.nan])
            print(arima_x[arima == np.inf])
            sys.exit()
#        print(arima_x)
#        sys.exit()
        print('ARIMA模型,训练8天的数据,默认时间序列周期为一天,通过每天建立模型,得到下一天的输出,最终形成8天8*72个预测值,放到BP中训练')
        from time_series_analysis_p1 import main_1 as arima
        #output: tollgate_id,direction time_window_s,pred,volume
        outputlist = []
        for day in range(8):
            #p,q load
            param_name = ''.join([str(tollgate_id),'-',str(direction),'-',str(train_seq[0].dayofyear)])
            pqr = GLOBAL[param_name]
            output = arima(arima_x,tollgate_id,direction,train_seq,step,pqr)
            print('输出长度应该与预期长度不一致,是否缺少一位 (0:一致,-1:少一位):' ,(len(output)-len(train_seq)))
            train_seq += Day(1)
            step = len(train_seq)
            if len(output) == len(train_seq):
                print('train seq match output length')
            else:
                print(len(output))
                print(output)
                sys.exit()
                
            outputlist.append(output)
#        print(output)
        #add weather param
        output = pd.concat(outputlist)
        print(len(output))
        output = output.set_index(['tollgate_id','direction','time_window_s'])
        print(output.loc[:,'pred'])
        print(len(output))
#        sys.exit()
        batch_x.loc[:,'pred'] = output.loc[:,'pred']
    
    batch_x = batch_x[batch_x['pred'].notnull()]
#    print(batch_x)
    #normalize
    batch_x.loc[:,'residual'] = batch_x.loc[:,'volume'] - batch_x.loc[:,'pred']#添加残差
    batch_x.loc[:,'pred'] = np.log(batch_x.loc[:,'pred'])
    batch_x.loc[:,'volume'] = np.log(batch_x.loc[:,'volume'])

    test_seq = batch_x.index
    batch_x.to_csv(''.join([str(tollgate_id),r'_arima_bp_log.csv']),index=True)
    #struct batch
    x_data = np.array(batch_x.reset_index()[['pressure','sea_pressure','wind_direction','wind_speed',
                                'temperature','rel_humidity','precipitation',
                                'dayofweek','hour','minute','direction','pattern','pred']])
    if np.any(np.isnan(x_data)):
        print('nan found ')
        sys.exit(0)
    y_data = np.array(batch_x.loc[:,'residual'])#修改为残差
    y_data = y_data.reshape((len(y_data),1))
#    print(x_data)
#    print(y_data)
    
    #begin bp train
    bp = bp_net()
    N,D = x_data.shape
    K = len(y_data[0])
    bp.middle = 150
    bp.modelname = str(tollgate_id)
    bp.training_iters = 25000
    bp.build(D,K)
    bp.fit(x_data,y_data,x_data,y_data)
#    print(train_seq)
    print('train',x_data.shape,y_data.shape)
    print('train seq ',train_seq[0],train_seq[-1])
    print('test seq',test_seq[0],test_seq[-1])
    
    
    
def test(freq = '20Min',tollgate_id = 1):
    '''
    model test
    output result
    '''
    result = []
    periods = 72
    df,train_seq,test_seq = gen_df(freq,normalize=False,test=False,pat=True,periods=periods)
    true = df[df['tollgate_id']==tollgate_id].set_index(['tollgate_id','direction','time_window_s'])
    
    #add ARIMA process
    print('训练时间段向前推')
    train_seq = pd.date_range(start = (test_seq[0]-Minute(20*periods)),periods=len(test_seq),freq='20Min')
    al_dir = []
    for direction,group in df[df['tollgate_id']==tollgate_id].groupby('direction'):
        arima_x = group.set_index(['time_window_s'])['volume'][train_seq]
        print('arima_x couting number of zero', len(arima_x[arima_x==0]))
        from time_series_analysis_p1 import main_1 as arima
        #output tollgate_id,direction time_window_s,pred,volume
        param_name = ''.join([str(tollgate_id),'-',str(direction),'-',str(train_seq[0].dayofyear)])
        pqr = GLOBAL[param_name]
        output = arima(arima_x,tollgate_id,direction,train_seq,len(test_seq),pqr)
        output = output.set_index(['tollgate_id','direction','time_window_s'])
        al_dir.append(output)

    true = pd.concat(al_dir)
    true.to_csv(''.join([str(tollgate_id),'-','arima_tmp.csv']))
    
    true.loc[:,'residual'] = true.loc[:,'volume'] - true.loc[:,'pred']#添加残差
    true.loc[:,'volume'] = np.log(true.loc[:,'volume'])
    true.loc[:,'pred'] = np.log(true.loc[:,'pred'])
    
    #模型训练了一天的这个时间段，看看能够预测多少天的数据
    for i in range(1):
        if i>0:
            test_seq = test_seq + DateOffset(days=1)
        #struct batch
        print(true)
#        sys.exit(0)
        x_data = np.array(true[['pressure','sea_pressure','wind_direction','wind_speed',
                                    'temperature','rel_humidity','precipitation',
                                    'dayofweek','hour','minute','direction','pattern','pred']])
        if np.any(np.isnan(x_data)):
            print('nan found ')
            sys.exit(0)
        y_data = np.array(true.loc[:,'residual'])
        y_data = y_data.reshape((len(y_data),1))
        print(x_data)
        print('test seq is',test_seq[0],test_seq[-1],test_seq[0].dayofweek)
        
        #begin test this model
        model = bp_net()
        pred = model.predict(x_data)
        print(np.sum(pred,axis=0))
        print(np.sum(y_data,axis=0))
        print(len(test_seq),pred.shape,y_data.shape)
        
        #plot pred and true data
        plot_pred(pred,y_data)
        print('mse is ',np.mean((pred-y_data)**2))
        temp = np.concatenate((y_data,pred),axis = 1)
        print(temp)
        print(temp.shape)
        
        temp = pd.DataFrame(temp,columns=['volume','pred'],index=test_seq)
        result.append(temp)

    df = pd.concat(result)
    df.to_csv(''.join([str(tollgate_id),'model_pre.csv']))
    
def next_train(freq = 'T'):
    df,train_seq,test_seq = gen_df(freq)
    train_seq = next_20min(train_seq,m=20)
    test_seq = next_20min(test_seq,m=20)
    
    batch_x,batch_y = get_all_batch(df,train_seq,k=1)
    x_data = np.array(batch_x)
    if np.any(np.isnan(x_data)):
        print('nan found ')
        sys.exit(0)
    y_data = np.array(batch_y)
    
    test_x,test_y = get_all_batch(df,test_seq,k=1)
    test_x = np.array(test_x)
    test_y = np.array(test_y)    
    
    
    #begin next train 20minute
    model = bp_net()
    model.training_iters = 500
    model.train1(x_data,y_data,test_x,test_y)
        
def next_test(freq = '20Min'):
    df,train_seq,test_seq = gen_df(freq)
    train_seq = next_20min(train_seq,m=20)
    test_seq = next_20min(test_seq,m=20)
    #模型训练了一天的这个时间段，看看能够预测多少天的数据
    for i in range(15):
        test_seq = test_seq + DateOffset(days=1)
        print(test_seq[0])
        batch_x,batch_y = get_all_batch(df,test_seq,k=1)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        #begin test this model
        model = bp_net()
        pred = model.predict(batch_x)
        print(np.sum(pred,axis=0))
        print(np.sum(batch_y,axis=0))        
        
        
        
        
        
        
def cache(config,fdir='global_config.txt'):
    if os.path.isfile(fdir):
        f = open(fdir,'r')
        config = json.load(f)
        f.close()
        return config
    else:
        f = open(fdir,'w')
#        json.dump(config,f)
        f.close()

GLOBAL = {}
if __name__ == '__main__':
    GLOBAL = cache(GLOBAL)
    print(GLOBAL)
    train(freq='20Min',tollgate_id=1)
#    test(freq='20Min',tollgate_id=1)






    '''
    输入点为当前时间t
    输出为需要预测的时间t+T
    
    所以预测8：20的流量 ，需要将时间点设置为8:00
    
    '''