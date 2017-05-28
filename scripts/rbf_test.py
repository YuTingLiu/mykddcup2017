# -*- coding: utf-8 -*-
"""
Created on Tue May 16 22:57:10 2017

@author: L
@RBF neural network

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


class rbf_net:
    '''
    '''
    
    def __init__(self,D=None,K=None):
        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()
        self.middle = 30
        self.training_iters = 1000
        self.modeldir = os.path.join('../model/')
        
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
                
            def rbf_kernel(input_x):
                gamma = tf.constant(-25.)
                sq_dists = tf.multiply(2, tf.matmul(input_x,tf.transpose(input_x)))
                return tf.exp(tf.multiply(gamma,tf.abs(sq_dists)))


            #z_1 = tf.add(tf.matmul(self.x, self.w_1), self.b_1)
            z_1 = tf.matmul(self.x, self.w_1)
            #a_1 = sigma(z_1)
            a_1 = rbf_kernel(z_1)
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
            self.saver.save(self.session,save_path=os.path.join(self.modeldir,'bp.model'), global_step=step)
            #test
            pred = self.session.run(a_2 , feed_dict={self.x:Xtest})
            print(pred)
            print(Ytest)


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
    
def gen_df(freq='T',normalize = False):
    '''
    help fun
    '''
    if freq is '20Min':
        df = load_volume(fdir='training_20min_avg_volume.csv')
        train_seq = produce_seq(start='09/19/2016 00:20',periods=6,freq='20Min',days = 1)
        test_seq = produce_seq(start='09/19/2016 00:20',periods=6,freq='20Min',days = 1)
        df = prep(df,pat=True,normalize=normalize)
    elif freq is 'T':
        in_file=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\training\volume(table 6)_training.csv'
        model = model1()
        df = model.load_volume_hour(fdir=in_file)
        train_seq = produce_seq(start='09/19/2016 00:20',periods=20,freq='T',days = 7)
        test_seq = produce_seq(start='09/26/2016 00:20',periods=20,freq='T',days = 1)
        df = prep(df,pat=True,normalize=normalize)
    else:
        sys.exit(0)
    return df,train_seq,test_seq


def produce_seq(start='09/19/2016 00:20',periods=20,freq='T',days = 7):
    '''
    help fun to produce time sequence
    '''
    seq = pd.date_range(start=start,periods=periods,freq=freq)
    seq1 = seq
    for day in  range(days):
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
    pred = np.exp(pred)
    y_true = np.exp(y_true)
    print(pred.shape)
    print(pred[:,0])
    print(pred[:,0].flatten().shape)
    plt.plot(pred[:,0].flatten(),'blue')
    plt.plot(pred[:,1].flatten(),'red')
    plt.plot(y_true[:,0].flatten(),'*')
    plt.plot(y_true[:,1].flatten(),'+')
    plt.show()
    
    
    
###############################################################################
def train(freq = '20Min'):
    df,train_seq,test_seq = gen_df(freq)
    
    batch_x,batch_y = get_all_batch(df,train_seq,k=1)
    x_data = np.array(batch_x)
    if np.any(np.isnan(x_data)):
        print('nan found ')
        sys.exit(0)
    y_data = np.array(batch_y)
    
    test_x,test_y = get_all_batch(df,test_seq,k=1)
    test_x = np.array(test_x)
    test_y = np.array(test_y)    
    
    #begin bp train
    bp = bp_net()
    N,D = x_data.shape
    K = len(y_data[0])
    bp.middle = 100
    bp.training_iters = 5000
    bp.build(D,K)
    bp.fit(x_data,y_data,test_x,test_y)
    print(train_seq)
def test(freq = '20Min'):
    df,train_seq,test_seq = gen_df(freq,normalize=True)
    #模型训练了一天的这个时间段，看看能够预测多少天的数据
    for i in range(7):
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
        plot_pred(pred,batch_y)
    print(test_seq)

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
        
        
        
        
        
        
        
if __name__ == '__main__':
#    train(freq='T')
#    test(freq='T')
#    train(freq='20Min')
    test(freq='20Min')
#    next_train(freq='T')
#    next_test(freq='T')






    '''
    输入点为当前时间t
    输出为需要预测的时间t+T
    
    所以预测8：20的流量 ，需要将时间点设置为8:00
    
    '''
    
