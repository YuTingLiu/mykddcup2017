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
import json
import pickle

from time_series_analysis_p1 import ARIMA_predictBynum
from time_series_analysis_p1 import main_1 as arima
from time_series_analysis_p1 import plot_compare

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
            cost = np.inf
            while step  < training_iters:
                # Run the optimizer using this batch of training data.
                # TensorFlow assigns the variables in feed_dict_train
                # to the placeholder variables and then runs the optimizer.
                # We also want to retrieve the global_step counter.
                self.session.run(optimizer , feed_dict={self.x: _X, 
                                                self.y_true: _Y,
                                                self.batch_size : N})
                if step % 5000 == 0:
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
                    if cost > 10:
                        print('delay schedure for 50000')
                        training_iters += 5000
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
                checked = ''.join([ckpt.model_checkpoint_path,'.meta'])
                print(checked)
                param = ''.join([self.modeldir,self.modelname,'-',str(self.training_iters),'.meta'])
                path = ''.join([self.modeldir,self.modelname,'-',str(self.training_iters)])
                if checked is param:
                    self.saver = tf.train.import_meta_graph(checked)
                    self.saver.restore(self.session,ckpt.model_checkpoint_path)
                else:
                    print('load your model',param)
                    try:
                        self.saver = tf.train.import_meta_graph(param)
                        self.saver.restore(self.session,path)
                    except Exception as e:
                        print('load model',e)
                        sys.exit(0)
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
                checked = ''.join([ckpt.model_checkpoint_path,'.meta'])
                print(checked)
                param = ''.join([self.modeldir,self.modelname,'-',str(self.training_iters),'.meta'])
                path = ''.join([self.modeldir,self.modelname,'-',str(self.training_iters)])
                if checked is param:
                    self.saver = tf.train.import_meta_graph(checked)
                    self.saver.restore(self.session,ckpt.model_checkpoint_path)
                else:
                    print('load your model',param)
                    try:
                        self.saver = tf.train.import_meta_graph(param)
                        self.saver.restore(self.session,path)
                    except Exception as e:
                        print('load model',e)
                        sys.exit(0)
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
            cost = np.inf
            while step  < self.training_iters:
                # Run the optimizer using this batch of training data.
                # TensorFlow assigns the variables in feed_dict_train
                # to the placeholder variables and then runs the optimizer.
                # We also want to retrieve the global_step counter.
                self.session.run(optimizer , feed_dict={_input: _X, 
                                                        _input1:_X,
                                                        _out : _Y,
                                                        _out1 : _Y})
                if step % 5000 == 0:
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
                    if cost < 1:
                        break
                step += 1
#            # 如果准确率大于50%,保存模型,完成训练
#            self.saver.save(self.session,save_path=os.path.join(self.modeldir,'bp.model'), global_step=step)
            #test
            pred = self.session.run(y_pre_cls,feed_dict={_input:Xtest,
                                                         _input1 : Xtest})
            print(np.sum(pred,axis=0))
            print(np.sum(Ytest,axis=0))
    
    
    def score(self ,_X,_Y):
        return 1-error_rate(self.predict(_X),_Y)
################################################################################    
    
