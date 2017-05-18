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

in_file=r'E:\大数据实践\天池大赛KDD CUP\data\dataSets\training\volume(table 6)_training.csv'
model = model1()
df = model.load_volume_hour(fdir=in_file)
df = prep(df,pat=True,normalize=False)
t_seq = pd.date_range(start='09/19/2016',end='10/17/2016',freq='20Min')
t_seq11 = pd.date_range(start='09/19/2016',end='09/28/2016',freq='20Min')
t_seq12 = pd.date_range(start='09/29/2016',end='09/30/2016',freq='20Min')
t_seq2 = pd.date_range(start='10/1/2016',end='10/7/2016',freq='20Min')
t_seq3 = pd.date_range(start='10/8/2016',end='10/17/2016',freq='20Min')
test_seq = pd.date_range(start='09/19/2016',periods=20,freq='T')

xs = tf.placeholder(tf.float32,[None,10]) # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32,[None,2])
batch_size = tf.placeholder(tf.float32,name='batch_size')

middle = 1000
w_1 = tf.Variable(tf.truncated_normal([10, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 2]))
b_2 = tf.Variable(tf.truncated_normal([1, 2]))
def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
    
z_1 = tf.add(tf.matmul(xs, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)

diff = tf.subtract(a_2, ys)

def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))

#backprogation and update the network
#如何修改误差公式？
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
#使用均方误差函数
#loss = tf.reduce_sum(tf.pow(a_2 - ys,2))/(batch_size)
#loss = tf.reduce_mean(tf.squared_difference(a_2, ys))
#loss = tf.nn.l2_loss(a_2 - ys)

#other loss
#loss = -tf.reduce_sum(ys * tf.log(a_2))
loss = tf.Print(loss, [loss], "cost") #print to the console 
step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(ys, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

batch_x,batch_y = get_all_batch(df,test_seq,k=1)
x_data = np.array(batch_x)
if np.any(np.isnan(x_data)):
    print('nan found ')
    sys.exit(0)
y_data = np.array(batch_y)
b_size = len(x_data)
print(x_data.shape,y_data.shape)
for i in range(1000):
    sess.run(step, feed_dict = {xs: x_data,
                                ys : y_data,
                                batch_size: b_size})
    if i % 100 == 0:
        res = sess.run(acct_res, feed_dict =
                               {xs: x_data,
                                ys : y_data})
        print (res,' t' ,len(y_data))
test_x,test_y = get_all_batch(df,t_seq12,k=1)
test_x = np.array(test_x)
test_y = np.array(test_y)
predict = sess.run(a_2,feed_dict={xs:test_x})
print(predict)
#print(sess.run(predict))
print('true')
print(test_y)
