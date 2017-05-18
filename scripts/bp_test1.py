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

in_file = 'training_20min_avg_volume.csv'
df = load_volume(in_file)
t_seq = pd.date_range(start='09/19/2016',end='10/17/2016',freq='20Min')
t_seq11 = pd.date_range(start='09/19/2016',end='09/28/2016',freq='20Min')
t_seq12 = pd.date_range(start='09/29/2016',end='09/30/2016',freq='20Min')
t_seq2 = pd.date_range(start='10/1/2016',end='10/7/2016',freq='20Min')
t_seq3 = pd.date_range(start='10/8/2016',end='10/17/2016',freq='20Min')



xs = tf.placeholder(tf.float32,[None,11]) # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32,[None,3])

middle = 100
w_1 = tf.Variable(tf.truncated_normal([11, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 3]))
b_2 = tf.Variable(tf.truncated_normal([1, 3]))

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

d_z_2 = tf.multiply(diff, sigmaprime(z_2))
d_b_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(xs), d_z_1)



eta = tf.constant(0.5)
step = [
    tf.assign(w_1,
            tf.subtract(w_1, tf.multiply(eta, d_w_1)))
  , tf.assign(b_1,
            tf.subtract(b_1, tf.multiply(eta,
                               tf.reduce_mean(d_b_1, axis=[0]))))
  , tf.assign(w_2,
            tf.subtract(w_2, tf.multiply(eta, d_w_2)))
  , tf.assign(b_2,
            tf.subtract(b_2, tf.multiply(eta,
                               tf.reduce_mean(d_b_2, axis=[0]))))
]


acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(ys, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


batch_x,batch_y = get_all_batch(df,t_seq11,k=1)
x_data = np.array(batch_x)
y_data = np.array(batch_y)
print(x_data.shape,y_data.shape)
for i in range(10000):
    sess.run(step, feed_dict = {xs: x_data,
                                ys : y_data})
    if i % 100 == 0:
        res = sess.run(acct_res, feed_dict =
                               {xs: x_data,
                                ys : y_data})
        print (res,' t' ,len(y_data))
test_x,test_y = get_all_batch(df,t_seq12,k=1)
predict = sess.run(a_2,feed_dict={xs:test_x})
print(predict)
#print(sess.run(predict))
print('true')
print(test_y)

