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
batch_x,batch_y = next_batch(df)
INPUT_SHAPE = 10
OUT_CLASS = 2

sess = tf.InteractiveSession()

# a batch of inputs of 2 value each
inputs = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE])

# a batch of output of 1 value each
desired_outputs = tf.placeholder(tf.float32, shape=[None, OUT_CLASS])

# [!] define the number of hidden units in the first layer
HIDDEN_UNITS = 4 

# connect 2 inputs to 3 hidden units
# [!] Initialize weights with random numbers, to make the network learn
weights_1 = tf.Variable(tf.truncated_normal([INPUT_SHAPE, HIDDEN_UNITS]))

# [!] The biases are single values per hidden unit
biases_1 = tf.Variable(tf.zeros([HIDDEN_UNITS]))

# connect 2 inputs to every hidden unit. Add bias
layer_1_outputs = tf.nn.sigmoid(tf.matmul(inputs, weights_1) + biases_1)

# [!] The XOR problem is that the function is not linearly separable
# [!] A MLP (Multi layer perceptron) can learn to separe non linearly separable points ( you can
# think that it will learn hypercurves, not only hyperplanes)
# [!] Lets' add a new layer and change the layer 2 to output more than 1 value

# connect first hidden units to 2 hidden units in the second hidden layer
weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS, 2]))
# [!] The same of above
biases_2 = tf.Variable(tf.zeros([2]))

# connect the hidden units to the second hidden layer
layer_2_outputs = tf.nn.sigmoid(
    tf.matmul(layer_1_outputs, weights_2) + biases_2)

# [!] create the new layer
weights_3 = tf.Variable(tf.truncated_normal([2, 1]))
biases_3 = tf.Variable(tf.zeros([1]))

logits = tf.nn.sigmoid(tf.matmul(layer_2_outputs, weights_3) + biases_3)

# [!] The error function chosen is good for a multiclass classification taks, not for a XOR.
error_function = 0.5 * tf.reduce_sum(tf.sub(logits, desired_outputs) * tf.sub(logits, desired_outputs))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(error_function)

sess.run(tf.initialize_all_variables())

training_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

training_outputs = [[0.0], [1.0], [1.0], [0.0]]

for i in range(20000):
    _, loss = sess.run([train_step, error_function],
                       feed_dict={inputs: np.array(training_inputs),
                                  desired_outputs: np.array(training_outputs)})
    print(loss)


batch_x,batch_y = get_all_batch(df,t_seq,k=1)
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
        print (res)