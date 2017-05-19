mport tensorflow as tf
from tensorflow.contrib import rnn

batch_size = 4
n_steps = 6
n_input = 1 
n_output = 1
data_length = 16

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_steps, n_output])

def RNN(x):
    x = tf.unstack(x, n_steps, 1)
    # x = tf.transpose(x,[1,0,2])
    # x = tf.reshape(x,[-1,n_input])
    # x = tf.split(x,n_steps)
    lstm_cell = rnn.BasicLSTMCell(n_output, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return outputs

pred = RNN(x)
pred1 = tf.reshape(pred, [n_steps,-1,n_output])
pred2 = tf.transpose(pred1, [1,0,2])


# cost = tf.contrib.losses.sigmoid_cross_entropy(pred2,y)
cost = tf.reduce_mean(tf.square(pred2 - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

# accuracy = tf.reduce_mean(tf.square(tf.subtract(pred2, y)))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        index = np.random.choice(data_length, batch_size, replace=False)
        print 'index',index
        sess.run(optimizer, feed_dict={x:inputx[index], y:inputy[index]})
        print 'cost:',sess.run(cost, feed_dict={x:inputx[index], y:inputy[index]})

