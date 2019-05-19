import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

### the weight of the function ####
x = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='x')

### definition of the loss ###
loss = 1 - tf.sin(x)/x

### definition of the optimizer ###
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

###### train of the function #####

##############################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    show_all_variables()

    for ii in range(10000):
        sess.run(train)
        # if ii % 1000 == 0:
        print(sess.run(x), sess.run(loss), ii)

line = np.linspace(-30, 30, 2000, dtype=float)
z = 1-np.sin(line)/line

plt.plot(line, z, 'r-', linewidth=1)

plt.show()