import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

end = (2.0*np.pi - 74)/18
start = -74/18.0

print("采样起点：%f，采样终点：%f"%(start, end))

line = np.linspace(start, end, 2000, dtype=float)

y = np.array(np.cos(line*18+74)).reshape([-1,1])
x = np.array([[ii, pow(ii,2), pow(ii,3)] for ii in line])
x1 = np.array(line).reshape([-1,1])

print(len(x), len(y))

##############################
### definition of the data ###
data =  tf.placeholder(tf.float32, [None, 3], name='data')
data1 = tf.placeholder(tf.float32, [None, 1], name='data1')

real_label = tf.placeholder(tf.float32, [None,1])

### the weight of the function ####
# weight =  tf.Variable(tf.random_normal([3,1], mean=200, stddev=200, seed=10), dtype=tf.float32, name='weight')
# bias = tf.Variable(tf.random_normal([1], mean=1100, stddev=100, seed=10), dtype=tf.float32, name='bias')

weight =  tf.Variable(tf.random_normal([3,1], mean=500, stddev=10), dtype=tf.float32, name='weight')
bias = tf.Variable(tf.random_normal([1], mean=1100, stddev=10), dtype=tf.float32, name='bias')
# weight =  tf.Variable(tf.random_normal([3,1], mean=500, stddev=10), dtype=tf.float32, name='weight')
# bias = tf.Variable(tf.random_normal([1], mean=500, stddev=10), dtype=tf.float32, name='bias')

### definition of the function ###
y_label = tf.add(tf.matmul(data, weight), bias, name='sum_op')

### definition of the loss ###
loss = tf.reduce_mean(tf.square(real_label-y_label))

### definition of the optimizer ###
# train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
train = tf.train.AdamOptimizer(0.2).minimize(loss)

### definition of the saver ###
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
###### train of the function #####

##############################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    show_all_variables()

    idxs = len(x) // 128
    a= sess.run(weight)
    b = sess.run(bias)
    print(a,b)
    # sess.run(tf.assign(weight, [[588.72],[74.77],[0.0000001]]))
    # sess.run(tf.assign(bias, [600]))
    for ii in range(2000000):
        sess.run(train, feed_dict={data:x, real_label:y})
        if ii % 1000 == 0:
            print(sess.run(loss,feed_dict={data:x, real_label:y}), ii)
            print(sess.run(weight), sess.run(bias))
    
    saver.save(sess, './ckpt/my_test', global_step=10000, write_meta_graph=True)

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['sum_op'])
    with tf.gfile.FastGFile('./pb/pb.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # sess.run(tf.assign(weight, [[588.72],[74.77],[0.0000001]]))
    # sess.run(tf.assign(bias, [1158]))
    # sess.run(tf.assign(weight, [[696.597],[122.25],[5.7268]]))
    # sess.run(tf.assign(bias, [1196]))
    forecast_set = sess.run(y_label, feed_dict={data:x})
    a= sess.run(weight)
    b = sess.run(bias)


print(a,b)
plt.plot(line, forecast_set, 'g:', linewidth=1)
plt.plot(line, y, 'r-', linewidth=1)

plt.show()

print(len(line))