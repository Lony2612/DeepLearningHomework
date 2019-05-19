import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

end = (2.0*np.pi - 74)/18
start = -74/18.0

print("采样起点：%f，采样终点：%f"%(start, end))

line = np.linspace(start, end, 2000, dtype=float)

y = np.array(np.cos(line*18+74)).reshape([-1,1])
x = np.array([[ii, pow(ii,2), pow(ii,3)] for ii in line])

print(len(x), len(y))

plt.plot(line, y, 'r-', linewidth=1)

##############################
### definition of the data ###
data =  tf.placeholder(tf.float32, [None, 3])

real_label = tf.placeholder(tf.float32, [None,1])

### the weight of the function ####
weight =  tf.Variable(tf.random_normal([3,1]), dtype=tf.float32, name='weight')
bias = tf.Variable(tf.ones([1]), dtype=tf.float32, name='bias')

### definition of the function ###
y_label = tf.add(tf.matmul(data, weight), bias)

### definition of the loss ###
loss = tf.reduce_mean(tf.square(real_label-y_label))

### definition of the optimizer ###
train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

### definition of the saver ###
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
###### train of the function #####

##############################
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./ckpt/')
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    print(ckpt_name)

    saver.restore(sess, os.path.join('./ckpt/', ckpt_name))

    a = sess.run(weight)
    b = sess.run(bias)
    print('the weight is [%f,%f,%f], the bias is %f'%(a[0],a[1],a[2],b))

    forecast_set = sess.run(y_label, feed_dict={data:x})
    
plt.plot(line, forecast_set, 'g:', linewidth=1)

plt.show()

print(len(line))