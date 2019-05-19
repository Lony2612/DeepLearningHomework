import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.platform import gfile

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
# data =  tf.placeholder(tf.float32, [None, 3])

# real_label = tf.placeholder(tf.float32, [None,1])

# ### the weight of the function ####
# weight =  tf.Variable(tf.random_normal([3,1]), dtype=tf.float32, name='weight')
# bias = tf.Variable(tf.ones([1]), dtype=tf.float32, name='bias')

# ### definition of the function ###
# y_label = tf.add(tf.matmul(data, weight), bias)

# ### definition of the loss ###
# loss = tf.reduce_mean(tf.square(real_label-y_label))

# ### definition of the optimizer ###
# train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

###### train of the function #####

##############################

with gfile.FastGFile('./pb/pb.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')
    with tf.Session() as sess:
        sess.graph.as_default()
        sess.run(tf.global_variables_initializer())
        weight = sess.graph.get_tensor_by_name('weight:0')
        bias = sess.graph.get_tensor_by_name('bias:0')
        data = sess.graph.get_tensor_by_name('data:0')

        y_label = sess.graph.get_tensor_by_name('sum_op:0')

        a = sess.run(weight)
        b = sess.run(bias)
        print('the weight is [%f,%f,%f], the bias is %f'%(a[0],a[1],a[2],b))

        forecast_set = sess.run(y_label, feed_dict={data:x})
    
plt.plot(line, forecast_set, 'g:', linewidth=1)

plt.show()

print(len(line))