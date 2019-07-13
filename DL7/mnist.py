import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import tensorflow.contrib.slim as slim
from utils import *


import sklearn.metrics as metrics

import time

import scipy.misc

tf.app.flags.DEFINE_boolean('test', False, 'train or test')
FLAGS = tf.app.flags.FLAGS

train = FLAGS.test

iterations = 20000
lr = 0.1
batch_size = 64
Q = tf.constant([5.0])

# 用于判断的距离阈值
thresh = 2.50

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def fully_connected(input_image, shape, scope_name, reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        fc_W = tf.get_variable("W", initializer=tf.truncated_normal(shape,stddev=0.1))
        fc_b = tf.get_variable("B", initializer=tf.zeros(shape[1]))
        fc = tf.matmul(input_image, fc_W) + fc_b
        tf.summary.histogram("weights", fc_W)
        tf.summary.histogram("biases", fc_b)
        return fc

def G(X, name="G"):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        fc0 = tf.nn.relu(fully_connected(input_image=X, shape=(784,500), scope_name='hidden_layer1'))
        # fc1 = fully_connected(input_image=fc0, shape=(500,10), scope_name='hidden_layer2')
        fc1 = tf.nn.relu(fully_connected(input_image=fc0, shape=(500,10), scope_name='hidden_layer2'))
        return fc1



mnist = input_data.read_data_sets('./data/mnist')

# 定义回归模型
x1 = tf.placeholder(tf.float32, [None, 784],name="x1")
x2 = tf.placeholder(tf.float32, [None, 784],name="x2")

y1 = tf.placeholder(tf.float32,[None, 10],name="y1")
y2 = tf.placeholder(tf.float32,[None, 10],name="y2")

y = 1 - tf.cast(tf.equal(tf.argmax(y1,1),tf.argmax(y2,1)),tf.float32)

G1 = G(x1)
G2 = G(x2)


Ew = tf.sqrt(tf.reduce_sum(tf.square(G1 - G2),1))
# 定义损失函数
L1 = 2 * (1 - y) *  tf.square(Ew) / Q
L2 = 2 * y * tf.exp(-2.77 * Ew / Q) * Q
Loss = tf.reduce_mean(L1 + L2)

tf.summary.histogram("Ew", Ew)
tf.summary.histogram("L1", L1)
tf.summary.histogram("L2", L2)
tf.summary.scalar("loss",Loss)

prediction = tf.greater(Ew, thresh)
# 定义准确率
correct_prediction = tf.equal(prediction, tf.cast(y, tf.bool))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.histogram("prediction", tf.cast(prediction,tf.int32))
tf.summary.scalar("accuracy", accuracy)

# 采用Adam作为优化器
train_step = tf.train.GradientDescentOptimizer(lr).minimize(Loss)
# 定义保存器
saver = tf.train.Saver(max_to_keep=4)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
show_all_variables()

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("LOGDIR/", sess.graph)

# 获取测试集
test_img1, test_img2, test_label, test_lab1, test_lab2 = getTestImage(one_hot=True)

if train == False:
	pre1_time = 0
	pre2_time = 0
	run_time = 0.0
	test_time = 0.0
	for ii in range(iterations):
		# start = time.time()
		batch_xs, batch_ys = mnist.train.next_batch(320)
		batch_x1, batch_x2, batch_ys,ys1,ys2 = balanced_batch(batch_xs, batch_ys, 10)
		ys1 = np.reshape(ys1, [-1])
		ys2 = np.reshape(ys2, [-1])

		ys1 = np.array( [ np.eye(10)[i] for i in ys1 ] )
		ys2 = np.array( [ np.eye(10)[i] for i in ys2 ] )

		sess.run(train_step, feed_dict={x1:batch_x1, x2:batch_x2,y1:ys1,y2:ys2})

		if ii % 10 == 0:
			s = sess.run(merged_summary,feed_dict={x1:test_img1, x2:test_img2, y1:test_lab1, y2:test_lab2})
			writer.add_summary(s, ii)
		# end = time.time()
		# run_time += end-start
		if ii % 1000 == 999:
			# print(run_time)
			# run_time = 0.0
			loss_v,acc,pre,yt= sess.run([Loss,accuracy,prediction,y], \
					feed_dict={x1:test_img1, x2:test_img2, y1:test_lab1, y2:test_lab2})
			saver.save(sess, './ckpt/mnist.ckpt')

			# print("--------------------------------")
			print("iteration %6d, loss %4.4f, test acc %4.4f"%(ii+1,loss_v,acc))
			# print("--------------------------------")
else:

	saver.restore(sess,'./ckpt/mnist.ckpt')
	e_w = sess.run(Ew, feed_dict={x1:test_img1, x2:test_img2, y1:test_lab1, y2:test_lab2})

	fig = plt.figure()
	threshs = [0.2, 0.5, 0.7 ,1., 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,5.0]
	for thresh in threshs:
		predictions = (e_w > thresh).astype(np.int8)
		labels = sess.run(y, feed_dict={y1:test_lab1, y2:test_lab2})
		#     labels = -(y_batch1.argmax(1) == y_batch2.argmax(1)).astype(np.float32) +1
		precision, recall, th = metrics.precision_recall_curve(labels, predictions)
		plt.plot(precision, recall, linewidth=1.0,label='thresh='+str(thresh))
		#plt.annotate("c="+str(c),xy=(0.5,-1+p))
	plt.plot([0.5,1], [0.5,1], linewidth=1.0,label='equal')
	plt.title("precision and recall curve")
	plt.legend()
	plt.xlabel("precision")
	plt.ylabel('recall')
	plt.show()
