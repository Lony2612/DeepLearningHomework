import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util


from tensorflow.examples.tutorials.mnist import input_data
from LeNet import LeNet

iterations = 20000
lr = 0.01
batch_size = 64

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# 动态图绘制器
class gif_drawer():
    def __init__(self):
        plt.ion()
        self.xs = [0, 0]
        self.ys = [0, 0]
    def draw(self, update_x, update_y):
        self.xs[0] = self.xs[1]
        self.ys[0] = self.ys[1]
        
        self.xs[1] = update_x
        self.ys[1] = update_y
        
        plt.title("Training Accuracy")
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.plot(self.xs, self.ys, )
        plt.pause(0.1)

# 定义占位符
raw_input_image = tf.placeholder(tf.float32, [None, 32, 32, 1])
raw_input_label = tf.placeholder(tf.float32, [None, 10])

# 调用网络
y = LeNet(raw_input_image)

# 定义损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=raw_input_label))

# 定义正确率
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(raw_input_label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定义优化器
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

gd = gif_drawer()

# 定义超参数
iterations = 20000
lr         = 0.1
batch_size = 64


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

show_all_variables()

with tf.Session(config=run_config) as sess:
    mnist = input_data.read_data_sets('./data/mnist',one_hot=True)
    sess.run(tf.global_variables_initializer())
    test_images = mnist.test.images
    test_images = np.reshape(test_images,[-1,28,28,1])
    test_images = np.pad(test_images,((0,0),(2,2),(2,2),(0,0)), 'constant')
    for ii in range(iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = np.reshape(batch_xs,[-1,28,28,1])
        batch_xs = np.pad(batch_xs, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        # sess.run(train_op, feed_dict={lenet_part.raw_input_image:batch_xs, lenet_part.raw_input_label: batch_ys})
        sess.run(train_op, feed_dict={raw_input_image:batch_xs, raw_input_label: batch_ys})

        if ii % 500 == 1:
            acc, los = sess.run([accuracy, cross_entropy], \
                feed_dict={raw_input_image:test_images,\
                raw_input_label:mnist.test.labels})
            print("Iteration [%5d/%5d]: accuracy is: %4f loss is: %4f"%(ii,iterations,acc,los))
            gd.draw(ii, acc)
    for ii in range(10):
        acc = sess.run(accuracy, \
                feed_dict={raw_input_image:test_images,\
                raw_input_label:mnist.test.labels})
        print("Test: accuracy is %4f"%(acc))
    