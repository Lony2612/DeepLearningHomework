import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util


from tensorflow.examples.tutorials.mnist import input_data

iterations = 20000
lr = 0.01
batch_size = 64

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def X_W(X):
    with tf.variable_scope("X_W", reuse=tf.AUTO_REUSE) as scope:
        W = tf.get_variable('W', [392, 10])
        return tf.matmul(X,W)

# plt.axis([0, 100, 0, 1])

###############
plt.ion()

xs = [0, 0]
ys = [0, 0]
###############

mnist = input_data.read_data_sets('./data/mnist',one_hot=True)
# print(mnist.validation.num_examples)
# print(mnist.train.num_examples)
# print(mnist.test.num_examples)

# 定义回归模型
x1 = tf.placeholder(tf.float32, [None, 392])
x2 = tf.placeholder(tf.float32, [None, 392])
# W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]), name="b")
y = tf.nn.softmax(X_W(x1)+X_W(x2)+b) #预测值

#定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])

# 交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# 采用SGD作为优化器
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
show_all_variables()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for ii in range(iterations):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#     print(batch_xs.shape)
    # batch_xs = batch_xs.reshape([64,-1])
    sess.run(train_step, feed_dict={x1: batch_xs[:,0:392], x2: batch_xs[:,392:784],y_:batch_ys})
    if ii % 100 == 1:
        acc, los = sess.run([accuracy, cross_entropy], feed_dict={x1:mnist.test.images[:,0:392],x2:mnist.test.images[:,392:784],y_:mnist.test.labels})
        print("Iteration [%5d/%5d]: accuracy is: %4f loss is: %4f"%(ii,iterations,acc,los))
        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = ii
        ys[1] = acc
        plt.title("Training Accuracy")
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.plot(xs, ys, )
        plt.pause(0.1)


test_acc = sess.run(accuracy, feed_dict={x1:mnist.test.images[:,0:392],x2:mnist.test.images[:,392:784],y_:mnist.test.labels})
print("Test: accuracy is: %4f"%(test_acc))