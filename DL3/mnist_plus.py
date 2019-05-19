import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util

from glob import glob
import os

import time

import scipy.misc

from tensorflow.examples.tutorials.mnist import input_data

iterations = 20000
lr = 0.01
batch_size = 64
Q = tf.constant([5.0])
MagicNum = tf.constant([2.0])
One = tf.constant([1.0])
Magic2 = tf.constant([2.77])

train_same_f = open("train_same_f.txt",'r')
train_dif_f  = open("train_dif_f.txt",'r')
test_same_f  = open("test_same_f.txt",'r')
test_dif_f   = open("test_dif_f.txt",'r')

list_tr_s = train_same_f.readlines()
list_tr_d = train_dif_f.readlines()
list_te_s = test_same_f.readlines()
list_te_d = test_dif_f.readlines()

for ii in range(len(list_te_d)):
    list_te_d[ii] = list_te_d[ii].rstrip('\n')
for ii in range(len(list_te_s)):
    list_te_s[ii] = list_te_s[ii].rstrip('\n')
for ii in range(len(list_tr_d)):
    list_tr_d[ii] = list_tr_d[ii].rstrip('\n')
for ii in range(len(list_tr_s)):
    list_tr_s[ii] = list_tr_s[ii].rstrip('\n')

train_dif_f.close()
train_same_f.close()
test_dif_f.close()
test_same_f.close()

def create_labels(batch_size):
    y_labels = []
    for _ in range(batch_size):
        if random.random() < 0.5:
            temp = 0.0
        else:
            temp = 1.0
        y_labels.append(temp)
    return y_labels

def get_image_paths(labels, train=True):
    train_X1 = []
    train_X2 = []

    for ii in range(len(labels)):
        images_path2 = get_image_path(labels[ii], train=True)
        train_X1.append(images_path2[0])
        train_X2.append(images_path2[1])
    return train_X1, train_X2

def get_image_path(label, train=True):
    if train == True:
        if label == 1.0:
            temp = random.sample(list_tr_d, 1)
        else:
            temp = random.sample(list_tr_s, 1)
    else:
        if label == 1.0:
            temp = random.sample(list_te_d, 1)
        else:
            temp = random.sample(list_te_s, 1)
    return temp[0].split()

def get_image(image_path, grayscale=False):
    image = scipy.misc.imread(image_path, flatten = True).astype(np.float)
    return image

def get_next_batch(batch_size, train=True):
    y_labels = create_labels(batch_size)
    dict_X1, dict_X2 = get_image_paths(y_labels, train=True)
    X_1 = [get_image(image_path, grayscale=True) for image_path in dict_X1]
    X_2 = [get_image(image_path, grayscale=True) for image_path in dict_X2]
    return np.array(X_1).reshape([batch_size,-1]), np.array(X_2).reshape([batch_size,-1]), np.array(y_labels).reshape([batch_size,-1])

# a = get_next_batch(16, train=True)
# print(a[0].shape)
# print(a[1].shape)
# print(a[2].shape)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def G(X, reuse=False):
    with tf.variable_scope("G") as scope:
        if reuse:
            scope.reuse_variables()
     
        fc0 = tf.nn.relu(tf.layers.dense(inputs=X,   units=500,activation=None,name='hidden_layer1'))
        fc1 = tf.nn.relu(tf.layers.dense(inputs=fc0, units=10, activation=None,name='hidden_layer2'))
        return fc1



# mnist = input_data.read_data_sets('./data/mnist',one_hot=True)

# 定义回归模型
x1 = tf.placeholder(tf.float32, [None, 784])
x2 = tf.placeholder(tf.float32, [None, 784])

G1 = G(x1)
G2 = G(x2,True)


#定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 1])

E_w = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(G1-G2),1)),[1,batch_size])
# E_w = tf.sqrt(tf.reduce_sum(tf.square(G1-G2),1))

loss = tf.add(tf.multiply(tf.divide(MagicNum,Q), tf.matmul(tf.square(E_w),(One-y_))) , tf.multiply(tf.multiply(MagicNum,Q),tf.matmul(tf.exp(- tf.multiply(tf.divide(Magic2,Q),E_w)), y_)))
# loss = 0.4 *tf.matmul(tf.square(E_w),(1.0-y_)) + 10.0 * tf.matmul(tf.exp(-(2.77/Q)*E_w), y_)
# loss = (2.0/Q)*tf.square(E_w)*(1.0-y_) +(2.0*Q)*tf.exp(-(2.77/Q)*E_w) * y_

# 采用Adam作为优化器
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

show_all_variables()

for ii in range(iterations):
    batch_x1, batch_x2, batch_ys = get_next_batch(batch_size, True)
    # print(batch_x1.shape, batch_x2.shape, batch_ys.shape)
    _, loss_v = sess.run([train_step, loss], feed_dict={x1:batch_x1, x2:batch_x2,y_:batch_ys})

    # batch_x1, batch_x2, batch_ys = get_next_batch(batch_size, False)
    # loss_value = sess.run(loss, feed_dict={x1:batch_x1, x2:batch_x2,y_:batch_ys})
    print(ii,loss_v)

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print(sess.run(accuracy, feed_dict={x1:mnist.test.images[:,0:392],x2:mnist.test.images[:,392:784],y_:mnist.test.labels}))
