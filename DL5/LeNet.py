import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util


from tensorflow.examples.tutorials.mnist import input_data
    
# 封装的MNIST预测器
def LeNet(images):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
    activation_fn=tf.nn.relu, 
    weights_initializer=tf.truncated_normal_initializer(0.0,0.1), 
    weights_regularizer=slim.l2_regularizer(0.0005)
    ):
        # layer 1
        net = slim.conv2d(images, 6, [5,5], stride=1, padding='VALID', scope='conv1')
        net = slim.max_pool2d(net, [2,2], stride=2, padding='VALID', scope='pool1')
        # layer 2
        net = slim.conv2d(net, 16, [5,5], stride=1, padding='VALID', scope='conv2')
        net = slim.max_pool2d(net, [2,2], stride=2, padding='VALID', scope='pool2')
        net = slim.flatten(net, scope='flatten')
        print(net.shape)
        # layer 3, 4
        net = slim.stack(net, slim.fully_connected, [120, 84], scope='fc')
        # layer 5
        net = slim.fully_connected(net, 10 , activation_fn=None, scope='fc_3')
    return net