import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util


from tensorflow.examples.tutorials.mnist import input_data
    
# 封装的MNIST预测器
class LeNet():
    def __init__(self, iterations=20000, lr=0.01, batch_size=64, mu=0, sigma=0.1):
        # self.sess       = sess 
        # 定义超参数
        self.iterations = iterations
        self.lr         = lr
        self.batch_size = batch_size
        self.mu = mu
        self.sigma = sigma

        # 定义梯度截断的最大最小值
        self.min = -0.001
        self.max = 0.001

        # 建立模型
        self._build_graph()

    def _build_graph(self, network_name="LeNet"):
        self._setup_placeholders_graph()
        self._build_network_graph("LeNet")
        self._compute_loss_graph()
        self._compute_acc_graph()  
        self._create_train_op_graph()      
    
    def _setup_placeholders_graph(self):
        # 定义占位符
        self.raw_input_image = tf.placeholder(tf.float32, [None, 32, 32, 1])
        self.raw_input_label = tf.placeholder(tf.float32, [None, 10])
    
    def _build_network_graph(self, network_name):
        # 调用网络
        self.y = self.LeNet(self.raw_input_image, network_name)
    
    def _compute_loss_graph(self):
        # 定义损失函数
        with tf.name_scope("loss_function"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.raw_input_label))

    def _compute_acc_graph(self):
        # 定义正确率
        with tf.name_scope("loss_function"):
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.raw_input_label,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def _create_train_op_graph(self):
        # 定义优化器
        with tf.name_scope("train_op"):
            # 获取梯度变量
            var_list = tf.trainable_variables()
            # 定义优化器
            optimizer = tf.train.AdamOptimizer(self.lr)
            # 获取梯度
            self.gradients = optimizer.compute_gradients(self.cross_entropy, var_list)
            # 对梯度进行截断
            capped_gradients = [(tf.clip_by_value(grad, self.min, self.max), var) for grad, var in self.gradients if grad is not None]
            # 应用截断梯度来更新参数
            self.train_op = optimizer.apply_gradients(capped_gradients)
            # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
    
    def _conv_(self, input_image, w_shape, strides, padding, scope_name, reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            conv_W = tf.Variable(tf.truncated_normal(shape=w_shape, mean = self.mu, stddev=self.sigma))
            conv_b = tf.Variable(tf.zeros(w_shape[3]))
            conv = tf.nn.conv2d(input_image, conv_W, strides=strides, padding=padding) + conv_b
            return conv
    
    def _pooling_(self, input_image, ksize, strides, padding, scope_name, reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            pool = tf.nn.max_pool(input_image, ksize=ksize, strides=strides,padding=padding)
            return pool
    
    def _fully_connected_(self, input_image, shape, scope_name, reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            fc_W = tf.Variable(tf.truncated_normal(shape=shape, mean=self.mu, stddev=self.sigma))
            fc_b = tf.Variable(tf.zeros(shape[1]))
            fc = tf.matmul(input_image, fc_W) + fc_b
            return fc
    
    # 定义网络模型
    def LeNet(self, image, name):
        
        # Solution: Layer 1: Convolutional. Input=32*32*1. Output=28*28*6
        conv1 = self._conv_(input_image=image,w_shape=(5,5,1,6),strides=[1,1,1,1],padding="VALID",scope_name="conv1")
        # Solution: Activation
        conv1 = tf.nn.relu(conv1)
        # Solution: Pooling. Input=28*28*6. Output=14*14*6.
        conv1 = self._pooling_(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID", scope_name="pooling1")
        
        # Solution: Layer 2: Convolutional. Output=10*10*16
        conv2 = self._conv_(input_image=conv1,w_shape=(5,5,6,16),strides=[1,1,1,1],padding="VALID",scope_name="conv2")
        # Solution: Activation
        conv2 = tf.nn.relu(conv2)
        # Solution: Pooling. Input=10*10*16. Output=5*5*16.
        conv2 = self._pooling_(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID", scope_name="pooling2")
        
        # Solution: Flatten. Input = 5*5*16. Output = 400.
        fc0 = tf.reshape(conv2,[-1,400])
        # Solution：Layer3: Fully Connected. Input = 400. Output = 120.
        fc1 = self._fully_connected_(fc0, shape=(400,120), scope_name="fc1")
        # Solution: Activation.
        fc1 = tf.nn.relu(fc1)
        
        # Solution: Layer 4: Fully Connected. Input=120. Output = 84.
        fc2 = self._fully_connected_(fc1, shape=(120,84), scope_name="fc2")
        # Solution: Activation.
        fc2 = tf.nn.relu(fc2)
        
        # Solution: Layer 5: Fully Connected. Input=84. Output=10.
        logits = self._fully_connected_(fc2, shape=(84,10), scope_name="fc3")
        print(logits.shape)
        # return tf.nn.softmax(logits)
        return logits