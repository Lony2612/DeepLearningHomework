import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util


from tensorflow.examples.tutorials.mnist import input_data

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
def LeNet(image):
    mu = 0
    sigma = 0.1

    # Padding: 28*28->32*32
    image = np.pad(image, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    # Solution: Layer 1: Convolutional. Input=32*32*1. Output=28*28*6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev=sigma), name="conv1_W")
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(image, conv1_W, strides=[1,1,1,1], padding="VALID") + conv1_b
    # Solution: Activation
    conv1 = tf.nn.relu(conv1)
    # Solution: Pooling. Input=28*28*6. Output=14*14*6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
    # Solution: Layer 2: Convolutional. Output=10*10*16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev=sigma), name="conv1_W")
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1,1,1,1], padding="VALID") + conv2_b
    # Solution: Activation
    conv2 = tf.nn.relu(conv2)
    # Solution: Pooling. Input=10*10*16. Output=5*5*16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
    # Solution: Flatten. Input = 5*5*16. Output = 400.
    fc0 = tf.reshape(conv2,[-1,400])
    # Solution：Layer3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    # Solution: Activation.
    fc1 = tf.nn.relu(fc1)
    # Solution: Layer 4: Fully Connected. Input=120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    # Solution: Activation.
    fc2 = tf.nn.relu(fc2)
    # Solution: Layer 5: Fully Connected. Input=84. Output=10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84,10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    # return tf.nn.softmax(logits)
    return logits
    
# 封装的MNIST预测器
class FC_MNIST():
    def __init__(self, sess, iterations=20000, lr=0.02, batch_size=64):
        self.sess       = sess 
        # 定义超参数
        self.iterations = iterations
        self.lr         = lr
        self.batch_size = batch_size
        
        # 初始化绘制器
        self.gd = gif_drawer()
        # 建立模型
        self._build_model()

    def _build_model(self):
        # 定义占位符
        self.x1 = tf.placeholder(tf.float32, [None, 392])
        self.x2 = tf.placeholder(tf.float32, [None, 392])
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        # 调用网络
        y = self.FCC(self.x1, self.x2)
        # 定义损失函数
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=self.y_))
        # 定义优化器
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        # 定义正确率
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义网络模型
    def LeNet(self, image):
        mu = 0
        sigma = 0.1

        # Padding: 28*28->32*32
        image = np.pad(image, ((0,0),(2,2),(2,2),(0,0)), 'constant')

        # Solution: Layer 1: Convolutional. Input=32*32*1. Output=28*28*6
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev=sigma), name="conv1_W")
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(image, conv1_W, strides=[1,1,1,1], padding="VALID") + conv1_b
        # Solution: Activation
        conv1 = tf.nn.relu(conv1)
        # Solution: Pooling. Input=28*28*6. Output=14*14*6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
        # Solution: Layer 2: Convolutional. Output=10*10*16
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev=sigma), name="conv1_W")
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1,1,1,1], padding="VALID") + conv2_b
        # Solution: Activation
        conv2 = tf.nn.relu(conv2)
        # Solution: Pooling. Input=10*10*16. Output=5*5*16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
        # Solution: Flatten. Input = 5*5*16. Output = 400.
        fc0 = tf.reshape(conv2,[-1,400])
        # Solution：Layer3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b
        # Solution: Activation.
        fc1 = tf.nn.relu(fc1)
        # Solution: Layer 4: Fully Connected. Input=120. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b
        # Solution: Activation.
        fc2 = tf.nn.relu(fc2)
        # Solution: Layer 5: Fully Connected. Input=84. Output=10.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(84,10), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        # return tf.nn.softmax(logits)
        return logits
    
    # 定义训练函数
    def train(self):
        tf.global_variables_initializer().run()
        for ii in range(self.iterations):
            batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
            self.sess.run(self.train_step, feed_dict={self.x1: batch_xs[:,0:392],\
                self.x2: batch_xs[:,392:784],self.y_:batch_ys})
            if ii % 500 == 1:
                acc, los = self.sess.run([self.accuracy, self.cross_entropy], \
                    feed_dict={self.x1:mnist.test.images[:,0:392],\
                    self.x2:mnist.test.images[:,392:784],\
                    self.y_:mnist.test.labels})
                print("Iteration [%5d/%5d]: accuracy is: %4f loss is: %4f"%(ii,self.iterations,acc,los))
                self.gd.draw(ii, acc)
    
    # 定义测试函数
    def test(self):
        acc = self.sess.run(self.accuracy, \
            feed_dict={self.x1:mnist.test.images[:,0:392],\
            self.x2:mnist.test.images[:,392:784],\
            self.y_:mnist.test.labels})
        print("Test: accuracy is %4f"%(acc))

if __name__ == '__main__':
    mnist = input_data.read_data_sets('./data/mnist',one_hot=True)
    # 新建会话
    with tf.Session() as sess:
        # 实例化对象
        fcc = FC_MNIST(sess)
        # 启动训练
        fcc.train()
        # 测试
        fcc.test()