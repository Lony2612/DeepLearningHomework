import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from matplotlib import image

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
        self.ys1 = [0, 0]
        self.ys2 = [0, 0]
        self.ys3 = [0, 0]
        

    def draw(self, update_x, update_y1, update_y2, update_y3):
        self.xs[0] = self.xs[1]
        self.ys1[0] = self.ys1[1]
        self.ys2[0] = self.ys2[1]
        self.ys3[0] = self.ys3[1]
        
        self.xs[1] = update_x
        self.ys1[1] = update_y1
        self.ys2[1] = update_y2
        self.ys3[1] = update_y3
        
        plt.title("Training Accuracy and Loss")
        plt.xlabel("iteration")
        plt.ylabel("accuracy")

        plt.plot(self.xs, self.ys1, "b--", linewidth=2.0, label="accuracy")
        plt.plot(self.xs, self.ys2, "r--",  linewidth=2.0, label="loss")
        plt.plot(self.xs, self.ys3, "g--",  linewidth=2.0, label="loss")
        
        plt.pause(0.1)
    
    def save(self, path): 
        plt.ioff()
        # plt.legend(loc="upper right", shadow=True)
        # plt.legend(loc="upper right", shadow=True)
        plt.savefig(path)
        
        pass


flags = tf.app.flags
flags.DEFINE_integer('iterations', 30000, 'This is the iterations in training')
flags.DEFINE_integer('batch_size', 64, 'This is the batch size of the model')
flags.DEFINE_float('lr', 0.005, 'This is the rate in training')

flags.DEFINE_float('mu', 0.0, 'This is the mean of the normalization distribution')
flags.DEFINE_float('sigma', 0.1, 'This is the sigma of the normalization distribution')

FLAGS = flags.FLAGS

def model(x):
    w1=tf.Variable(dtype=tf.float32, initial_value=np.random.rand(784,1500),name='w1')
    w2=tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1500,1000),name='w2')
    w3=tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1000,500),name='w3')
    w4=tf.Variable(dtype=tf.float32, initial_value=np.random.rand(500,10),name='w4')
    b1=tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1500),name='b1')
    b2=tf.Variable(dtype=tf.float32, initial_value=np.random.rand(1000),name='b2')
    b3=tf.Variable(dtype=tf.float32, initial_value=np.random.rand(500),name='b3')
    b4=tf.Variable(dtype=tf.float32, initial_value=np.random.rand(10),name='b4')
 
    fc1=tf.nn.relu(tf.matmul(x,w1)+b1)
    fc2=tf.nn.relu(tf.matmul(fc1,w2)+b2)
    fc3=tf.nn.relu(tf.matmul(fc2,w3)+b3)
    fc4=tf.matmul(fc3,w4)+b4
    return fc4


def main(_):
    iterations = FLAGS.iterations
    lr = FLAGS.lr
    batch_size = FLAGS.batch_size

    mu = FLAGS.mu
    sigma = FLAGS.sigma

    
    gd = gif_drawer()

    input_image = tf.placeholder(tf.float32, [None, 784])
    input_label = tf.placeholder(tf.float32, [None, 10])

    logits = model(input_image)

    # logits = tf.nn.softmax(logits)

    # 注意，使用softmax_cross_entropy_with_logits_v2时，logits对应fc直接输出，不要再加softmax
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_label, logits=logits)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_label, logits=logits)

    loss = tf.reduce_mean(loss)
    tv = tf.trainable_variables()
    lambda_l = 0.0005 
    Regularization_term = lambda_l * tf.reduce_sum( [ tf.nn.l2_loss(v) for v in tv ])
    loss = Regularization_term + loss

    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # 准确率
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits),1), tf.argmax(input_label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




    # 显示所有变量
    show_all_variables()

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        mnist = input_data.read_data_sets('./data/mnist',one_hot=True)
        sess.run(tf.global_variables_initializer())
        validation_images = mnist.validation.images
        validation_labels = mnist.validation.labels
        test_images = mnist.test.images
        test_labels = mnist.test.labels
        for ii in range(iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={input_image:batch_xs, input_label: batch_ys})

            if  ii > 300 and ii % 100 == 99:
                train_loss = sess.run(loss, feed_dict={input_image:batch_xs, input_label: batch_ys})
                validation_loss = sess.run(loss, feed_dict={input_image:validation_images, input_label: validation_labels})
                validation_accu = sess.run(accuracy, feed_dict={input_image:validation_images, input_label: validation_labels})
                
                print("Iter [%5d/%5d]: valid acc is: %4f train loss is: %4f valid loss is: %4f"\
                    %(ii,iterations,validation_accu,train_loss,validation_loss))
                gd.draw(ii, 0,0,validation_accu)
                # gd.draw(ii, train_loss,validation_loss,validation_accu)
        gd.save('./train-%s.png'%lr)
        for ii in range(1):
            acc = sess.run(accuracy, \
                    feed_dict={input_image:test_images,\
                    input_label:test_labels})
            print("Test: accuracy is %4f"%(acc))
        
        w1=sess.run('w1:0')
        w1_min = np.min(w1)
        w1_max = np.max(w1)
        w1_0_to_1 = (w1 - w1_min) / (w1_max - w1_min)
        image.imsave('./w1_0_to_1.png', w1_0_to_1)

        w2=sess.run('w2:0')
        w2_min = np.min(w2)
        w2_max = np.max(w2)
        w2_0_to_1 = (w2 - w2_min) / (w2_max - w2_min)
        image.imsave('./w2_0_to_1.png', w2_0_to_1)

        w3=sess.run('w3:0')
        w3_min = np.min(w3)
        w3_max = np.max(w3)
        w3_0_to_1 = (w3 - w3_min) / (w3_max - w3_min)
        image.imsave('./w3_0_to_1.png', w3_0_to_1)
        
        w4=sess.run('w4:0')
        w4_min = np.min(w4)
        w4_max = np.max(w4)
        w4_0_to_1 = (w4 - w4_min) / (w4_max - w4_min)
        image.imsave('./w4_0_to_1.png', w4_0_to_1)

        # plt.imshow(w3_0_to_1)
        # plt.show()
        

if __name__ == '__main__':
    tf.app.run()


    