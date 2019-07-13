import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import graph_util

from tensorflow.examples.tutorials.mnist import input_data
from LeNet_gd import LeNet


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
        

    def draw(self, update_x, update_y1, update_y2):
        self.xs[0] = self.xs[1]
        self.ys1[0] = self.ys1[1]
        self.ys2[0] = self.ys2[1]
        
        self.xs[1] = update_x
        self.ys1[1] = update_y1
        self.ys2[1] = update_y2
        
        plt.title("Training Accuracy and Loss")
        plt.xlabel("iteration")
        plt.ylabel("accuracy")

        plt.plot(self.xs, self.ys1, "b--", linewidth=2.0, label="accuracy")
        plt.plot(self.xs, self.ys2, "r--",  linewidth=2.0, label="loss")
        
        plt.pause(0.1)
    
    def save(self, path): 
        plt.ioff()
        # plt.legend(loc="upper right", shadow=True)
        # plt.legend(loc="upper right", shadow=True)
        plt.savefig(path)
        
        pass


flags = tf.app.flags
flags.DEFINE_integer('iterations', 1000, 'This is the iterations in training')
flags.DEFINE_integer('batch_size', 8, 'This is the batch size of the model')
flags.DEFINE_float('lr', 0.001, 'This is the rate in training')

flags.DEFINE_float('mu', 0.0, 'This is the mean of the normalization distribution')
flags.DEFINE_float('sigma', 0.1, 'This is the sigma of the normalization distribution')

FLAGS = flags.FLAGS

def main(_):
    iterations = FLAGS.iterations
    lr = FLAGS.lr
    batch_size = FLAGS.batch_size

    mu = FLAGS.mu
    sigma = FLAGS.sigma

    lenet_part = LeNet(iterations=iterations, lr=lr, batch_size=batch_size, mu=mu, sigma=sigma)

    gd = gif_drawer()

    show_all_variables()

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        mnist = input_data.read_data_sets('./data/mnist',one_hot=True)
        sess.run(tf.global_variables_initializer())
        test_images = mnist.test.images
        test_images = np.reshape(test_images,[-1,28,28,1])
        test_images = np.pad(test_images,((0,0),(2,2),(2,2),(0,0)), 'constant')
        for ii in range(lenet_part.iterations):
            batch_xs, batch_ys = mnist.train.next_batch(lenet_part.batch_size)
            batch_xs = np.reshape(batch_xs,[-1,28,28,1])
            batch_xs = np.pad(batch_xs, ((0,0),(2,2),(2,2),(0,0)), 'constant')
            sess.run(lenet_part.train_op, feed_dict={lenet_part.raw_input_image:batch_xs, lenet_part.raw_input_label: batch_ys})

            if ii % 10 == 0:
                validation_images, validation_labels = mnist.validation.next_batch(100)
                validation_images = np.reshape(validation_images,[-1,28,28,1])
                validation_images = np.pad(validation_images,((0,0),(2,2),(2,2),(0,0)), 'constant')

                acc, los = sess.run([lenet_part.accuracy, lenet_part.cross_entropy], \
                    feed_dict={lenet_part.raw_input_image:validation_images,\
                    lenet_part.raw_input_label:validation_labels})
                print("Iteration [%5d/%5d]: accuracy is: %4f loss is: %4f"%(ii,lenet_part.iterations,acc,los))
                gd.draw(ii, acc, los)
        gd.save('./train_gd.png')
        for ii in range(10):
            acc = sess.run(lenet_part.accuracy, \
                    feed_dict={lenet_part.raw_input_image:test_images,\
                    lenet_part.raw_input_label:mnist.test.labels})
            print("Test: accuracy is %4f"%(acc))

if __name__ == '__main__':
    tf.app.run()


    