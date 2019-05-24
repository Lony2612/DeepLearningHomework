import tensorflow as tf
import tensorflow.contrib.slim as slim
from mobilenet_v2 import mobilenetv2,mobilenet_arg_scope
from cnn_utils import create_readable_names_for_imagenet_labels
import cv2
import numpy as np


inputs = tf.placeholder(tf.uint8, [None, None, 3])
images = tf.expand_dims(inputs, 0)
images = tf.cast(images, tf.float32) / 128. - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

with slim.arg_scope(mobilenet_arg_scope(is_training=False)):
    logits ,endpoints= mobilenetv2(images)

# Restore using exponential moving average since it produces (1.5-2%) higher
# accuracy
ema = tf.train.ExponentialMovingAverage(0.999)
vars = ema.variables_to_restore()

saver = tf.train.Saver(vars)

print(len(tf.global_variables()))
for var in tf.global_variables():
    print(var)
checkpoint_path = "mobilenet_v2_1.ckpt"
image_file = "panda.jpg"
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = endpoints['Predictions'].eval(feed_dict={inputs: img})
    label_map = create_readable_names_for_imagenet_labels()  
    print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())