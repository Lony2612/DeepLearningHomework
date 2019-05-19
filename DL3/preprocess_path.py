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

iterations = 100000
lr = 0.01
batch_size = 64
Q = tf.constant([5.0])

def create_labels(batch_size):
    y_labels = []
    for _ in range(batch_size):
        if random.random() < 0.5:
            temp = 0.0
        else:
            temp = 1.0
        y_labels.append(temp)
    return y_labels

def get_image_path(label, train=True):
    if label == 1.0:
        dict_name_1 = str(random.randint(0,9))
        dict_name_2 = str(random.randint(0,9))
        if train == True:
            dict_1 = glob(os.path.join('./mnist_data/train',dict_name_1,'*.png'))
            dict_2 = glob(os.path.join('./mnist_data/train',dict_name_2,'*.png'))
        else:
            dict_1 = glob(os.path.join('./mnist_data/test',dict_name_1,'*.png'))
            dict_2 = glob(os.path.join('./mnist_data/test',dict_name_2,'*.png'))
        image_name_1 = random.sample(dict_1, 1)
        image_name_2 = random.sample(dict_2, 1)
        return image_name_1 + image_name_2
    else:
        dict_name = str(random.randint(0,9))
        if train == True:
            dicretory = glob(os.path.join('./mnist_data/train',dict_name,'*.png'))
        else:
            dicretory = glob(os.path.join('./mnist_data/test',dict_name,'*.png'))
        image_name = random.sample(dicretory, 2)
        return image_name

def get_image_paths(labels, train=True):
    train_X1 = []
    train_X2 = []

    for ii in range(len(labels)):
        images_path2 = get_image_path(labels[ii], train=True)
        train_X1.append(images_path2[0])
        train_X2.append(images_path2[1])
    return train_X1, train_X2




def get_paths(batch_size, train=True):
    if train == True:
        train_same_f = open("train_same_f.txt",'a')
        train_dif_f  = open("train_dif_f.txt",'a')
        for _ in range(25):
            y_labels = create_labels(batch_size)
            for ii in range(len(y_labels)):
                # print("xxx%d"%ii)
                images_path2 = get_image_path(y_labels[ii], train=True)
                if y_labels[ii] == 1.0:
                    train_dif_f.write("%s %s"%(images_path2[0], images_path2[1]))
                    train_dif_f.write('\n')
                else:
                    train_same_f.write("%s %s"%(images_path2[0], images_path2[1]))
                    train_same_f.write('\n')
    else:
        test_same_f  = open("test_same_f.txt",'a')
        test_dif_f   = open("test_dif_f.txt",'a')
        for _ in range(25):
            y_labels = create_labels(batch_size)
            for ii in range(len(y_labels)):
                images_path2 = get_image_path(y_labels[ii], train=False)
                if y_labels[ii] == 1.0:
                    test_dif_f.write("%s %s"%(images_path2[0], images_path2[1]))
                    test_dif_f.write('\n')
                else:
                    test_same_f.write("%s %s"%(images_path2[0], images_path2[1]))
                    test_same_f.write('\n')
    if train == True:
        train_same_f.close()
        train_dif_f.close()
    else:
        test_same_f.close()
        test_dif_f.close()

start = time.time()
get_paths(64)
mid = time.time()
get_paths(64,train=False)
end = time.time()
print(mid-start, end-mid)
