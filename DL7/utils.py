import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist',one_hot=True)
test_images = mnist.test.images
test_labels = mnist.test.labels

def getLabel(one_hot):
    for ii in range(len(one_hot)):
        if one_hot[ii] == 1.0:
            return ii

# 获取均衡测试集
def getTestImage(one_hot=False):
    lab = []
    test1 = []
    test2 = []

    lab1 = []
    lab2 = []

    lab_one1 = []
    lab_one2 = []
    # 获取相同标签数据集 
    for label in range(0,10):
        temp_image = test_images[0]
        count = 0
        for ii in range(10000):
            if getLabel(test_labels[ii]) == label:
                if count >= 900:
                    break
                if count % 2 == 0:
                    test1.append(test_images[ii])
                    lab1.append(label)
                    lab_one1.append(test_labels[ii])
                else:
                    test2.append(test_images[ii])
                    lab2.append(label)
                    lab_one2.append(test_labels[ii])
                    temp_image = test_images[ii]
                    lab.append(0.0)
                count += 1
        # 数字5只有446组数据,直接用最后一组在最后再重复4次补足
        if label == 5:
            for _ in range(4):
                test1.append(temp_image)
                test2.append(temp_image)
                lab.append([0.0])
                lab1.append(label)
                lab2.append(label)
                lab_one1.append(test_labels[ii])
                lab_one2.append(test_labels[ii])
            pass
        print("Positive (%d,%d): %d"%(label,label,count/2))
    # 获取不同标签数据集
    for label1 in range(0,10):
        for label2 in range(label1+1,10):
            # 获取第一个标签的数据
            count  = 0
            for ii in range(10000):
                if getLabel(test_labels[ii]) == label1:
                    if count >= 100:
                        break
                    test1.append(test_images[ii])
                    lab1.append(label1)
                    lab_one1.append(test_labels[ii])
                    count += 1
            # 获取第二个标签的数据
            count  = 0
            for ii in range(10000):
                if getLabel(test_labels[ii]) == label2:
                    if count >= 100:
                        break
                    test2.append(test_images[ii])
                    lab2.append(label2)
                    lab_one2.append(test_labels[ii])
                    count += 1
                    lab.append([1.0])
            print("Negtive (%d,%d): %d"%(label1,label2,count))
    if one_hot:
        return test1, test2, lab, lab_one1, lab_one2
    else:
        return test1, test2, lab, lab1, lab2




def minist_draw(im):
    im = im.reshape(28, 28)
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.axis('off')
    plt.imshow(im, cmap='gray')
    plt.show()
    # plt.savefig("test.png")  
    plt.close()
 
def balanced_batch(batch_x, batch_y, num_cls):
    batch_size=len(batch_y)
    pos_per_cls_e=round(batch_size/2/num_cls/2)
    pos_per_cls_e*=2
 
    index=batch_y.argsort()
    ys_1=batch_y[index]
    
    num_class=[]
    pos_samples=[]
    neg_samples=set()
    cur_ind=0
    for item in set(ys_1):
        num_class.append((ys_1==item).sum())
        num_pos=pos_per_cls_e
        while(num_pos>num_class[-1]):
            num_pos-=2
        pos_samples.extend(np.random.choice(index[cur_ind:cur_ind+num_class[-1]],num_pos,replace=False).tolist())
        neg_samples=neg_samples|(set(index[cur_ind:cur_ind+num_class[-1]])-set(list(pos_samples)))
        cur_ind+=num_class[-1]
    
    neg_samples=list(neg_samples)
    
    x1_index=pos_samples[::2]
    x2_index=pos_samples[1:len(pos_samples)+1:2]
 
    x1_index.extend(neg_samples[::2])
    x2_index.extend(neg_samples[1:len(neg_samples)+1:2])
    
    p_index=np.random.permutation(len(x1_index))
    x1_index=np.array(x1_index)[p_index]
    x2_index=np.array(x2_index)[p_index]
 
    r_x1_batch=batch_x[x1_index]
    r_x2_batch=batch_x[x2_index]
    r_y_batch=np.array(batch_y[x1_index]!=batch_y[x2_index],dtype=np.float32)
    return r_x1_batch,r_x2_batch,r_y_batch.reshape([-1,1]),batch_y[x1_index].reshape([-1,1]),batch_y[x2_index].reshape([-1,1])