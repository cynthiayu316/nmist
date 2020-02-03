import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np
from tensorflow.contrib.layers import flatten
import random
import matplotlib.pyplot as plt
import pylab

# load in mnist data for the use of training
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
# set x as image, y as label
X_test, y_test = mnist.test.images, mnist.test.labels
# 第一个参数是待填充数组；第二个参数是填充的形状，（2，3）表示前面两个，后面三个；第三个参数是填充的方法
# 填充方法
# constant连续一样的值填充，有关于其填充值的参数。constant_values=（x, y）时前面用x填充，后面用y填充。缺参数是为0。
# edge边缘值填充
# linear_ramp边缘递减的填充方式
# maximum, mean, median, minimum分别用最大值、均值、中位数和最小值填充
# reflect, symmetric都是对称填充。前一个是关于边缘对称，后一个是关于边缘外的空气对称╮(╯▽╰)╭
# wrap用原数组后面的值填充前面，前面的值填充后面
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')


# 把照片传入进行处理
def pre_pic(picName):
    # 先打开传入的原始图片
    img = Image.open('./lenet train/Picture9.png').convert('L')
    # 使用消除锯齿的方法resize图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 变成灰度图，转换成矩阵
    im_arr = np.array(reIm.convert("L"))
    threshold = 100  # 对图像进行二值化处理，设置合理的阈值，可以过滤掉噪声，让他只有纯白色的点和纯黑色点
    for i in range(28):
        for j in range(28):
            # 黑白反转
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0 #black
            else:
                im_arr[i][j] = 255 #white
    # 将图像矩阵拉成1行784列，并将值变成浮点型（像素要求的仕0-1的浮点型输入）
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32) #astype不改变数组长度的改变数据类型
    img_ready = np.multiply(nm_arr, 1.0 / 255.0).reshape(1, 28, 28, 1) #reshape成四维

    # a是输入的矩阵；axis : 选择shape中的一维条目的子集。如果在shape大于1的情况下设置axis，则会引发错误。
    tmp_img = img_ready[0].squeeze()  #去掉维度为1的
    print(tmp_img.shape)
    print(tmp_img)
    plt.figure(figsize=(1, 1))
    plt.imshow(tmp_img, cmap="gray")
    plt.savefig('./pic1.png')  # 要先保存再show
    print('pic saved')
    pylab.show()
    return img_ready


X_test, y_test = pre_pic('Picture4.png'), np.array([[1]])
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')


# print(X_test[5][5])
# print(X_test)


def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1': 6,
        'layer_2': 16,
        'layer_3': 120,
        'layer_f1': 84
    }

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits


# Train LeNet to classify MNIST data.
# x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001 #learning rate

logits = LeNet(x)


def predict(X_data, y_data):
    num_examples = len(X_data) #length
    sess = tf.get_default_session()
    BATCH_SIZE = 1 # 每次只测试一个样本
    # 多加了一个argmax函数 找最大值预测
    p = tf.argmax(logits, 0)
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_test[offset:offset + BATCH_SIZE], y_test[offset:offset + BATCH_SIZE]
        res = sess.run(logits, feed_dict={x: batch_x, y: batch_y})  # batch size就是开头：结尾
        # 只要有sess.run 就得有feed_dict来run placeholder里的数据
    print(res, y_data[offset:offset + 1])


saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./lalala')) #restore model
    predict(X_test, y_test)
