import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

# load the data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

# check the image size and data size
assert (len(X_train) == len(y_train))
assert (len(X_validation) == len(y_validation))
assert (len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

import numpy as np

# Pad images with 0s
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
print(X_test)
print(X_test.shape)

print('updated size of train images ={}'.format(X_train[0].shape))
print('updated size of validation images ={}'.format(X_validation[0].shape))
print('updated size of test images ={}'.format(X_test[0].shape))

# View a sample from the dataset.
import random
import matplotlib.pyplot as plt
import pylab

# % matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

print(image)
print(image.shape)

# plot the image
plt.figure(figsize=(1, 1))
plt.imshow(image, cmap="gray")
plt.savefig('./pic2.png')  # 要先保存再show
print('pic saved')
pylab.show()
print(y_train[index])

# def pre_pic(picName):
#     img = Image.open('./pic2.png')
#     #使用消除锯齿的方法resize图片
#     reIm = img.resize((32, 32), Image.ANTIALIAS)
#     # 变成灰度图，转换成矩阵
#     im_arr = np.array(reIm.convert("L"))
#     threshold = 50  # 对图像进行二值化处理，设置合理的阈值，可以过滤掉噪声，让他只有纯白色的点和纯黑色点
#     for i in range(32):
#         for j in range(32):
#             im_arr[i][j] = 255 - im_arr[i][j]
#             if (im_arr[i][j] < threshold):
#                 im_arr[i][j] = 0
#             else:
#                 im_arr[i][j] = 255
#     # 将图像矩阵拉成1行784列，并将值变成浮点型（像素要求的仕0-1的浮点型输入）
#     print(im_arr)
#     nm_arr = im_arr.reshape([1, 1024])
#     nm_arr = nm_arr.astype(np.float32)
#     print(nm_arr)
#     img_ready = np.multiply(nm_arr, 1.0 / 255.0).reshape(1, 32, 32, 1)
#     np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
#     print(img_ready)
#     return img_ready


# X_test = pre_pic('pic2.png')
# print(X_test)
# print(X_test.shape)

from sklearn.utils import shuffle

# Shuffle the training data.
X_train, y_train = shuffle(X_train, y_train)  # 数据打乱

# The EPOCH and BATCH_SIZE values affect the training speed and model accuracy.
EPOCHS = 10  # 训练十次
BATCH_SIZE = 128  # 每次训练128个样本

from tensorflow.contrib.layers import flatten


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
    # Activation function relu
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
    fc1 = flatten(pool_2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    # Activation function relu
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # Activation function relu
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    # Multiply matrix
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits


# Train LeNet to classify MNIST data.
# x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y) # softmax和cross——entropy
loss_operation = tf.reduce_mean(cross_entropy) # 计算每次误差
optimizer = tf.train.AdamOptimizer(learning_rate=rate) # 优化用的是adam
training_operation = optimizer.minimize(loss_operation) # 减小loss

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1)) #argmax是取max
# cast将correct转换为float32
accuracy_opertaion = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


# calcluate the accuracy
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE] # 起点：终点
        accuracy, logits_ = sess.run([accuracy_opertaion, logits], feed_dict={x: batch_x, y: batch_y})
        total_accuracy = total_accuracy + (accuracy * len(batch_x))
        res = zip(logits_, y_data) # 把logits和y-data每一项打包
        for r in res:
            print(r)
        print(total_accuracy / num_examples)
    return total_accuracy / num_examples


save_file = './lalala/train_model'  # file path

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 初始化模型的参数
    num_examples = len(X_train)

    print('training...')
    print('')

    # training
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train) # 打乱顺序
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y}) # 让placeholder启动
        validation_accuracy = evaluate(X_validation, y_validation)
        save_path = saver.save(sess, save_file, global_step=i) # save the file for 10 times, each epoch
        print('model saved in :', save_path)
        print('EPOCH{}...'.format(i + 1))
        print('Validation Accuracy = {:.3f}'.format(validation_accuracy))
        print()

with tf.Session() as sess:
    saver.restore(sess, './lalala/train_model-9') # 保存以后restore model 这里调用的是最新的第9个
    evaluate(X_test, y_test)
    # test_accuracy = evaluate(X_test_normalized, y_test)
    # print('test accuracy={:.3f}'.format(test_accuracy))
