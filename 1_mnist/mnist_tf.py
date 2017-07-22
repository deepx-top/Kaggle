# -*- coding:utf-8 -*-

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from load_mnist import load_mnist
import numpy as np

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train = load_mnist('./train.csv')

x_ = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# model
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x_, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# session
sess = tf.InteractiveSession()

# train
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

batch_size = 50
for i in range(20000):
    batch = train.next_batch(batch_size, shuffle=True)
    if i % 100 == 0:
        train_accuracy = sess.run(
            accuracy, feed_dict={x_: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={
             x_: batch[0], y_: batch[1], keep_prob: 0.5})

# test predict
predict = tf.argmax(y_conv, 1)
test = load_mnist('./test.csv')
predicted_lables = np.zeros(test.num_examples)
for i in range(0, int(test.num_examples / batch_size)):
    batch = test.next_batch(batch_size)
    predicted_lables[i * batch_size:(i + 1) * batch_size] = sess.run(
        predict, feed_dict={x_: batch[0], keep_prob: 1.0})

# save results
np.savetxt('submission_softmax.csv',
           np.c_[range(1, test.num_examples + 1), predicted_lables],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')

# for i in range(100):
#     batch = mnist.test.next_batch(50)
#     test_accuracy = sess.run(accuracy, feed_dict={
#         x_: batch[0], y_: batch[1], keep_prob: 1.0})
#     test_acc_sum = tf.add(test_acc_sum, test_accuracy)
#     sess.run(test_acc_sum)
# avg_test_acc = test_acc_sum / 100.0
# print("test accuracy %g" % sess.run(avg_test_acc))
