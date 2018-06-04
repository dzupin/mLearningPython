# Tensorflow python package is designed to run on CPU build prior to 2011 (without AdvancedVectorExtension instruction set)
# Therefore, disable this warning if you use CPU 7 years old or newer (or alternatively create your personal tensorflow build).
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#The downloaded MNIST database of handwritten digits, from http://yann.lecun.com/exdb/mnist/, is for our convenience split into three parts:
# 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation).
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   #one_hot is vector with 1 and all 0's like:  [0, 0, 0 ,0 ,1 ,0 ]
#Sample code to process sample mnist data
import tensorflow as tf
#for performance benchmanrk measure start and later end time stamp
import time
start = time.clock()

#Define your x(input) W(weight) and b(bias)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implement our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#TRAINING SECTION
#define true results
y_ = tf.placeholder(tf.float32, [None, 10])
#Evaluate how good/bad is our model by calculating cost (variability between known results y_ and our model result y)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#Backpropagation algorithm that tries to minimize cost by optimizing variables in our model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Run our training cycle 1000 times
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluate our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#get and then show elapsed time
end = time.clock()
print(end-start)