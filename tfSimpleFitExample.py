import tensorflow as tf
# Tensorflow python package is designed to run on CPU build prior to 2011 (without AdvancedVectorExtension instruction set)
# Therefore, disable this warning if you use CPU 7 years old or newer (or alternatively create your personal tensorflow build).
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
#for performance benchmanrk measure start and later end time stamp
import time
start = time.clock()

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]


#get and then show elapsed time
end = time.clock()
print(end-start)