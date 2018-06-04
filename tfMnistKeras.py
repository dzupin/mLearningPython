import tensorflow as tf

# Tensorflow python package is designed to run on CPU build prior to 2011 (without AdvancedVectorExtension instruction set)
# Therefore, disable this warning if you use CPU 7 years old or newer (or alternatively create your personal tensorflow build).
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable eager execution. Eager execution is a mode for running TensorFlow that works just like regular Python.
tf.enable_eager_execution()

import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

# Reshape from (N, 28, 28) to (N, 784)
train_images = np.reshape(train_images, (TRAINING_SIZE, 784))
test_images = np.reshape(test_images, (TEST_SIZE, 784))

# Convert the array to float32 as opposed to uint8
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

# Convert the pixel values from integers between 0 and 255 to floats between 0 and 1
train_images /= 255
test_images /=  255

NUM_DIGITS = 10

print("Before", train_labels[0]) # The format of the labels before conversion

train_labels  = tf.keras.utils.to_categorical(train_labels, NUM_DIGITS)

print("After", train_labels[0]) # The format of the labels after conversion

test_labels = tf.keras.utils.to_categorical(test_labels, NUM_DIGITS)

# Cast the labels to floats, needed later
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# Create a TensorFlow optimizer, rather than using the Keras version
# This is currently necessary when working in eager mode
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()


# Create a tf.data Dataset. We use the tf.data.Dataset API to convert the Numpy arrays into a TensorFlow dataset.
BATCH_SIZE=128

# Because tf.data may work with potentially **large** collections of data
# we do not shuffle the entire dataset by default
# Instead, we maintain a buffer of SHUFFLE_SIZE elements
# and sample from there.
SHUFFLE_SIZE = 10000

# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)

# Iterate over the dataset and train model using model.train_on_batch.
EPOCHS = 5
for epoch in range(EPOCHS):
    for images, labels in dataset:
        train_loss, train_accuracy = model.train_on_batch(images, labels)

    # Here you can gather any metrics or adjust your training parameters
    print('Epoch #%d\t Loss: %.6f\tAccuracy: %.6f' % (epoch + 1, train_loss, train_accuracy))

loss, accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy: %.2f' % (accuracy))