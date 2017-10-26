import tensorflow as tf

import numpy as np


def train_simple(train_data, train_labels, validation_data, validation_labels):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    for i in range(100):
        sess.run(train_step, feed_dict={x: train_data, y_: train_labels})
        train_accuracy = accuracy.eval(feed_dict={x: train_data, y_: train_labels}, session=sess)
        print('step %d, training accuracy %g' % (i, train_accuracy))

    validation_accuracy = accuracy.eval(feed_dict={x: validation_data, y_: validation_labels},
                                        session=sess)
    print('------------------------------------------')
    print('Resulting accuracy %g' % validation_accuracy)
    print('------------------------------------------')


def train_cnn(train_data, train_labels, validation_data, validation_labels):
    """
    Train using convolutional neural network
    Structure of network is following:
    Input layer --> Conv layer 1 --> Max Pooling layer 1 --> Conv layer 2 -->
        Max Pooling layer 2 --> Fully connected layer --> Output layer

    """
    # reshape so that we have 28x28 images
    train_data = np.reshape(train_data, [-1, 28, 28, 1])
    validation_data = np.reshape(validation_data, [-1, 28, 28, 1])

    ###################################################
    # specify input data
    ###################################################

    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    ###################################################
    # specify layers of network
    ###################################################

    # CONVOLUTIONAL LAYER 1
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # MAX POOLING LAYER 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # CONVOLUTIONAL LAYER 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # MAX POOLING LAYER 1
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # flatten last layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # FULLY CONNECTED LAYER
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # OUTPUT LAYER
    logits = tf.layers.dense(inputs=dense, units=10)

    ###################################################
    # specify cost function and how training is performed
    ###################################################

    #TODO try some other cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(tf.nn.softmax(logits)), axis=1))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    ###################################################
    # specify function to calculate accuracy
    ###################################################

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ###################################################
    # actual training
    ###################################################
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    for i in range(1000):
        p = np.random.randint(1000, train_data.shape[0])
        sess.run(train_step, feed_dict={inputs: train_data[p-1000:p],
                                        labels: train_labels[p-1000:p]})
        train_accuracy = accuracy.eval(feed_dict={inputs: train_data[p-1000:p],
                                                  labels: train_labels[p-1000:p]},
                                       session=sess)
        print('step %d, training accuracy %g' % (i, train_accuracy))

    ###################################################
    # check accuracy on validation data
    ###################################################
    validation_accuracy = accuracy.eval(feed_dict={inputs: validation_data[:1000],
                                                   labels: validation_labels[:1000]},
                                        session=sess)
    print('------------------------------------------')
    print('Resulting accuracy %g' % validation_accuracy)
    print('------------------------------------------')
