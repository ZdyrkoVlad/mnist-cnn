import tensorflow as tf


def train_simple(features):
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
    print("Simple NN training is running")
    for i in range(100):
        batch_xs, batch_ys = features.train.next_batch(1000)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess)
        print('step %d, training accuracy %g' % (i, train_accuracy))

    validation_accuracy = accuracy.eval(feed_dict={x: features.test.images, y_: features.test.labels},
                                        session=sess)
    print('------------------------------------------')
    print('Resulting accuracy %g' % validation_accuracy)
    print('------------------------------------------')


def train_cnn(features, conf, print_accuracy=False):
    """
    Train using convolutional neural network
    Structure of network is following:
    Input layer --> Conv layer 1 --> Max Pooling layer 1 --> Conv layer 2 -->
        Max Pooling layer 2 --> Fully connected layer --> Output layer

    """
    # --- specify input data
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10])

    # --- specify layers of network
    # TODO try another strides for conv layer
    # TODO try custom filter initialization
    # TODO try to get rid of pooling layer
    conv1 = tf.layers.conv2d(inputs=inputs, filters=conf[0], kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=conf[0], kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * conf[0]])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=10)

    # --- specify cost function and how training is performed
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    train_step = tf.train.AdamOptimizer(0.015).minimize(cross_entropy)

    # --- specify function to calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # --- actual training
    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        batch_size = 1000
        # TODO find out how to save state of network
        for i in range(conf[1]):
            batch_xs, batch_ys = features.train.next_batch(batch_size)  # random batch of data
            sess.run(train_step, feed_dict={inputs: batch_xs, labels: batch_ys})
            if print_accuracy:
                # check accuracy on training data
                train_accuracy = sess.run(accuracy, feed_dict={inputs: batch_xs, labels: batch_ys})
                print('step %d, training accuracy %g' % (i, train_accuracy))
    
        # --- check accuracy on validation data
        validation_accuracy = 0.0
        batch_number = len(features.test.images) // batch_size
        for i in range(batch_number):
            validation_accuracy += sess.run(accuracy, feed_dict={inputs: features.test.images[i*1000: (i+1)*1000],
                                                                 labels: features.test.labels[i*1000: (i+1)*1000]})
        return validation_accuracy / batch_number
