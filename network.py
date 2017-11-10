import tensorflow as tf
import time


def select_activation(activation):
    result = tf.nn.relu
    if activation == 'relu':
        result = tf.nn.relu
    elif activation == 'sigmoid':
        result = tf.nn.sigmoid
    elif activation == 'tanh':
        result = tf.nn.tanh
    return result


class SimpleNet:
    """
    Train using simple neural network
    Structure of network is following:
    Input layer --> Fully connected layer --> Output layer
    """
    def __init__(self, max_epochs):
        self.inputs = self.labels = self.train_step = self.accuracy = None
        self.max_epochs = max_epochs

    def build(self, configuration):
        tf.reset_default_graph()

        # --- specify input data
        self.inputs = tf.placeholder(tf.float32, [None, 784])
        self.labels = tf.placeholder(tf.float32, [None, 10])

        # --- specify layers of network
        dense = tf.layers.dense(inputs=self.inputs, units=configuration[1], activation=select_activation(configuration[0]))
        logits = tf.layers.dense(inputs=dense, units=10)

        # --- specify cost function and how training is performed
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
        self.train_step = tf.train.AdamOptimizer(0.015).minimize(cross_entropy)

        # --- specify function to calculate accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, data, checkpoints):
        stats = []
        batch_size = 1000
        time_spent = 0.0
        start = time.time()
        with tf.Session() as sess:
            tf.global_variables_initializer().run(session=sess)
            for i in range(self.max_epochs):
                batch_xs, batch_ys = data.train.next_batch(batch_size)
                sess.run(self.train_step, feed_dict={self.inputs: batch_xs, self.labels: batch_ys})
                # write stats at checkpoints
                if (i + 1) in checkpoints:
                    validation_accuracy = sess.run(self.accuracy, feed_dict={self.inputs: data.test.images,
                                                                             self.labels: data.test.labels})
                    if validation_accuracy < 0.5:
                        # wasn't able to train
                        return None
                    time_spent += time.time() - start
                    start = time.time()
                    stats.append([i+1, validation_accuracy, time_spent])
        return stats


class ConvNet:
    """
    Train using convolutional neural network
    Structure of network is following:
    Input layer --> Conv layer 1 --> Max Pooling layer 1 --> Conv layer 2 -->
        Max Pooling layer 2 --> Fully connected layer --> Output layer
    """
    def __init__(self, max_epochs):
        self.inputs = self.labels = self.train_step = self.accuracy = None
        self.max_epochs = max_epochs
        self.summary = None

    def build(self,  configuration):
        tf.reset_default_graph()

        # --- specify input data
        self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        self.labels = tf.placeholder(tf.float32, [None, 10], name='labels')
        # tf.summary.image('input', inputs, 3)
        # TODO add name scopes and summaries

        # --- specify layers of network
        # TODO try another strides for conv layer
        # TODO try to get rid of pooling layer
        conv1 = tf.layers.conv2d(inputs=self.inputs, filters=configuration[0], kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
        conv2 = tf.layers.conv2d(inputs=pool1, filters=configuration[1], kernel_size=[5, 5], padding="same",
                                 activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
        flattened = tf.reshape(pool2, [-1, 7 * 7 * configuration[1]])
        dense = tf.layers.dense(inputs=flattened, units=1024, activation=tf.nn.relu, name='fc')
        logits = tf.layers.dense(inputs=dense, units=10, name='output')

        # --- specify cost function and how training is performed
        with tf.name_scope("train"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            self.train_step = tf.train.AdamOptimizer(0.015).minimize(cross_entropy)

        # --- specify function to calculate accuracy
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

        self.summary = tf.summary.merge_all()

    def train(self, data, checkpoints):
        stats = []
        time_spent = 0.0
        batch_size = 1000
        start = time.time()

        sess = tf.Session()
        writer = tf.summary.FileWriter('summaries')
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        # TODO implement early stopping
        # TODO find out how to save state of network and when to do it
        for i in range(self.max_epochs):
            batch_xs, batch_ys = data.train.next_batch(batch_size)  # random batch of data
            sess.run(self.train_step, feed_dict={self.inputs: batch_xs, self.labels: batch_ys})
            if i % 5 == 0:
                summary = sess.run(self.summary, feed_dict={self.inputs: batch_xs, self.labels: batch_ys})
                writer.add_summary(summary, i)

            # write stats at checkpoints
            if (i + 1) in checkpoints:
                validation_accuracy = 0.0
                batch_num = data.test.images.shape[0] // batch_size
                for p in range(batch_num):
                    validation_accuracy += sess.run(self.accuracy,
                                                   feed_dict={self.inputs: data.test.images[p*batch_size: (p+1) * batch_size],
                                                              self.labels: data.test.labels[p*batch_size: (p+1) * batch_size]})
                if validation_accuracy / batch_num < 0.5:
                    # wasn't able to train
                    return None
                time_spent += time.time() - start
                start = time.time()
                stats.append([i+1, validation_accuracy / batch_num, time_spent])
        return stats
