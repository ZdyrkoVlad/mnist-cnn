import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import itertools

configurations = [[5], [5]]
#mnist = read_data_sets('.\\data', one_hot=True, reshape=True)
# TODO deal with runtime statictics
# TODO deal with hyperparameters


def train(data, configuration, max_epochs=1):
    tf.reset_default_graph()

    # --- specify input data
    inputs = tf.placeholder(tf.float32, [None, 28 * 28], name='x')
    inputs_reshaped = tf.reshape(inputs, [-1, 28, 28, 1])
    labels = tf.placeholder(tf.float32, [None, 10], name='labels')

    # --- specify layers of network
    # TODO try another strides for conv layer
    # TODO try to get rid of pooling layer
    conv1 = tf.layers.conv2d(inputs=inputs_reshaped, filters=configuration[0], kernel_size=[5, 5], padding="same",
                             activation=tf.nn.relu, name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
    conv2 = tf.layers.conv2d(inputs=pool1, filters=configuration[1], kernel_size=[5, 5], padding="same",
                             activation=tf.nn.relu, name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
    flattened = tf.reshape(pool2, [-1, 7 * 7 * configuration[1]])
    dense = tf.layers.dense(inputs=flattened, units=1024, activation=tf.nn.relu, name='fc')
    logits = tf.layers.dense(inputs=dense, units=10, name='output')

    with tf.name_scope("train"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        train_step = tf.train.AdamOptimizer(0.015).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summary = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test')

    # TODO implement early stopping
    sess.run(tf.global_variables_initializer())
    '''
    for i in range(max_epochs):
        batch_xs, batch_ys = data.train.next_batch(1000)  # random batch of data
        if i % 5 == 0:
            summ = sess.run(summary, feed_dict={inputs: data.test.images, labels: data.test.labels})
            test_writer.add_summary(summ, i)
        elif i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summ, _ = sess.run([summary, train_step], feed_dict={inputs: batch_xs, labels: batch_ys},
                               options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summ, i)
        else:     
            summ, _ = sess.run([summary, train_step], feed_dict={inputs: batch_xs, labels: batch_ys})
            train_writer.add_summary(summ, i) 
    '''

    var = [v for v in tf.trainable_variables() if v.name == "conv1/kernel:0"]
    values = sess.run(var)
    print(values)

if __name__ == '__main__':
    all_configurations = list(itertools.product(*configurations))
    train([], all_configurations[0])
