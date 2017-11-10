from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import itertools
import logging
from datetime import datetime
from tabulate import tabulate

import network

# Configuration for CNN consists of 2 lists:
#   1. Filter size of Conv layer #1;
#   2. Filter size of Conv layer #2;
#configurations_cnn = [[5, 6, 7, 8, 10], [5, 10, 15, 20, 32]]
configurations_cnn = [[5], [5]]

# Configuration for simple NN consists of 2 lists:
#   1. Type of activation function;
#   2. Number of neurons in hidden layer;
configurations_simple_nn = [['relu', 'sigmoid', 'tanh'], [60, 90, 120, 150, 180]]

# Number of epochs at which to check accuracy of our model
checkpoints = [100, 300, 500, 700, 1000]


def main(args):
    if args and args[0] == 'cnn':
        network_type = 'convolutional'
        reshape_data = False
        neural_network = network.ConvNet(max_epochs=1000)
        all_configurations = list(itertools.product(*configurations_cnn))  # create all possible combinations
        headers = ['1 layer', '2 layer', 'Epochs', 'Accuracy', 'Training time']
    else:
        network_type = 'simple'
        reshape_data = True
        neural_network = network.SimpleNet(max_epochs=1000)
        all_configurations = list(itertools.product(*configurations_simple_nn))  # create all possible combinations
        headers = ['Activation', 'Hidden neurons', 'Epochs', 'Accuracy', 'Training time']

    # --- setup logging to file and stdout
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(message)s")
    file_name = "mnist_" + network_type + "_log_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_handler = logging.FileHandler("logs/{0}.log".format(file_name))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    # --- load MNIST data
    mnist = read_data_sets('.\\data', one_hot=True, reshape=reshape_data)
    root_logger.info('Loaded MNIST data!')
    root_logger.info(network_type.title() + ' neural network training starts:\n')

    # --- start training
    total_elapsed_time = 0.0; best_accuracy = 0.0; best_conf = None; all_results = []
    for conf in all_configurations:
        neural_network.build(configuration=conf)
        stats = None
        while stats is None:
            stats = neural_network.train(data=mnist, checkpoints=checkpoints)
        elapsed_time = stats[len(stats) - 1][2]
        total_elapsed_time += elapsed_time

        for item in stats:
            all_results.append([conf[0], conf[1], item[0], item[1], item[2]])
        all_results.append([])
        accuracy = max([row[1] for row in stats])  # take max value from all checkpoints
        print('Training finished. Configuration: {}. Accuracy: {:.4f}, Time: {:.1f} sec'
              .format(conf, accuracy, elapsed_time))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_conf = conf

    # --- log results to file
    root_logger.info(tabulate(all_results, headers=headers, floatfmt=".4f", numalign='center', stralign='center'))
    root_logger.info('----------------------------------------------------------------------')
    root_logger.info('Best accuracy: {:.4f}, configuration: {}'.format(best_accuracy, best_conf))
    root_logger.info('Total elapsed time: {:.0f} minutes, {:.1f} seconds'
                     .format(total_elapsed_time // 60, total_elapsed_time % 60))
    root_logger.info('----------------------------------------------------------------------')

    logging.shutdown()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
