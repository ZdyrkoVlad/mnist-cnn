from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import itertools
import logging
from datetime import datetime
from timeit import default_timer as timer

import network

# Configuration consists of 2 lists:
#   1. Filter size of Conv layers;
#   2. Number of epochs in training
configurations = [[5, 10, 20, 32], [100, 400, 700]]


def main(args):
    # --- setup logging to file and stdout
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(message)s")
    file_name = "mnist_cnn_log_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_handler = logging.FileHandler("logs/{0}.log".format(file_name))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    if args and args[0] == 'cnn':
        start = timer()
        mnist = read_data_sets('.\\data', one_hot=True, reshape=False)
        root_logger.info("Loaded data! Elapsed time: {} sec.".format(timer() - start))
        all_configurations = list(itertools.product(*configurations))  # create all possible combinations

        root_logger.info('Training starts:\n')
        total_elapsed_time = 0.0
        best_accuracy = 0.0
        best_conf = None
        for conf in all_configurations:
            start = timer()
            accuracy = network.train_cnn(mnist, conf)
            elapsed_time = timer() - start
            total_elapsed_time += elapsed_time
            root_logger.info('CNN training with configuration: {} finished. Accuracy: {}. Elapsed time: {} sec'
                             .format(conf, accuracy, elapsed_time))
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_conf = conf

        root_logger.info('----------------------------------------------------------------------')
        root_logger.info('Best accuracy of our CNN: {}, configuration: {}'.format(best_accuracy, best_conf))
        root_logger.info('Total elapsed time: {} minutes, {} seconds'
                         .format(total_elapsed_time // 60, total_elapsed_time % 60))
        root_logger.info('----------------------------------------------------------------------')

    else:
        mnist = read_data_sets('.\\data', one_hot=True, reshape=True)
        network.train_simple(mnist)

    logging.shutdown()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
