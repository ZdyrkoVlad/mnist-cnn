from mnist import MNIST
import network
import numpy as np


def create_labels(labels):
    labels_new = np.zeros((len(labels), 10), dtype=np.float32)
    for i, l in enumerate(labels):
        labels_new[i][l] = 1.0
    return labels_new


def normalize_image(image):
    image_new = np.array([pix / 255.0 for row in image for pix in row], dtype=np.float32)
    return np.reshape(image_new, [-1, 784])


def main():
    mndata = MNIST('.\\data')
    images, labels = mndata.load_training()
    eval_images, eval_labels = mndata.load_testing()
    print("Loaded!!!!!!")

    # TODO improve performace of loading and copying
    images = normalize_image(images)
    eval_images = normalize_image(eval_images)
    labels = create_labels(labels)
    eval_labels = create_labels(eval_labels)
    print("Copied!!!!!!")

    #network.train_simple(images, labels, eval_images, eval_labels)
    network.train_cnn(images, labels, eval_images, eval_labels)

if __name__ == '__main__':
    main()