import numpy as np
import struct
import array
import torch

from src.utils.constants import Folders


"""
functions for loading data and labels of MNIST dataset
"""
def read_mnist_image_file(file_name: str) -> np.ndarray:
    """

    slightly modified from https://www.kaggle.com/code/fold10/mnist-image-classification-with-pytorch-lightning
    """

    with open(Folders.DATA + 'mnist/' + file_name, 'rb') as f:
        # IDX file format
        magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
        image_data = array.array('B', f.read())

    images = []
    for i in range(size):
        image = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
        images.append(image)

    return np.array(images)


def read_mnist_labels(file_name: str) -> np.ndarray:
    """
    slightly modified from https://www.kaggle.com/code/fold10/mnist-image-classification-with-pytorch-lightning
    """

    with open(Folders.DATA + 'mnist/' + file_name, 'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

        labels = np.array(array.array('B', f.read()))

    return labels