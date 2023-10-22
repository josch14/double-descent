from random import randrange
import numpy as np
import torch

from .utils.constants import Datasets
from .utils.functions import read_mnist_image_file, read_mnist_labels


class DataLoader:

    def __init__(self, dataset_parameters: dict):

        self.dataset = dataset_parameters['dataset']
        self.n_train = dataset_parameters['n_train']
        self.n_classes = dataset_parameters['n_classes']
        self.label_noise = dataset_parameters['label_noise']

        # determine interpolation threshold
        self.interpolation_threshold = self.n_classes * self.n_train


    def load(self):

        match self.dataset:

            case Datasets.MNIST:
                # we do not use the same train test split as given
                X_train = read_mnist_image_file('train-images.idx3-ubyte')
                X_test = read_mnist_image_file('t10k-images.idx3-ubyte')
                Y_train = read_mnist_labels('train-labels.idx1-ubyte')
                Y_test = read_mnist_labels('t10k-labels.idx1-ubyte')

                # scale the maximum range of each feature to the interval [0, 1]
                X_train = X_train / 255.
                X_test = X_test / 255.

            case _:
                exit(f'error: loading data of dataset {self.dataset} not implement')

        # shuffle X and Y together
        X_train, Y_train = DataLoader.shuffle_X_Y(X_train, Y_train)
        X_test, Y_test = DataLoader.shuffle_X_Y(X_test, Y_test)

        # limit train data if desired
        X_train = X_train[:self.n_train]
        Y_train = Y_train[:self.n_train]

        # apply label noise
        DataLoader.apply_label_noise(Y_train, noise=self.label_noise, n_classes=self.n_classes)

        return X_train, X_test, Y_train, Y_test


    @staticmethod
    def apply_label_noise(Y_train: np.ndarray, noise: float, n_classes: int):

        # apply label noise to a subset of train samples
        if 0.0 < noise:

            def produce_false_label(label: int):

                false_label = label
                while label == false_label:
                    false_label = randrange(n_classes)

                return false_label

            n_false_labels = int(noise * len(Y_train))

            for i in range(n_false_labels):
                Y_train[i] = produce_false_label(Y_train[i])

        return Y_train


    @staticmethod
    def get_train_batches(X_train: np.ndarray, Y_train: np.ndarray, batch_size: int):

        X_train, Y_train = DataLoader.shuffle_X_Y(X_train, Y_train)

        batches_train = []
        n_batches = (len(X_train) // batch_size) + (len(X_train) % batch_size > 0)
        for i in range(n_batches):
            batches_train.append(
                (
                    torch.tensor(X_train[i*batch_size:(i+1)*batch_size, :], dtype=torch.float32),
                    torch.tensor(Y_train[i*batch_size:(i+1)*batch_size], dtype=torch.float32)
                )
            )

        return batches_train


    @staticmethod
    def shuffle_X_Y(X: np.ndarray, Y: np.ndarray):

        assert len(X) == len(Y)

        p = np.random.permutation(len(X))

        return X[p], Y[p]
