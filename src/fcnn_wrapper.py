from typing import Tuple
import numpy as np
import torch
from torch import nn

from src.dataloader import DataLoader
from src.fcnn import FCNN


class WrapperFCNN:

    def __init__(self,
                 in_nodes: int,
                 hidden_nodes: int,
                 out_nodes: int,
                 fcnn_parameters: dict,
                 interpolation_threshold: int,
                 smaller_model: FCNN):

        self.fcnn_parameters = fcnn_parameters

        # interpolation threshold
        self.n_classes = out_nodes
        self.interpolation_threshold = interpolation_threshold

        # initialize model
        self.model = FCNN(
            in_nodes=in_nodes,
            hidden_nodes=hidden_nodes,
            out_nodes=out_nodes,
            final_activation=self.fcnn_parameters['final_activation'],
            weight_reuse=self.fcnn_parameters['weight_reuse'],
            weight_initialization=self.fcnn_parameters['weight_initialization'],
            interpolation_threshold=interpolation_threshold,
            smaller_model=smaller_model,
            dropout=self.fcnn_parameters['dropout']
        )

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, training_parameters: dict):

        # use parameter info
        n_epochs = training_parameters['n_epochs']
        batch_size = training_parameters['batch_size']
        step_size_reduce_epochs = training_parameters['step_size_reduce_epochs']
        step_size_reduce_percent = training_parameters['step_size_reduce_percent']
        stop_at_zero_error = training_parameters['stop_at_zero_error']

        # initialize loss
        match training_parameters['loss']:

            case 'squared_loss':
                Loss = nn.MSELoss()
            case 'cross_entropy':
                Loss = nn.CrossEntropyLoss()
            case _:
                exit('error: usage of loss function \'{}\' not implement'.format(training_parameters['loss']))


        # initialize optimizer
        match training_parameters['optimizer']:

            case 'sgd':
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=training_parameters['learning_rate'],
                    momentum=0.95,
                    weight_decay=training_parameters['weight_decay'],
                )
            case _:
                exit('error: usage of optimizer \'{}\' not implement'.format(training_parameters['optimizer']))


        # apply preprocessing
        X_train = self.__preprocess(X_train)

        # train loop
        self.model.train()

        for epoch in range(1, n_epochs + 1):

            # lr schedule below interpolation threshold
            if epoch % step_size_reduce_epochs == 0 and self.model.n_parameters < self.interpolation_threshold:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * (1 - step_size_reduce_percent)

            classification_error = 0

            batches_train = DataLoader.get_train_batches(X_train, Y_train, batch_size=batch_size)

            for x, labels in batches_train:
                optimizer.zero_grad()

                outputs = self.model.forward(x)

                # classification error
                _, predicted = outputs.max(1)
                classification_error += torch.count_nonzero(labels != predicted)

                # compute loss and gradients
                labels = nn.functional.one_hot(labels.long(), num_classes=self.n_classes).float()
                loss = Loss(outputs, labels)
                loss.backward()

                # adjust learning weights
                optimizer.step()

            # networks smaller than the interpolation threshold: training is stopped after classification error reached
            # zero or 6000 epochs, whichever happens earlier
            if stop_at_zero_error and self.model.n_parameters < self.interpolation_threshold and classification_error == 0:
                break


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input array X.

        :param X: input samples of shape (n_samples, n_features)
        :return: predicted classes of shape (n_samples,)
        """

        X = self.__preprocess(X)

        # convert to torch tensor
        X = torch.tensor(X, dtype=torch.float32)

        # perform prediction
        self.model.eval()
        p = self.model.forward(X)

        # convert to numpy array
        p = p.detach().numpy()

        return p


    def predict_class_and_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class memberships and probabilities for input array X.

        :param X: input samples of shape (n_samples, n_features)
        :return: predicted classes of shape (n_samples,)
        """

        p = self.predict_proba(X)
        c = np.argmax(p, axis=1)

        return c, p


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class memberships for input array X.

        :param X: input samples of shape (n_samples, n_features)
        :return: predicted classes of shape (n_samples,)
        """

        p = self.predict_proba(X)
        c = np.argmax(p, axis=1)

        return c


    def __preprocess(self, X: np.ndarray) -> np.ndarray:

        # either vector or image data
        assert X.ndim <= 3

        # image data has to be flattened
        if X.ndim == 3:
            X = np.reshape(X, (len(X), -1))

        return X
