import numpy as np
from sys import stdout
from tqdm import tqdm

from src.dataloader import DataLoader
from src.evaluator import Evaluator
from src.fcnn_wrapper import WrapperFCNN
from src.file_handler import FileHandler


class Experiment:

    def __init__(self, config: dict, file_handler: FileHandler):

        # config
        self.config = config
        self.dataset_parameters = self.config['dataset_parameters']
        self.fcnn_parameters = self.config['fcnn_parameters']
        self.training_parameters = self.config['training_parameters']

        # file handler
        self.file_handler = file_handler


    def run(self):

        repetitions = self.config['repetitions']
        hidden_nodes = self.fcnn_parameters['hidden_nodes']

        zero_one_loss_train, squared_loss_train, entropy_loss_train, precision_train, recall_train, f1_train = \
            [np.zeros(shape=(repetitions, len(hidden_nodes))) for _ in range(6)]
        zero_one_loss_test, squared_loss_test, entropy_loss_test, precision_test, recall_test, f1_test = \
            [np.zeros(shape=(repetitions, len(hidden_nodes))) for _ in range(6)]


        with tqdm(total=repetitions * len(hidden_nodes), file=stdout, smoothing=0.1) as pbar:

            for r in range(repetitions):

                smaller_model = None

                # load dataset
                data_loader = DataLoader(
                    dataset_parameters=self.dataset_parameters
                )
                X_train, X_test, Y_train, Y_test = data_loader.load()

                for i, h in enumerate(hidden_nodes):

                    # initialize model
                    model = WrapperFCNN(
                        in_nodes=self.dataset_parameters['input_dim'],
                        hidden_nodes=h,
                        out_nodes=self.dataset_parameters['n_classes'],
                        fcnn_parameters=self.fcnn_parameters,
                        interpolation_threshold=data_loader.interpolation_threshold,
                        smaller_model=smaller_model
                    )

                    # train model
                    model.train(X_train=X_train,
                                Y_train=Y_train,
                                training_parameters=self.training_parameters)
                    smaller_model = model.model

                    # predict on train and test data
                    c_train, p_train = model.predict_class_and_proba(X_train)
                    c_test, p_test = model.predict_class_and_proba(X_test)

                    zero_one_loss, squared_loss, entropy_loss, precision, recall, f1 = Evaluator.evaluate(y_pred=c_train,
                                                                                                          p_pred=p_train,
                                                                                                          y_true=Y_train,
                                                                                                          n_classes=data_loader.n_classes)
                    zero_one_loss_train[r][i] += zero_one_loss
                    squared_loss_train[r][i] += squared_loss
                    entropy_loss_train[r][i] += entropy_loss
                    precision_train[r][i] += precision
                    recall_train[r][i] += recall
                    f1_train[r][i] += f1

                    zero_one_loss, squared_loss, entropy_loss, precision, recall, f1 = Evaluator.evaluate(y_pred=c_test,
                                                                                                          p_pred=p_test,
                                                                                                          y_true=Y_test,
                                                                                                          n_classes=data_loader.n_classes)
                    zero_one_loss_test[r][i] += zero_one_loss
                    squared_loss_test[r][i] += squared_loss
                    entropy_loss_test[r][i] += entropy_loss
                    precision_test[r][i] += precision
                    recall_test[r][i] += recall
                    f1_test[r][i] += f1

                    # upgrade progress bar
                    pbar.update(1)


        self.file_handler.save_logs(config=self.config,
                                    zero_one_loss_train=zero_one_loss_train,
                                    squared_loss_train=squared_loss_train,
                                    entropy_loss_train=entropy_loss_train,
                                    precision_train=precision_train,
                                    recall_train=recall_train,
                                    f1_train=f1_train,
                                    zero_one_loss_test=zero_one_loss_test,
                                    squared_loss_test=squared_loss_test,
                                    entropy_loss_test=entropy_loss_test,
                                    precision_test=precision_test,
                                    recall_test=recall_test,
                                    f1_test=f1_test)
