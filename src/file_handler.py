import os
import numpy as np
import yaml

from src.utils.constants import Folders


class FileHandler:

    def __init__(self, config_name: str, read_mode: bool):

        self.config_name = config_name
        self.log_folder = os.path.join(Folders.LOGS, config_name)

        if not read_mode:
            # FileHandler used for saving logs
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)
            else:
                if len(os.listdir(self.log_folder)) == 0:
                    # no problem when the folder already exists, but does not contain any files
                    pass
                else:
                    exit(f'error: logs for {self.config_name} already exist at {self.log_folder}: delete or move folder and run again')

        else:
            # FileHandler used for reading logs
            if not os.path.exists(self.log_folder):
                exit(f'error: results can not be shown: no logs exist for {self.config_name} at {self.log_folder}')
            else:
                # fine
                pass


    """
    Config File 
    """
    def save_config(self, config: dict):
        file = os.path.join(self.log_folder, self.config_name + '.yaml')

        with open(file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


    def load_config(self):
        file = os.path.join(self.log_folder, self.config_name + '.yaml')

        with open(file, 'r') as f:
            config = yaml.safe_load(f)
            # no validation required

        return config


    """
    Numpy Array
    """
    def save_numpy(self, metric_name: str, arr: np.ndarray):
        file = os.path.join(self.log_folder, metric_name + '.npy')

        with open(file, 'wb') as f:
            np.save(f, arr)

    def load_numpy(self, metric_name):
        file = os.path.join(self.log_folder, metric_name + '.npy')

        with open(file, 'rb') as f:
            arr = np.load(f)

        return arr


    """
    Logs
    """
    def save_logs(self,
                  config: dict,
                  zero_one_loss_train: np.ndarray,
                  squared_loss_train: np.ndarray,
                  entropy_loss_train: np.ndarray,
                  precision_train: np.ndarray,
                  recall_train: np.ndarray,
                  f1_train: np.ndarray,
                  zero_one_loss_test: np.ndarray,
                  squared_loss_test: np.ndarray,
                  entropy_loss_test: np.ndarray,
                  precision_test: np.ndarray,
                  recall_test: np.ndarray,
                  f1_test: np.ndarray):

        self.save_config(config=config)

        # train results
        self.save_numpy('zero_one_loss_train', zero_one_loss_train)
        self.save_numpy('squared_loss_train', squared_loss_train)
        self.save_numpy('entropy_loss_train', entropy_loss_train)
        self.save_numpy('precision_train', precision_train)
        self.save_numpy('recall_train', recall_train)
        self.save_numpy('f1_train', f1_train)

        # test results
        self.save_numpy('zero_one_loss_test', zero_one_loss_test)
        self.save_numpy('squared_loss_test', squared_loss_test)
        self.save_numpy('entropy_loss_test', entropy_loss_test)
        self.save_numpy('precision_test', precision_test)
        self.save_numpy('recall_test', recall_test)
        self.save_numpy('f1_test', f1_test)


    def load_logs(self):

        # config
        config = self.load_config()

        # statistics
        statistics = {
            'train': {
                'zero_one_loss': self.load_numpy('zero_one_loss_train'),
                'squared_loss': self.load_numpy('squared_loss_train'),
                'entropy_loss': self.load_numpy('entropy_loss_train'),
                'precision': self.load_numpy('precision_train'),
                'recall': self.load_numpy('recall_train'),
                'f1': self.load_numpy('f1_train'),
            },
            'test': {
                'zero_one_loss': self.load_numpy('zero_one_loss_test'),
                'squared_loss': self.load_numpy('squared_loss_test'),
                'entropy_loss': self.load_numpy('entropy_loss_test'),
                'precision': self.load_numpy('precision_test'),
                'recall': self.load_numpy('recall_test'),
                'f1': self.load_numpy('f1_test'),
            },
        }

        return config, statistics

