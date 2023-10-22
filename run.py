import os
import yaml
import argparse

from src.experiment import Experiment
from src.file_handler import FileHandler
from src.utils.constants import Folders

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='fcnn_mnist_label_noise_0_10',
                    help='name of config which should be used for experiment')
args = parser.parse_args()


if __name__ == '__main__':

    # load and validate config
    config_file = os.path.join(Folders.CONFIGS, args.config + '.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    experiment = Experiment(
        config=config,
        file_handler=FileHandler(config_name=args.config, read_mode=False),
    )
    experiment.run()
