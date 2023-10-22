import argparse
import os

from src.file_handler import FileHandler
from src.plotter import Plotter
from src.utils.constants import Folders, Metrics

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='fcnn_mnist_label_noise_0_10',
                    help='name of results folder existing in {}'.format(Folders.LOGS))
parser.add_argument('-s', '--save', type=bool, default=False,
                    help='whether to show plots or to save the generated plots to disk')
args = parser.parse_args()


if __name__ == '__main__':

    # create file handler
    file_handler = FileHandler(config_name=args.config, read_mode=True)

    # load config and results
    config, statistics = file_handler.load_logs()

    # save to disc
    plot_folder = os.path.join(Folders.PLOTS, args.config) if args.save else None
    if args.save and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # plot
    for metric in [Metrics.SQUARED_LOSS, Metrics.ZERO_ONE_LOSS, Metrics.ENTROPY_LOSS, Metrics.PRECISION, Metrics.RECALL, Metrics.F1_SCORE]:
        Plotter.plot_statistics(statistics=statistics,
                                config=config,
                                metric=metric,
                                plot_folder=plot_folder)
