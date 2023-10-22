import os
import numpy as np
from matplotlib import pyplot as plt
from src.utils.constants import Metrics


class Plotter:

    @staticmethod
    def plot_statistics(statistics: dict,
                        config: dict,
                        metric: tuple,
                        plot_folder: os.path,
                        run_idx: int = None):

        # get information from statistics and config
        statistics_train = statistics['train'][metric[0]]
        statistics_test = statistics['test'][metric[0]]
        hidden_nodes = config['fcnn_parameters']['hidden_nodes']
        input_dim = config['dataset_parameters']['input_dim']
        n_classes = config['dataset_parameters']['n_classes']
        n_train = config['dataset_parameters']['n_train']
        interpolation_threshold = n_train * n_classes

        # apply averaging or only use single run
        mean_train = statistics_train[run_idx, :] if run_idx else np.mean(statistics_train, axis=0)
        mean_test = statistics_test[run_idx, :] if run_idx else np.mean(statistics_test, axis=0)

        # determine number of parameters per FCNN size
        n_parameters = [(input_dim + 1)*h + (h + 1)*n_classes for h in hidden_nodes]


        # create plot
        fig, ax = plt.subplots(figsize=(10, 3.5))

        # plot vertical line at interpolation threshold
        ax.axvline(x=interpolation_threshold, label='Interpolation Threshold', linewidth=2.5, color='black', linestyle='--')

        # plot horizontal line indicating optimal performance
        if metric in [Metrics.ZERO_ONE_LOSS, Metrics.SQUARED_LOSS]:
            ax.axhline(y=0, linewidth=1, color='gray', linestyle='--')
        elif metric in [Metrics.PRECISION, Metrics.RECALL, Metrics.F1_SCORE]:
            ax.axhline(y=1, linewidth=1, color='gray', linestyle='--')

        # plot std deviation
        if run_idx is None:
            std_train = statistics_train[run_idx, :] if run_idx else np.std(statistics_train, axis=0)
            std_test = statistics_test[run_idx, :] if run_idx else np.std(statistics_test, axis=0)

            ax.fill_between(n_parameters, mean_train - std_train, mean_train + std_train, color='C1', alpha=0.3)
            ax.fill_between(n_parameters, mean_test - std_test, mean_test + std_test, color='C0', alpha=0.3)

        # plot mean
        ax.plot(n_parameters, mean_train, color='C1', marker='D', markersize=3, linewidth=1, linestyle='-', label='Train')
        ax.plot(n_parameters, mean_test, color='C0', marker='D', markersize=3, linewidth=1, linestyle='-', label='Test')


        # label x-axis
        def big_number_as_string(n):
            n = '{}'.format(round((n / 1000), 1))
            n = n.rstrip('0').rstrip('.')
            n += 'k'
            return n

        ax.set_xlabel(
            '# Parameters / # Hidden Nodes',
            fontsize=11,
            weight="bold"
        )
        ax.set_xlim((0, 105000))
        hidden_nodes_ticks = [2, 10, 25, 40, 60, 75, 100, 126]
        n_parameters_ticks = [(input_dim + 1)*h + (h + 1)*n_classes for h in hidden_nodes_ticks]
        ax.set_xticks(
            n_parameters_ticks + [interpolation_threshold],  # positions
            ['{}\n{}'.format(big_number_as_string(p), h) for h, p in zip(hidden_nodes_ticks, n_parameters_ticks)] +
            ['{}'.format(big_number_as_string(interpolation_threshold))]  # labels
        )


        # label y-axis
        ax.set_ylabel(
            f'{metric[1]}' if metric != Metrics.ZERO_ONE_LOSS else f'{Metrics.ZERO_ONE_LOSS[1]} (%)',
            fontsize=11,
            weight="bold"
        )

        y_limits, y_ticks, y_labels = None, None, None
        if metric == Metrics.ZERO_ONE_LOSS:
            y_limits = (-0.01, 0.65)
            y_ticks = [0.0, 0.2, 0.4, 0.6]
            y_labels = [('   {0: d}'.format(int(y*100))) for y in y_ticks]

        elif metric in [Metrics.RECALL, Metrics.PRECISION, Metrics.F1_SCORE]:
            y_limits = (0, 1.02)
            y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            y_labels = [('   {}'.format(y)) for y in y_ticks]

        else:
            y_limits = ax.get_ylim()
            y_ticks = ax.get_yticks()
            if metric == Metrics.ENTROPY_LOSS:
                y_labels = [('   {}'.format(round(y, 5))) for y in y_ticks]

            if metric == Metrics.SQUARED_LOSS:
                y_labels = [(' {:1.2f}'.format(round(y, 5))) for y in y_ticks]

        ax.set_yticks(y_ticks, y_labels)
        ax.set_ylim(y_limits)

        # legend
        location = 'lower right' if metric in [Metrics.RECALL, Metrics.PRECISION, Metrics.F1_SCORE] else 'upper right'
        ax.legend(loc=location)

        # save or show plot
        plt.tight_layout(pad=0.0)
        if plot_folder:
            plot_folder = os.path.join(plot_folder, f'{metric[0]}.svg')
            fig.savefig(plot_folder, format='svg')
        else:
            plt.show()
