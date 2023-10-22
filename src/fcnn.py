from __future__ import annotations
import torch.nn as nn
from math import sqrt


class FCNN(nn.Module):

    def __init__(self,
                 in_nodes: int,
                 hidden_nodes: int,
                 out_nodes: int,
                 final_activation: str,
                 weight_reuse: str,
                 weight_initialization: str,
                 interpolation_threshold: int,
                 dropout: float,
                 smaller_model: FCNN):

        super().__init__()

        # hidden layer
        self.hidden_layer = nn.Linear(
            in_features=in_nodes,
            out_features=hidden_nodes
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # output layer
        self.out_layer = nn.Linear(
            in_features=hidden_nodes,
            out_features=out_nodes
        )

        # final activation following output layer
        match final_activation:

            case 'none':
                self.final_activation = None
            case 'softmax':
                self.final_activation = nn.Softmax(dim=1)
            case _:
                exit('error: usage of activation function \'{}\' not implement'.format(final_activation))

        self.n_parameters = sum(p.numel() for p in self.parameters())
        self.__weight_init(weight_reuse=weight_reuse,
                           weight_initialization=weight_initialization,
                           interpolation_threshold=interpolation_threshold,
                           smaller_model=smaller_model)


    def __weight_init(self, weight_reuse: str, weight_initialization: str, interpolation_threshold: int, smaller_model: FCNN):
        """
        initialize weights (xavier uniform in original paper)
        1) over-parametrized regime: use xavier uniform
        2) under-parametrized regime
          2.1) no weight reuse: use xavier uniform
          2.2) weight reuse: obtain parameters from trained smaller model, initialize remaining weights with
                             normally distributed random numbers (mean 0 and variance 0.01)
        """

        if smaller_model is None \
                or weight_reuse == 'none' \
                or (weight_reuse == 'under-parametrized' and interpolation_threshold <= self.n_parameters):

            # in case of weight_reuse, the smallest model and all models above the interpolation threshold
            # are initialized randomly

            # final activation following output layer
            match weight_initialization:

                case 'xavier_uniform':
                    init_function_ = nn.init.xavier_uniform_
                case _:
                    exit('error: usage of weight initialization function \'{}\' not implement'.format(weight_initialization))


            init_function_(self.hidden_layer.weight, gain=1.0)
            self.hidden_layer.bias.data.fill_(0.01)
            init_function_(self.out_layer.weight, gain=1.0)
            self.out_layer.bias.data.fill_(0.01)

        else:
            assert weight_reuse == "continuous" \
                   or (weight_reuse == "under-parametrized" and self.n_parameters < interpolation_threshold)

            # initialize (all) weights with normally distributed random numbers
            nn.init.normal(self.hidden_layer.weight, mean=0.0, std=sqrt(0.01))
            nn.init.normal(self.hidden_layer.bias, mean=0.0, std=sqrt(0.01))
            nn.init.normal(self.out_layer.weight, mean=0.0, std=sqrt(0.01))
            nn.init.normal(self.out_layer.bias, mean=0.0, std=sqrt(0.01))

            hidden_nodes_sm = smaller_model.hidden_layer.out_features
            # hidden layers have same input dimensions
            self.hidden_layer.weight.data[:hidden_nodes_sm, :] = smaller_model.hidden_layer.weight.data
            self.hidden_layer.bias.data[:hidden_nodes_sm] = smaller_model.hidden_layer.bias.data
            # out layers have same output dimension
            self.out_layer.weight.data[:, :hidden_nodes_sm] = smaller_model.out_layer.weight.data
            self.out_layer.bias.data = smaller_model.out_layer.bias.data


    def forward(self, x):

        # hidden layer
        x = self.hidden_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        # output layer
        x = self.out_layer(x)
        if self.final_activation:
            x = self.final_activation(x)

        return x
