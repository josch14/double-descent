import torch


def OneZeroLoss(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    zero-one classification loss

    :param y_true: ground truth labels of shape (n_samples,)
    :param y_pred: predicted labels of shape (n_samples,)
    :return: fraction of misclassifications
    """
    assert y_true.dim() == 1 and y_true.size() == y_pred.size()

    n = y_true.size(dim=0)
    loss = (n - (y_pred == y_true).sum().item()) / n

    return loss
