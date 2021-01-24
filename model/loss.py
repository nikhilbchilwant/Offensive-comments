from torch import nn


def cross_entropy_loss(output, target):
    cross_entropy_loss = nn.CrossEntropyLoss()
    return cross_entropy_loss(output, target)
