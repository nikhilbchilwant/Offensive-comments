import torch
from sklearn import metrics


def accuracy(output, target):
    with torch.no_grad():
        _, pred = torch.max(output, dim=1)
        assert len(pred) == len(target)
        # correct = 0
        # correct += torch.sum(pred == target).item()        
    return metrics.accuracy_score(target.tolist(), pred.tolist())
