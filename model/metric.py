import torch
from sklearn import metrics

def accuracy(output, target):
    with torch.no_grad():
        _, pred = torch.max(output, dim=1)
        assert len(pred) == len(target)
        # correct = 0
        # correct += torch.sum(pred == target).item()        
    return metrics.accuracy_score(target.tolist(), pred.tolist())

# def auc(output, target):
#     sm = torch.nn.Softmax(dim=1) 
#     probability = sm(output)
#     fpr, tpr, _ = metrics.roc_curve(target.to_list(), probability[:,1].to_list())
#     return metrics.auc(fpr, tpr)