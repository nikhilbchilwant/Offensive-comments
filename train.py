import argparse
import collections
import torch
import numpy as np
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from sklearn.model_selection import KFold
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(project_config, num_samples=10, max_num_epochs=1, gpus_per_trial=0):

    tune_config = {
        "lr": tune.loguniform(1e-6, 1e-2),
        "momentum": tune.uniform(0.0, 1.0)
    }
    reporter = CLIReporter(
        # parameter_columns=["lr"],
        metric_columns=["loss", "accuracy", "roc_auc", "training_iteration"])

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    result = tune.run(
        partial(train, project_config=project_config),
        resources_per_trial={"cpu": 4, "gpu": project_config['n_gpu']},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

def train(tune_config, project_config=None):
    logger = project_config.get_logger('train')
    # setup data_loader instances
    data_loader_factory = project_config.init_obj('data_loader', module_data)
    # build model architecture, then print to console
    model = project_config.init_obj('arch', module_arch)
    # logger.info(model)
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(project_config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # get function handles of loss and metrics
    criterion = getattr(module_loss, project_config['loss'])
    metrics = [getattr(module_metric, met) for met in project_config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = project_config.init_obj('optimizer', torch.optim, trainable_params)
    optimizer = optim.SGD(trainable_params, lr=tune_config["lr"], momentum=tune_config["momentum"])
    lr_scheduler = project_config.init_obj('lr_scheduler', torch.optim.lr_scheduler,
                                   optimizer)
    fold_count = project_config['k-folds']
    kf = KFold(n_splits=fold_count)
    total_kfold_performance = {}
    tune_metrics = ['val_loss', 'val_accuracy', 'roc_auc']
    for key in tune_metrics:
        total_kfold_performance[key] = 0.0
    k = 1

    for train_indices, val_indices in kf.split(data_loader_factory.get_data()):
        train_data_loader, val_data_loader, test_data_loader = data_loader_factory.get_dataloaders(
            train_indices, val_indices)
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=project_config,
                          device=device,
                          data_loader=train_data_loader,
                          valid_data_loader=val_data_loader,
                          test_data_loader=test_data_loader,
                          lr_scheduler=lr_scheduler)
        logger.info(f'Training for k-fold (k={k})')
        best_epoch_log = trainer.train()

        for key, value in total_kfold_performance.items():
            if key in tune_metrics:
                total_kfold_performance[key] = best_epoch_log[key] + value

        # trainer.test()
        k = k + 1

    tune.report(loss=total_kfold_performance['val_loss']/fold_count,
                accuracy=total_kfold_performance['val_accuracy']/fold_count,
                roc_auc=total_kfold_performance['roc_auc']/fold_count)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    project_config = ConfigParser.from_args(args, options)
    main(project_config)
    # train(None, project_config)