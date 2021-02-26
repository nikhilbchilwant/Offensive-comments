import argparse
import collections
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import StratifiedKFold

import data_loader as module_data
import model as module_arch
import model.domain_loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import ray
import pprint

import os

# fix random seeds for reproducibility
SEED = 10
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def search_hyperparameters(project_config, num_samples=20):
    ray.init(local_mode=(project_config["ray_local_mode"]=="True")) #enable for debugging

    tune_config = {
        # "lr": tune.randn(0.0055573, 5e-5),
        # "momentum": tune.randn(0.143631, 5e-3)
         "lr": tune.loguniform(1e-4, 1e-2),
        "momentum": tune.uniform(0.01, 0.99)
    }

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "roc_auc", "f1", "epoch", "training_iteration"])

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    result = tune.run(
        partial(kfold_train, project_config=project_config),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir='/data/users/nchilwant/training_output/ray_tune',
        name="mtl-domain-adaptation",
        # log_to_file=("stdout.log", "stderr.log"),
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    return best_trial


def kfold_train(tune_config, project_config=None):
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
    criterion = project_config.init_obj('loss', module_loss)
    criterion.to(device)
    metrics = [getattr(module_metric, met) for met in project_config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    model_trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # criterion_trainable_params = filter(lambda p: p.requires_grad, criterion.parameters())
    # trainable_params = list(model_trainable_params)
    # trainable_params.extend(list(criterion_trainable_params))
    # optimizer = project_config.init_obj('optimizer', torch.optim, trainable_params)        #TODO: add loss function parameters
    #TODO: add loss function parameters
    # optimizer = optim.SGD([{'params': model_trainable_params},
    #             {'params': criterion_trainable_params}], 
    #             lr=tune_config["lr"], momentum=tune_config["momentum"])
    optimizer = optim.SGD(model_trainable_params, 
                lr=tune_config["lr"], momentum=tune_config["momentum"])
    lr_scheduler = project_config.init_obj('lr_scheduler', torch.optim.lr_scheduler,
                                   optimizer)
    fold_count = project_config['k-folds']
    kf = StratifiedKFold(n_splits=fold_count, shuffle=True)
    total_kfold_performance = {}
    tune_metrics = ['val_loss', 'val_accuracy', 'val_roc_auc', 'val_f1', 'epoch']
    for key in tune_metrics:
        total_kfold_performance[key] = 0.0
    k = 1
    task_datasets = data_loader_factory.get_datasets()

    comment_index = 0
    label_index = 1

    dataset_split_indices = []
    for dataset in task_datasets:
        dataset = dataset.to_numpy()
        train_indices_list = []
        val_indices_list = []
        for train_indices, val_indices in kf.split(np.zeros(len(dataset[:,comment_index])),
            dataset[:,label_index].tolist()):
            train_indices_list.append(train_indices)
            val_indices_list.append(val_indices)
        dataset_split_indices.append([train_indices_list, val_indices_list])
    
    dataset_split_indices = np.asarray(dataset_split_indices)

    for split_no in range(0, kf.get_n_splits()):
        train_splits = []
        val_splits = []
        for split_index in dataset_split_indices:
            train_splits.append(split_index[0][split_no])
            val_splits.append(split_index[1][split_no])

        train_data_loader = data_loader_factory.get_train_dataloader(train_splits)
        val_data_loader = data_loader_factory.get_val_dataloader(val_splits)
        target_data_loader = data_loader_factory.get_target_dataloader()

        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=project_config,
                          device=device,
                          data_loader=train_data_loader,
                          valid_data_loader=val_data_loader,
                          test_data_loader=None,
                          target_data_loader=target_data_loader,
                          lr_scheduler=lr_scheduler)
        logger.info(f'Training for k-fold (k={k})')
        best_epoch_log = trainer.train()

        for key, value in total_kfold_performance.items():
            if key in tune_metrics:
                total_kfold_performance[key] = best_epoch_log[key] + value

        k = k + 1

    tune.report(loss=total_kfold_performance['val_loss']/fold_count,
                accuracy=total_kfold_performance['val_accuracy']/fold_count,
                roc_auc=total_kfold_performance['val_roc_auc']/fold_count,
                f1=total_kfold_performance['val_f1']/fold_count,
                epoch=best_epoch_log['epoch'])

def train_test(project_config, best_trial_config={"lr":0.003269242203193986,
                                                  "momentum":0.5628770732845748}):
    logger = project_config.get_logger('train')
    # setup data_loader instances
    data_loader_factory = project_config.init_obj('data_loader', module_data)
    # build model architecture, then print to console
    model = project_config.init_obj('arch', module_arch)
    logger.info('Model loaded.')
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(project_config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # get function handles of loss and metrics
    criterion = project_config.init_obj('loss', module_loss)
    criterion.to(device)

    metrics = [getattr(module_metric, met) for met in project_config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    model_trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # criterion_trainable_params = filter(lambda p: p.requires_grad, criterion.parameters())
    # trainable_params = list(model_trainable_params)
    # trainable_params.extend(list(criterion_trainable_params))
    # optimizer = optim.SGD([{'params': model_trainable_params},
    #             {'params': criterion_trainable_params}], lr=best_trial_config["lr"],
    #                       momentum=best_trial_config["momentum"])
    optimizer = optim.SGD(model_trainable_params, lr=best_trial_config["lr"],
                          momentum=best_trial_config["momentum"])

    lr_scheduler = project_config.init_obj('lr_scheduler', torch.optim.lr_scheduler,
                                   optimizer)

    task_datasets = data_loader_factory.get_datasets()
    train_indices = []
    for dataset in task_datasets:
        dataset = dataset.to_numpy()
        train_indices.append(np.arange(len(dataset)))

    train_data_loader = data_loader_factory.get_train_dataloader(train_indices)

    # train_indices = np.arange(len(task_datasets))
    train_data_loader = data_loader_factory.get_train_dataloader(train_indices)
    test_data_loader = data_loader_factory.get_test_dataloader()
    target_data_loader = data_loader_factory.get_target_dataloader()
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=project_config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=None,
                      test_data_loader=test_data_loader,
                      target_data_loader=target_data_loader,
                      lr_scheduler=lr_scheduler,
                      # save_checkpoint=True
                      )
    trainer.train()
    if test_data_loader is not None:
        trainer.test()

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

    pp = pprint.PrettyPrinter(indent=4, depth=6, sort_dicts=False)
    pp.pprint(f'The project conifguration is :{dict(project_config.config.items())}')
    best_trial = search_hyperparameters(project_config)
    # project_config['trainer']['epochs'] = best_trial.last_result['epoch']
    train_test(project_config, best_trial.config)
    # train_test(project_config)