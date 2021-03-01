import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from dataset.datasets import Toxic_Dataset
from torch.utils.data.dataset import ConcatDataset
from data_loader.batch_sampler import BalancedBatchSchedulerSampler


def get_balanced_dataloader(train_datasets, tokenizer, batch_size, num_workers):
    task_train_datasets = []
    for task_name, train_dataset in train_datasets.items():
        train_labels = train_dataset['toxic_label_max']
        train_comments = train_dataset['comment_text']
        train_labels, train_comments = train_labels.to_list(), train_comments.to_list()  
        class_sample_count = np.array(
            [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        train_samples_weight = 1. / class_sample_count
        # print('Class sample count: ', class_sample_count)
        train_samples_weight = np.array(
            [train_samples_weight[label] for label in train_labels])
        train_samples_weight = torch.from_numpy(train_samples_weight)
        train_samples_weight = train_samples_weight.double()
        # train_sampler = WeightedRandomSampler(train_samples_weight,
                                            #   len(train_samples_weight))
        train_dataset = Toxic_Dataset(train_labels, train_comments, tokenizer, task_name=task_name, weights=train_samples_weight)
        task_train_datasets.append(train_dataset)

    task_datasets = ConcatDataset(task_train_datasets) 
    batch_sampler = BalancedBatchSchedulerSampler(dataset=task_datasets, batch_size=batch_size)

    train_init_kwargs = {
        'dataset': task_datasets,
        'batch_size': batch_size,
        'shuffle': False,
        # 'collate_fn': collate_fn,
        'num_workers': num_workers,
        'sampler':batch_sampler
    }

    return DataLoader(**train_init_kwargs)


def get_dataloader(datasets, tokenizer, batch_size, num_workers):
    task_val_datasets = []
    for dataset in datasets:
        val_labels = dataset['toxic_label_max']
        val_comments = dataset['comment_text']
        val_labels, val_comments = val_labels.to_list(), val_comments.to_list()
        dataset = Toxic_Dataset(val_labels, val_comments, tokenizer)
        task_val_datasets.append(dataset)

    task_datasets = ConcatDataset(task_val_datasets)
    batch_sampler = BalancedBatchSchedulerSampler(dataset=task_datasets, batch_size=batch_size)

    val_init_kwargs = {
        'dataset': task_datasets,
        'batch_size': batch_size,
        'shuffle': False,
        # 'collate_fn': collate_fn,
        'num_workers': num_workers,
        'sampler': batch_sampler
    }

    return DataLoader(**val_init_kwargs)


def get_reduced_data(data_series, multi_factor):
    if multi_factor>1.0:
        multi_factor = 1.0
        
    return data_series[0:(int)(len(data_series) * multi_factor)]

def clean_text(text):
    return ' '.join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-zäöüÄÖÜß \t])|(\w+:\/\/\S+)", " ",
               text).split())
