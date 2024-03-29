import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from dataset.datasets import Toxic_Dataset


def get_balanced_dataloader(train_data, tokenizer, batch_size, num_workers):
    train_labels = train_data['toxic_label_max']
    train_comments = train_data['comment_text']
    train_labels, train_comments = train_labels.to_list(), train_comments.to_list()

    class_sample_count = np.array(
        [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    train_samples_weight = 1. / class_sample_count
    # print('Class sample count: ', class_sample_count)
    train_samples_weight = np.array(
        [train_samples_weight[label] for label in train_labels])
    train_samples_weight = torch.from_numpy(train_samples_weight)
    train_samples_weight = train_samples_weight.double()
    train_sampler = WeightedRandomSampler(train_samples_weight,
                                          len(train_samples_weight))

    train_dataset = Toxic_Dataset(train_labels, train_comments, tokenizer)

    train_init_kwargs = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'shuffle': False,
        # 'collate_fn': collate_fn,
        'num_workers': num_workers
    }

    return DataLoader(**train_init_kwargs)


def get_dataloader(validate_data, tokenizer, batch_size, num_workers):
    val_labels = validate_data['toxic_label_max']
    val_comments = validate_data['comment_text']
    val_labels, val_comments = val_labels.to_list(), val_comments.to_list()

    val_dataset = Toxic_Dataset(val_labels, val_comments, tokenizer)

    val_init_kwargs = {
        'dataset': val_dataset,
        'batch_size': batch_size,
        'shuffle': False,
        # 'collate_fn': collate_fn,
        'num_workers': num_workers
    }

    return DataLoader(**val_init_kwargs)


def get_reduced_data(data_series, data_reduction_fac):
    return data_series[0:(int)(len(data_series) / data_reduction_fac)]

def clean_text(text):
    return ' '.join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ",
               text).split())
