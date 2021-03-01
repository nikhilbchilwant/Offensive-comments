import re

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from base import BaseDataLoader
from . import get_reduced_data, get_dataloader, \
    get_balanced_dataloader, clean_text

from util import *

class GermEvalDataLoader(BaseDataLoader):
    def __init__(self, data_dirs, test_dir, target_domain_dir, batch_size, tokenizer_name,
                 num_workers=1, multi_factor=1.0):
        self.data_dirs = data_dirs
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.datasets = []
        for data_dir in data_dirs:            
            dataset = pd.read_csv(data_dir)
            dataset = get_reduced_data(dataset, multi_factor)
            self.datasets.append(dataset)
        
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        self.eternio_test = pd.read_table(test_dir)
        self.eternio_test = self._format_eternio(self.eternio_test)

        self.eternio_target = pd.read_table(target_domain_dir)
        self.eternio_target = self._format_eternio(self.eternio_test)

    def get_train_dataloader(self, train_indices, task_names):
        dataset_index = 0
        sliced_datasets = {}
        assert (len(train_indices) == len(self.datasets)),'Dimension msimatch.'

        for dataset_index in range(0, len(self.datasets)):
            sliced_datasets.update({task_names[dataset_index] : 
            self.datasets[dataset_index].take(train_indices[dataset_index])})

        return get_balanced_dataloader(sliced_datasets,
                                       self.tokenizer, self.batch_size, self.num_workers)

    def get_val_dataloader(self, val_indices, task_names):
        dataset_index = 0
        sliced_datasets = {}
        for dataset_index in range(0, len(self.datasets)):
            sliced_datasets.update({task_names[dataset_index] :
             self.datasets[dataset_index].take(val_indices[dataset_index])})

        return get_balanced_dataloader(sliced_datasets, self.tokenizer,
                              self.batch_size, self.num_workers)

    def get_test_dataloader(self):
        return get_dataloader([self.eternio_test], self.tokenizer,
                              self.batch_size, self.num_workers)

    def get_target_dataloader(self):
        return get_dataloader([self.eternio_target], self.tokenizer,
                            self.batch_size, self.num_workers)

    def get_datasets(self):
        return self.datasets

    def _format_germ_eval(self, data_series):
        data_series.columns = ['comment_text', 'label_1', 'label_2']
        data_series['comment_text'] = data_series['comment_text'].apply(
            lambda x: clean_text(x))
        data_series = data_series.replace(
            {'OFFENSE': 1, 'INSULT': 1, 'ABUSE': 1, 'PROFANITY': 1, 'OTHER': 0})
        data_series['toxic_label_max'] = data_series[['label_1', 'label_2']].apply(
            lambda x: np.max(x), axis=1)
        return data_series

    def _format_eternio(self, data_series):
        # data_series.columns = ['id' , 'comment_text', 'toxic_label_max']
        data_series['comment_text'] = data_series['comment_text'].apply(
            lambda x: clean_text(x))
        data_series['toxic_label_max'] = data_series['toxic_label_max']
        return data_series