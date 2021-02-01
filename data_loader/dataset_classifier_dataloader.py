import re

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from base import BaseDataLoader
from . import get_reduced_data, get_dataloader, \
    get_balanced_dataloader, clean_text

class DatasetClassifierDataloader(BaseDataLoader):
    def __init__(self, data_dir, test_dir, batch_size, tokenizer_name,
                 num_workers=1, data_red_factor=1):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.germ_eval_eternio = pd.read_table(data_dir, encoding='utf-8')
        self.germ_eval_eternio = get_reduced_data(self.germ_eval_eternio, data_red_factor)
        # n_samples = len(self.germ_eval)
        self.germ_eval = pd.read_table(data_dir, encoding='utf-8')
        self.germ_eval = self._format_germ_eval(self.germ_eval)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def get_train_dataloader(self, train_indices):
        return get_balanced_dataloader(self.germ_eval_eternio.take(train_indices),
                                       self.tokenizer, self.batch_size, self.num_workers)

    def get_val_dataloader(self, val_indices):
        return get_dataloader(self.germ_eval_eternio.take(val_indices), self.tokenizer,
                              self.batch_size, self.num_workers)

    def get_test_dataloader(self):
        return get_dataloader(self.germ_eval_eternio, self.tokenizer,
                              self.batch_size, self.num_workers)

    def get_data(self):
        return self.germ_eval_eternio

    def _format_germ_eval(self, data_series):
        data_series.columns = ['comment_text', 'label_1', 'label_2']
        data_series['comment_text'] = data_series['comment_text'].apply(
            lambda x: clean_text(x))
        data_series = data_series.replace(
            {'OFFENSE': 1, 'INSULT': 1, 'ABUSE': 1, 'PROFANITY': 1, 'OTHER': 0})
        data_series['toxic_label_max'] = data_series[['label_1', 'label_2']].apply(
            lambda x: np.max(x), axis=1)
        data_series['dataset_label'] = np.zeros(len(data_series['comment_text']),
                                          dtype=np.int8)
        return data_series