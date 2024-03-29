import re

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from base import BaseDataLoader
from . import get_reduced_data, get_dataloader, \
    get_balanced_dataloader, clean_text

class GermEvalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, test_dir, batch_size, tokenizer_name,
                 num_workers=1, data_red_factor=1):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.germ_eval = pd.read_table(data_dir)
        self.germ_eval = get_reduced_data(self.germ_eval, data_red_factor)
        n_samples = len(self.germ_eval)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.germ_eval = self._format_germ_eval(self.germ_eval)

        self.eternio_test = pd.read_csv(test_dir)
        self.eternio_test = self._format_eternio(self.eternio_test)

    def get_train_dataloader(self, train_indices):
        return get_balanced_dataloader(self.germ_eval.take(train_indices),
                                       self.tokenizer, self.batch_size, self.num_workers)

    def get_val_dataloader(self, val_indices):
        return get_dataloader(self.germ_eval.take(val_indices), self.tokenizer,
                              self.batch_size, self.num_workers)

    def get_test_dataloader(self):
        return get_dataloader(self.eternio_test, self.tokenizer,
                              self.batch_size, self.num_workers)

    def get_data(self):
        return self.germ_eval


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
        data_series.columns = ['id' , 'comment_text', 'toxic_label_max']
        data_series['comment_text'] = data_series['comment_text'].apply(
            lambda x: clean_text(x))
        data_series['toxic_label_max'] = data_series['toxic_label_max']
        return data_series