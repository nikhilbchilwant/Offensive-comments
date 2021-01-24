import re

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from base import BaseDataLoader
from dataloader_util import get_reduced_data, get_val_dataloader, \
    get_train_dataloader, clean_text


class GermEvalDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, tokenizer_name,
                 validation_split=0.0, num_workers=1, data_red_factor=1,
                 val_data_dir=None):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        germ_eval = pd.read_table(data_dir)
        germ_eval = get_reduced_data(germ_eval, data_red_factor)
        n_samples = len(germ_eval)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        germ_eval = self._format_columns(germ_eval)

        if val_data_dir is None:
            self.train_data, self.validate_data = np.split(
                germ_eval.sample(frac=1),
                [int((1.0 - validation_split) * n_samples)])
        else:
            self.train_data = germ_eval

            germ_eval_val = pd.read_table(val_data_dir)
            germ_eval_val = get_reduced_data(germ_eval_val, data_red_factor)
            self.validate_data = self._format_columns(germ_eval_val)

    def get_train_dataloader(self):
        return get_train_dataloader(self.train_data, self.tokenizer,
                                    self.batch_size, self.num_workers)

    def get_val_dataloader(self):
        return get_val_dataloader(self.validate_data, self.tokenizer,
                                  self.batch_size, self.num_workers)

    def _format_columns(self, data_series):
        data_series.columns = ['comment_text', 'label_1', 'label_2']
        data_series['comment_text'] = data_series['comment_text'].apply(
            lambda x: clean_text(x))
        data_series = data_series.replace(
            {'OFFENSE': 1, 'INSULT': 1, 'ABUSE': 1, 'PROFANITY': 1, 'OTHER': 0})
        data_series['toxic_label_max'] = data_series[['label_1', 'label_2']].apply(
            lambda x: np.max(x), axis=1)
        return data_series