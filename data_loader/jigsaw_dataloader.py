import numpy as np
import pandas as pd
from transformers import BertTokenizer

from base import BaseDataLoader
from . import get_reduced_data, get_balanced_dataloader, \
    get_dataloader


class JigsawDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, tokenizer_name,
                 validation_split=0.0, num_workers=1, data_red_factor=1):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        jigsaw_data = pd.read_csv(data_dir)
        jigsaw_data = get_reduced_data(jigsaw_data, data_red_factor)
        n_samples = len(jigsaw_data)
        jigsaw_data['toxic_label_max'] = jigsaw_data[
            ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
             'identity_hate']].apply(lambda x: np.max(x), axis=1)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.train_data, self.validate_data = np.split(
            jigsaw_data.sample(frac=1),
            [int((1.0 - validation_split) * n_samples)])

    def get_train_dataloader(self):
        return get_balanced_dataloader(self.train_data, self.tokenizer,
                                       self.batch_size, self.num_workers)

    def get_val_dataloader(self):
        return get_dataloader(self.validate_data, self.tokenizer,
                              self.batch_size, self.num_workers)
