import numpy as np
import pandas as pd
from transformers import BertTokenizer

from base import BaseDataLoader
from dataloader_util import get_train_dataloader, get_reduced_data, \
    get_val_dataloader


class JigsawUnintendedToxicDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, tokenizer_name,
                 validation_split=0.0, num_workers=1, data_red_factor=1):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        jigsaw_data = pd.read_csv(data_dir)
        jigsaw_data = get_reduced_data(jigsaw_data, data_red_factor)
        n_samples = len(jigsaw_data)
        jigsaw_data['toxic_label_max'] = \
            jigsaw_data[
                ['toxic', 'severe_toxicity', 'obscene', 'threat', 'insult',
                 'identity_attack']] \
                .apply(lambda x: np.max(x), axis=1)
        jigsaw_data[jigsaw_data['toxic_label_max'] > 0] = 1
        jigsaw_data['toxic_label_max'] = jigsaw_data['toxic_label_max'].astype(
            int)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.train_data, self.validate_data = np.split(
            jigsaw_data.sample(frac=1),
            [int((1.0 - validation_split) * n_samples)])

    def get_train_dataloader(self):
        return get_train_dataloader(self.train_data, self.tokenizer,
                                    self.batch_size, self.num_workers)

    def get_val_dataloader(self):
        return get_val_dataloader(self.validate_data, self.tokenizer,
                                  self.batch_size, self.num_workers)