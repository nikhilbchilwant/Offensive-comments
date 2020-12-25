from base import BaseDataLoader
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np

MAX_LEN = 128
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

class Toxic_Dataset(Dataset):
    def __init__(self, ys, Xs, tokenizer, max_len):
        self.targets = ys
        self.comments = Xs
        self.tokenizer = tokenizer
        self.max_len = max_len
        # print('self.comments : ', self.comments)
    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        # print('idx=', idx, 'len(self.comments)=',len(self.comments),'\n')
        comment = str(self.comments[idx])
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
          comment,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )
        return {
          'comment_text': comment,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

class JigsawDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        toxic_comment = pd.read_csv(data_dir)
        toxic_comment['toxic_label_max'] = toxic_comment[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].apply(lambda x: np.max(x), axis=1)
        toxic_comment = toxic_comment[0:(int)(len(toxic_comment)/10000)]
        toxic_comment = toxic_comment.sample(frac=1.0, replace=False)

        labels, comments = toxic_comment['toxic_label_max'], toxic_comment['comment_text']
        #Converting panda series data to list. Otherwise it was raising exception during the training step.
        labels, comments = labels.to_list(), comments.to_list()

        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.dataset = Toxic_Dataset(labels, comments, tokenizer, MAX_LEN)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
