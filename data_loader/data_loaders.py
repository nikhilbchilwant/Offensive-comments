from base import BaseDataLoader
from torch.utils.data import DataLoader
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import WeightedRandomSampler
from dataset.datasets import Toxic_Dataset
import re

def get_train_dataloader(train_data, tokenizer, batch_size, num_workers):
  train_labels = train_data['toxic_label_max']
  train_comments = train_data['comment_text']
  train_labels, train_comments = train_labels.to_list(), train_comments.to_list()

  class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
  train_samples_weight = 1. / class_sample_count
  # print('Class sample count: ', class_sample_count)
  train_samples_weight = np.array([train_samples_weight[label] for label in train_labels])
  train_samples_weight = torch.from_numpy(train_samples_weight)
  train_samples_weight = train_samples_weight.double()
  train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight))

  train_dataset = Toxic_Dataset(train_labels, train_comments, tokenizer)
  
  train_init_kwargs = {
    'dataset': train_dataset,
    'batch_size': batch_size,
    'shuffle': False,
    # 'collate_fn': collate_fn,
    'num_workers': num_workers
  }
  
  return DataLoader(**train_init_kwargs)

def get_val_dataloader(validate_data, tokenizer, batch_size, num_workers):
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
  return data_series[0:(int)(len(data_series)/data_reduction_fac)]

class JigsawDataLoader(BaseDataLoader):

  def __init__(self, data_dir, batch_size, tokenizer_name, validation_split=0.0, num_workers=1, data_red_factor=1):
      self.data_dir = data_dir
      self.num_workers = num_workers        
      self.batch_size = batch_size

      jigsaw_data = pd.read_csv(data_dir)
      jigsaw_data = get_reduced_data(jigsaw_data, data_red_factor)
      n_samples = len(jigsaw_data)
      jigsaw_data['toxic_label_max'] = jigsaw_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].apply(lambda x: np.max(x), axis=1)
      
      self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
      self.train_data, self.validate_data = np.split(jigsaw_data.sample(frac=1),
                                [int((1.0 - validation_split) * n_samples)])

  def get_train_dataloader(self):
    return get_train_dataloader(self.train_data, self.tokenizer, self.batch_size, self.num_workers)

  def get_val_dataloader(self):
    return get_val_dataloader(self.validate_data, self.tokenizer, self.batch_size, self.num_workers)

class JigsawUnintendedToxicDataLoader(BaseDataLoader):
  def __init__(self, data_dir, batch_size, tokenizer_name, validation_split=0.0, num_workers=1, data_red_factor=1):
      self.data_dir = data_dir
      self.num_workers = num_workers        
      self.batch_size = batch_size

      jigsaw_data = pd.read_csv(data_dir)
      jigsaw_data = get_reduced_data(jigsaw_data, data_red_factor)
      n_samples = len(jigsaw_data)
      jigsaw_data['toxic_label_max'] = \
        jigsaw_data[['toxic', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']] \
        .apply(lambda x: np.max(x), axis=1)   
      jigsaw_data[jigsaw_data['toxic_label_max'] > 0] = 1
      jigsaw_data['toxic_label_max'] = jigsaw_data['toxic_label_max'].astype(int)

      self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
      self.train_data, self.validate_data = np.split(jigsaw_data.sample(frac=1),
                                [int((1.0 - validation_split) * n_samples)])

  def get_train_dataloader(self):
    return get_train_dataloader(self.train_data, self.tokenizer, self.batch_size, self.num_workers)

  def get_val_dataloader(self):
    return get_val_dataloader(self.validate_data, self.tokenizer, self.batch_size, self.num_workers)

class GermEvalDataLoader(BaseDataLoader):
  def __init__(self, data_dir, batch_size, tokenizer_name, validation_split=0.0, num_workers=1, data_red_factor=1, val_data_dir=None):
    self.data_dir = data_dir
    self.num_workers = num_workers        
    self.batch_size = batch_size

    germ_eval = pd.read_table(data_dir)
    germ_eval = get_reduced_data(germ_eval, data_red_factor)
    n_samples = len(germ_eval)

    self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    germ_eval = _format_columns(germ_eval)

    if val_data_dir is None:
      self.train_data, self.validate_data = np.split(germ_eval.sample(frac=1),
                                [int((1.0 - validation_split) * n_samples)])
    else:
      self.train_data = germ_eval
      
      germ_eval_val = pd.read_table(val_data_dir)
      germ_eval_val = get_reduced_data(germ_eval_val, data_red_factor)
      self.validate_data = _format_columns(germ_eval_val)
      

  def get_train_dataloader(self):
    return get_train_dataloader(self.train_data, self.tokenizer, self.batch_size, self.num_workers)

  def get_val_dataloader(self):
    return get_val_dataloader(self.validate_data, self.tokenizer, self.batch_size, self.num_workers)

def _clean_tweet(tweet):
  return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())

def _format_columns(data_series):
    data_series.columns = ['comment_text', 'label_1', 'label_2']
    data_series['comment_text'] = data_series['comment_text'].apply(lambda x: _clean_tweet(x))
    data_series = data_series.replace({'OFFENSE':1, 'INSULT':1, 'ABUSE':1, 'PROFANITY':1, 'OTHER':0})
    data_series['toxic_label_max'] = data_series[['label_1', 'label_2']].apply(lambda x: np.max(x), axis=1)
    return data_series


