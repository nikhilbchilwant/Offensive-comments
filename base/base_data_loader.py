# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import default_collate
# from torch.utils.data.sampler import SubsetRandomSampler
# import dataset.datasets as dataset_module
from abc import abstractmethod


class BaseDataLoader():

    @abstractmethod
    def get_train_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def get_val_dataloader(self):
        raise NotImplementedError