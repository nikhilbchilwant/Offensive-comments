import torch
from torch.utils.data import Dataset

class Toxic_Dataset(Dataset):
    def __init__(self, ys, Xs, tokenizer, weights=None, max_len=128):
        self.targets = ys
        self.comments = Xs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.weights = weights
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