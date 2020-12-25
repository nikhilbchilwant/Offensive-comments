import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import BertModel


class JigsawBERTmodel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2) #2 classes: toxic, non-toxic
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output= output[0][:,0]
        output = self.fc(output)
        return output #self.sm(output)
