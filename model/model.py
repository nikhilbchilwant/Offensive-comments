import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import BertModel


class JigsawBERTmodel(BaseModel):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output= output[0][:,0]
        output = self.fc(output)
        return output #self.sm(output)
