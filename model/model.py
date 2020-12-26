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

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0][:,0]
        output = self.fc(output)
        return output

class JigsawDropoutBERTmodel(BaseModel):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0][:,0]
        output = self.drop(output)
        output = self.fc(output)
        return output