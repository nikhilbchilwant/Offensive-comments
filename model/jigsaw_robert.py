import torch.nn as nn
from transformers import RobertaModel

from base import BaseModel


class JigsawRoBERTmodel(BaseModel):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        roberta_model_name = pretrained_model_name
        self.bert = RobertaModel.from_pretrained(roberta_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0][:, 0]
        output = self.fc(output)
        return output
