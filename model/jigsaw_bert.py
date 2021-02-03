import torch.nn as nn
from transformers import BertModel

from base import BaseModel


class JigsawBERTmodel(BaseModel):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        bert_model_name = pretrained_model_name
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        latent = self.bert(input_ids, attention_mask)
        output = latent[0][:, 0]
        output = self.fc(output)
        return output, latent.last_hidden_state