import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import BertModel
from transformers import RobertaModel
from .sentiment import SentimentModel
from torch.nn import Softmax

class JigsawBERTmodel(BaseModel):
    def __init__(self, pretrained_model_names, num_classes):
        super().__init__()
        bert_model_name = pretrained_model_names[0]
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0][:,0]
        output = self.fc(output)
        return output

class JigsawRoBERTmodel(BaseModel):
    def __init__(self, pretrained_model_names, num_classes):
        super().__init__()
        roberta_model_name = pretrained_model_names[0]
        self.bert = RobertaModel.from_pretrained(roberta_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0][:,0]
        output = self.fc(output)
        return output

class JigsawDropoutBERTmodel(BaseModel):
    def __init__(self, pretrained_model_names, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_names[0])
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        # self.dropx = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = output[0][:,0]
        # output = self.drop(output)
        output = self.fc(output)
        return output

class BERTSentimentEnsemble(BaseModel):
    def __init__(self, pretrained_model_names, num_classes):
        super().__init__()
        self.bert = JigsawBERTmodel(pretrained_model_names, num_classes)
        self.sentiment = SentimentModel()
        self.sm = Softmax(dim=1) 
        self.classifier = nn.Linear(5, num_classes)

    def forward(self, input_ids, attention_mask):
        offensive_output = self.sm(self.bert(input_ids, attention_mask))
        sentiment_output = F.softmax(self.sentiment(input_ids).logits, dim=1)

        x = torch.cat((offensive_output, sentiment_output), dim=1)
        x = self.classifier(F.relu(x))
        return x