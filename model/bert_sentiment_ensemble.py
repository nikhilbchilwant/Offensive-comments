import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class BERTSentimentEnsemble(BaseModel):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        # self.bert = JigsawBERTmodel(pretrained_model_name, num_classes)
        # self.sentiment = SentimentModel()
        # self.fc1 = nn.Linear(5, num_classes)
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)
        self.fc3 = nn.Linear(3, num_classes)
        # self.bl1 = nn.Bilinear(3,2,2)

    def forward(self, offensive_output, sentiment_output):
        offensive_output = F.softmax(offensive_output, dim=1)
        sentiment_output = F.softmax(sentiment_output, dim=1)
        # output = torch.tensor([offensive_output[0][1], sentiment_output[0][0],
        # sentiment_output[0][1], sentiment_output[0][2], offensive_output[0][0]])
        # output = output.to(offensive_output.device)
        # output = output.unsqueeze(dim=0)
        # offensive_output = F.softmax(self.bert(input_ids, attention_mask))
        # sentiment_output = F.softmax(self.sentiment(input_ids).logits, dim=1)
        positive_output = offensive_output[:, 0]
        negative_output = offensive_output[:, 1]

        positive_output = positive_output.unsqueeze(dim=0)
        negative_output = negative_output.unsqueeze(dim=0)

        positive_output = torch.transpose(positive_output, 0, 1)
        negative_output = torch.transpose(negative_output, 0, 1)

        output = torch.cat((positive_output, sentiment_output), dim=1)
        output = torch.cat((output, negative_output), dim=1)

        # output = torch.cat((offensive_output, sentiment_output), dim=1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        # output = self.bl1(sentiment_output, offensive_output)
        return F.relu(output)
