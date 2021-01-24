import re

import torch
from transformers import BertForSequenceClassification

from base import BaseModel


class SentimentModel(BaseModel):
    def __init__(self,
                 pretrained_model_names=["oliverguhr/german-sentiment-bert"],
                 num_classes=3):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_names[0])
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    # def predict_sentiment(self, texts: List[str])-> List[str]:
    # texts = [self.clean_text(text) for text in texts]
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    # input_ids = self.tokenizer.batch_encode_plus(texts,padding=True, add_special_tokens=True)
    # input_ids = torch.tensor(input_ids["input_ids"])
    def forward(self, input_ids):
        output = torch.empty(0, 0)
        # input_ids = torch.unsqueeze(input_ids, 0)
        with torch.no_grad():
            output = self.model(input_ids)

        # label_ids = torch.argmax(logits[0], axis=1)

        # labels = [self.model.config.id2label[label_id] for label_id in label_ids.tolist()]
        return output
