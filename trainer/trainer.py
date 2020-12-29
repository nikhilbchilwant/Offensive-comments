import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from sklearn.metrics import roc_auc_score
from torch.nn import Softmax
from model import JigsawBERTmodel
from model.sentiment import SentimentModel

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.softmax = Softmax(dim=1)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        ### temporary code
        # self.bert = JigsawBERTmodel(["bert-base-german-cased"], 2)
        # bert_checkpoint = torch.load("/data/users/nchilwant/training_output/models/germ-eval/1228_143410/model_best.pth")
        # bert_state_dict = super()._remove_module_prefix(bert_checkpoint['state_dict'])
        # self.bert.load_state_dict(bert_state_dict)
        # self.bert.to(self.device)
        # self.sentiment = SentimentModel()
        # self.sentiment = self.sentiment.to(self.device)
        ### end temporary code

        for (batch_idx, batch_data) in enumerate(self.data_loader):
            input_ids = batch_data.get("input_ids").to(self.device)
            attention_mask = batch_data.get("attention_mask").to(self.device)
            target = batch_data.get("targets").to(self.device)
            
            self.optimizer.zero_grad()
            
            # bert_output = self.bert(input_ids, attention_mask)
            # bert_output = bert_output.to(self.device)
            # sentiment_output = self.sentiment(input_ids).logits
            # sentiment_output = sentiment_output.to(self.device)
            # output = self.model(bert_output, sentiment_output)
            output = self.model(input_ids, attention_mask)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, roc_auc = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update({'ROC AUC':roc_auc})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation and ROC AUC score.
        """
        self.model.eval()
        self.valid_metrics.reset()
        toxic_prob = []
        target_labels = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_data_loader):
                input_ids = batch_data.get("input_ids").to(self.device)
                attention_mask = batch_data.get("attention_mask").to(self.device)
                target = batch_data.get("targets").to(self.device)

                # bert_output = self.bert(input_ids, attention_mask)
                # bert_output = bert_output.to(self.device)
                # sentiment_output = self.sentiment(input_ids).logits
                # sentiment_output = sentiment_output.to(self.device)
                # output = self.model(bert_output, sentiment_output)

                # output = self.model(self.bert(input_ids, attention_mask), self.sentiment(input_ids).logits)
                output = self.model(input_ids, attention_mask)
                loss = self.criterion(output, target)
                pred_prob = self.softmax(output)
                _, pred_label = torch.max(output, dim=1)
                toxic_prob = toxic_prob + pred_prob[:,1].tolist()
                target_labels = target_labels + target.tolist()

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))


        try:
            roc_auc = roc_auc_score(target_labels, toxic_prob)
        except ValueError:
            roc_auc = -1

        return self.valid_metrics.result(), roc_auc

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)