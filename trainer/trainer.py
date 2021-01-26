import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from sklearn.metrics import roc_auc_score
from torch.nn import Softmax
from model import JigsawBERTmodel
from model.sentiment import SentimentModel
from ray import tune

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None):
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
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.softmax = Softmax(dim=1)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for (batch_idx, batch_data) in enumerate(self.data_loader):
            input_ids = batch_data.get("input_ids").to(self.device)
            attention_mask = batch_data.get("attention_mask").to(self.device)
            target = batch_data.get("targets").to(self.device)
            
            self.optimizer.zero_grad()

            output = self.model(input_ids, attention_mask)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, roc_auc = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update({'roc_auc':roc_auc})

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
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_data_loader):
                input_ids = batch_data.get("input_ids").to(self.device)
                attention_mask = batch_data.get("attention_mask").to(self.device)
                target = batch_data.get("targets").to(self.device)

                output = self.model(input_ids, attention_mask)
                loss = self.criterion(output, target)
                pred_prob = self.softmax(output)
                _, pred_label = torch.max(output, dim=1)
                toxic_prob = toxic_prob + pred_prob[:,1].tolist()
                target_labels = target_labels + target.tolist()

                self.valid_metrics.update('loss', loss.item())
                val_loss += loss.cpu().numpy()
                val_steps = val_steps + 1
                correct += (pred_label == target).sum().item()
                total += pred_label.size(0)

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        try:
            roc_auc = roc_auc_score(target_labels, toxic_prob)
        except ValueError:
            roc_auc = -1

        return self.valid_metrics.result(), roc_auc

    def _test(self):
        self.model.eval()
        self.test_metrics.reset()
        toxic_prob = []
        target_labels = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_data_loader):
                input_ids = batch_data.get("input_ids").to(self.device)
                attention_mask = batch_data.get("attention_mask").to(self.device)
                target = batch_data.get("targets").to(self.device)

                output = self.model(input_ids, attention_mask)
                loss = self.criterion(output, target)
                pred_prob = self.softmax(output)
                _, pred_label = torch.max(output, dim=1)
                toxic_prob = toxic_prob + pred_prob[:,1].tolist()
                target_labels = target_labels + target.tolist()

                self.test_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))


        try:
            roc_auc = roc_auc_score(target_labels, toxic_prob)
        except ValueError:
            roc_auc = -1

        return self.test_metrics.result(), roc_auc

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)