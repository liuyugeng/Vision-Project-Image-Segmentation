import os
import time
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torchlars import LARS
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

def loss_f(inputs, targets, weight=None):
    n, c, h, w = inputs.size()
    nt, ht, wt = targets.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    targets = targets.view(-1)
    loss = F.cross_entropy(
        inputs, targets, weight=weight, reduction='mean', ignore_index=255
    )
    return loss

class task2_train():
    def __init__(self, device, train_loader, val_loader, model, MODEL_PATH):
        self.device = device
        self.model_path = MODEL_PATH
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model.to(device)
        base_optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = GradScaler()

    def train(self):
        self.model.train()
        train_loss = 0
        

        label_pred = []
        label_true = []

        AC_total, SE_total, SP_total, DC_total, JS_total = 0,0,0,0,0

        start = time.time()
        
        for batch_idx, (inputs, GT) in enumerate(self.train_loader):
            if batch_idx % 10 == 0 and batch_idx:
                end = time.time()
                print("finished training %s images, using %.4fs" % (str(batch_idx*8), end-start))
            inputs, GT = inputs.to(self.device), GT.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                SR = self.model(inputs)
                loss = loss_f(inputs=SR, targets=GT.long())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item()

            label_pred.append(SR.data.max(dim=1)[1].cpu().numpy())
            label_true.append(GT.data.cpu().numpy())

        label_pred = np.concatenate(label_pred, axis=0)
        label_true = np.concatenate(label_true, axis=0)

        for lbt, lbp in zip(label_true, label_pred):
            AC, SE, SP, DC, JS = self.evaluate(lbt, lbp)
            AC_total += AC
            SE_total += SE
            SP_total += SP
            DC_total += DC
            JS_total += JS

        print('Training Loss: %.4f\nAC: %.4f, SE: %.4f, SP: %.4f, DC: %.4f, JS: %.4f' % (train_loss, AC_total/len(label_true), SE_total/len(label_true), SP_total/len(label_true), DC_total/len(label_true), JS_total/len(label_true)))

        

    def _fast_hist(self, truth, pred):
        mask = (truth >= 0) & (truth < 19)
        hist = np.bincount(19 * truth[mask].astype(int) + pred[mask], minlength=19 ** 2).reshape(19, 19)
        return hist


    def evaluate(self, ground_truth, predictions, smooth=1):

        confusion_matrix = np.zeros((19, 19))

        for lt, lp in zip(ground_truth, predictions):
            confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

        fn = confusion_matrix.sum(1) - np.diag(confusion_matrix)
        fp = confusion_matrix.sum(0) - np.diag(confusion_matrix)
        tp = np.diag(confusion_matrix)
        tn = np.array([confusion_matrix.sum() for i in range(19)]) - confusion_matrix.sum(1) - confusion_matrix.sum(0) + np.diag(confusion_matrix)


        AC_array = (tp + tn) / np.maximum(1.0, fn + fp + tp + tn)
        AC = AC_array.mean()
        SE_array = (tp) / np.maximum(1.0, confusion_matrix.sum(1))
        SE = SE_array.mean()
        SP_array = tn / np.maximum(1.0, tn + fp)
        SP = SP_array.mean()
        DC_array = (2*tp) / np.maximum(1.0, confusion_matrix.sum(0) + confusion_matrix.sum(1))
        DC = DC_array.mean()
        JS_array = (tp) / np.maximum(1.0, confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix))

        JS = JS_array.mean()
        

        return AC, SE, SP, DC, JS

    def saveModel(self):
        torch.save(self.model.state_dict(), self.model_path + "model.pth")



class OhemCrossEntropy2dTensor(nn.Module):
    def __init__(self, ignore_label, reduction='elementwise_mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class CriterionOhemDSN(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh=thresh, min_kept=min_kept)
        self.criterion2 = nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2 * 0.4


class task3_train():
    def __init__(self, device, train_loader, val_loader, model, MODEL_PATH):
        self.device = device
        self.model_path = MODEL_PATH
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        self.criterion = CriterionOhemDSN(thresh=0.7, min_kept=100000)
        self.scaler = GradScaler()

    def train(self):
        self.model.train()
        
        train_loss = 0

        for batch_idx, (inputs, GT) in enumerate(self.train_loader):
            inputs, GT = inputs.to(self.device), GT.long().to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                SR = self.model(inputs)
                loss = self.criterion(SR, GT)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item()

        print('Training Loss: %.4f' % (train_loss))

    def saveModel(self):
        torch.save(self.model.state_dict(), self.model_path + "model1.pth")