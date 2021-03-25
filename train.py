import torch
import numpy as np
import torchvision
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable


class task2_train():
    def __init__(self, device, train_loader, val_loader, test_loader, model, MODEL_PATH):
        self.device = device
        self.model_path = MODEL_PATH
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.criterion = torch.nn.BCELoss()

    def get_accuracy(self, SR, GT, threshold=0.5):
        SR = SR > threshold
        GT = GT == torch.max(GT)
        corr = torch.sum(SR==GT)
        tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
        AC = float(corr)/float(tensor_size)

        return AC

    def get_sensitivity(self, SR, GT, threshold=0.5):
        # Sensitivity == Recall
        SR = SR > threshold
        GT = GT == torch.max(GT)

        # TP : True Positive
        # FN : False Negative
        TP = ((SR==1)+(GT==1))==2
        FN = ((SR==0)+(GT==1))==2

        SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
        
        return SE

    def get_specificity(self, SR, GT, threshold=0.5):
        SR = SR > threshold
        GT = GT == torch.max(GT)

        # TN : True Negative
        # FP : False Positive
        TN = ((SR==0)+(GT==0))==2
        FP = ((SR==1)+(GT==0))==2

        SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
        
        return SP

    def get_precision(self, SR, GT, threshold=0.5):
        SR = SR > threshold
        GT = GT == torch.max(GT)

        # TP : True Positive
        # FP : False Positive
        TP = ((SR==1)+(GT==1))==2
        FP = ((SR==1)+(GT==0))==2

        PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

        return PC

    def get_F1(self, SR, GT, threshold=0.5):
        # Sensitivity == Recall
        SE = self.get_sensitivity(SR,GT,threshold=threshold)
        PC = self.get_precision(SR,GT,threshold=threshold)

        F1 = 2*SE*PC/(SE+PC + 1e-6)

        return F1

    def get_JS(self, SR, GT, threshold=0.5):
        # JS : Jaccard similarity
        SR = SR > threshold
        GT = GT == torch.max(GT)
        
        Inter = torch.sum((SR+GT)==2)
        Union = torch.sum((SR+GT)>=1)
        
        JS = float(Inter)/(float(Union) + 1e-6)
        
        return JS

    def get_DC(self, SR, GT, threshold=0.5):
        # DC : Dice Coefficient
        SR = SR > threshold
        GT = GT == torch.max(GT)

        Inter = torch.sum((SR+GT)==2)
        DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

        return DC

    def train(self):
        self.model.train()
        
        AC = 0.                     # Accuracy
        SE = 0.                     # Sensitivity (Recall)
        SP = 0.                     # Specificity
        PC = 0.                     # Precision
        F1 = 0.                     # F1 Score
        JS = 0.                     # Jaccard Similarity
        DC = 0.                     # Dice Coefficient
        length = 0
        train_loss = 0
        
        for batch_idx, (inputs, GT) in enumerate(self.train_loader):
            inputs, GT = inputs.to(self.device), GT.to(self.device)

            SR = self.model(inputs)
            SR_probs = F.sigmoid(SR)
            SR_flat = SR_probs.view(SR_probs.size(0), -1)

            GT_flat = GT.view(GT.size(0), -1)
            loss = self.criterion(SR_flat,GT_flat)
            train_loss += loss.item()

            # Backprop + optimize
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            AC += self.get_accuracy(SR,GT)
            SE += self.get_sensitivity(SR,GT)
            SP += self.get_specificity(SR,GT)
            PC += self.get_precision(SR,GT)
            F1 += self.get_F1(SR,GT)
            JS += self.get_JS(SR,GT)
            DC += self.get_DC(SR,GT)
            length += inputs.size(0)

        AC = AC/length
        SE = SE/length
        SP = SP/length
        PC = PC/length
        F1 = F1/length
        JS = JS/length
        DC = DC/length

        print('Training Loss: %.4f\nAC: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (train_loss,AC,SE,SP,PC,F1,JS,DC))