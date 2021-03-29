import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from train import *
from model import *
from tqdm import tqdm
from get_dataset import *
from torch.utils import data
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
torch.multiprocessing.set_sharing_strategy('file_system')


def train_model(device, train_loader, val_loader):
    MODEL_PATH = "./models/task3/"
    model = DualSeg_res50(training=True)
    model = nn.DataParallel(model)
    task3 = task3_train(device, train_loader, val_loader, model, MODEL_PATH)

    EPOCHS = 50000

    for epoch in range(EPOCHS):
        print("====================> Model Training Epoch: " + str(epoch))
        task3.train()
        # task3.validate()

    task3.saveModel()

def evaluate_model(device, test_loader):
    MODEL_PATH = "./models/task3/"
    model = DualSeg_res50().to(device)
    model.load_state_dict(torch.load(MODEL_PATH + "model.pth"), strict=False)
    model = nn.DataParallel(model)

    AC, SE, SP, DC, JS = eval(model, test_loader, device)

    print('AC: %.4f, SE: %.4f, SP: %.4f, DC: %.4f, JS: %.4f' % (AC, SE, SP, DC, JS))

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_VARS = np.array((1, 1, 1), dtype=np.float32)

    trainset = Cityscapes(root='./data/cityscapes', list_path="./data/cityscapes/train.txt", mean=IMG_MEAN, vars=IMG_VARS, scale=1, mirror=False)
    train_loader = data.DataLoader(trainset, batch_size=8, shuffle=False, pin_memory=True)

    testset = Cityscapes(root='./data/cityscapes', list_path="./data/cityscapes/test.txt", mean=IMG_MEAN, vars=IMG_VARS, scale=1, mirror=False)
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    # train_model(device, train_loader, val_loader=None)
    evaluate_model(device, test_loader)