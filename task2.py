import os
import glob
import json
import torch
import pickle
import imageio
import functools
import collections
import torchvision
import numpy as np
import scipy.io as io
import torch.nn as nn
import scipy.misc as m
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from train import *
from model import *
from PIL import Image
from tqdm import tqdm
from get_dataset import *
from torch.utils import data
from os.path import join as pjoin
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP


os.environ["CUDA_VISIBLE_DEVICES"] = "2,6,5,4,1"
torch.multiprocessing.set_sharing_strategy('file_system')

def train_model(device, train_loader, val_loader=None):
    MODEL_PATH = "./models/task2/"
    model = R2U_Net()
    model = nn.DataParallel(model)
    
    task2 = task2_train(device, train_loader, val_loader, model, MODEL_PATH)

    EPOCHS = 50

    for epoch in range(EPOCHS):
        print("====================> Model Training Epoch: " + str(epoch))
        task2.train()

    task2.saveModel()

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_VARS = np.array((1, 1, 1), dtype=np.float32)

    trainset = Cityscapes(root='./data/cityscapes', list_path="./data/cityscapes/train.txt", crop_size=(128, 256), mean=IMG_MEAN, vars=IMG_VARS,
                        scale=True, mirror=False)
    train_loader = data.DataLoader(trainset, batch_size=8, shuffle=False, pin_memory=True)

    # for a, b in train_loader:
    #     print(a.shape)
    #     print(b.tolist())

    #     break

    train_model(device, train_loader)
    