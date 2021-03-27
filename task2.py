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
from torch.utils import data
from os.path import join as pjoin
from torchvision import transforms


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,0,3,4,5,6,7"
torch.multiprocessing.set_sharing_strategy('file_system')

def train_model(device, train_loader, val_loader, test_loader):
    MODEL_PATH = "./models/task2/"
    model = R2U_Net()
    model = nn.DataParallel(model)
    task2 = task2_train(device, train_loader, val_loader, test_loader, model, MODEL_PATH)

    EPOCHS = 150

    for epoch in range(EPOCHS):
        print("====================> Model Training Epoch: " + str(epoch))
        task2.train()
        # task2.validate()

    task2.saveModel()

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.Cityscapes(
        './data/cityscapes', split='train', mode='fine', target_type='semantic', target_transform=transform, transform=transform)

    val_dataset = torchvision.datasets.Cityscapes(
        './data/cityscapes', split='val', mode='fine', target_type='semantic', target_transform=transform, transform=transform)

    test_dataset = torchvision.datasets.Cityscapes(
        './data/cityscapes', split='test', mode='fine', target_type='semantic', target_transform=transform, transform=transform)

    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=2, shuffle=True)

    val_loader = data.DataLoader(
        dataset=val_dataset, batch_size=32, shuffle=True)
    
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=True)

    train_model(device, train_loader, val_loader, test_loader)
    