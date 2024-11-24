import os.path

import numpy as np
import torchvision.transforms
from torch.utils.data import DataLoader
import torch.utils.data.dataloader
from torch import nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchsummary
import util as utl
import cv2
#%%

SAVE_DIR = os.path.join(os.getcwd(), 'data', 'dataset', 'MNIST_1')
#SAVE_DIR = os.getcwd()
b = os.path.exists(SAVE_DIR)
tns = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
ds = torchvision.datasets.MNIST(root='data/dataset', train=True, download=True)
print(len(ds))
for i in range(len(ds)):
    item = ds[i]
    im = item[0]
    sub_dir = str(item[1])
    path = os.path.join(SAVE_DIR, sub_dir, str(i)+'.bmp')
    im.save(path)

