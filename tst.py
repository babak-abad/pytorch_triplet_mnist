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
ds_path = os.path.join(os.getcwd(), 'data', 'dataset', 'MNIST_1')
ds = utl.MNIST_DS(ds_path)
dl = DataLoader(dataset=ds, batch_size=1, shuffle=True)

for k in range(10):
    for i, (q, p, n) in enumerate(dl):

        im = q[0].numpy()
        cv2.imshow('q', q[0].numpy())
        cv2.moveWindow('q', 100, 100)

        cv2.imshow('p', p[0].numpy())
        cv2.moveWindow('p', 200, 200)

        cv2.imshow('n', n[0].numpy())
        cv2.moveWindow('n', 300, 300)

    ds.end()
