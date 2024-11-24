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

def tensor_to_im(t):
    im = t[0][0].numpy()
    im = im.astype('uint8')
    return im

for k in range(10):
    for i, (q, p, n) in enumerate(dl):
        cv2.imshow('q', tensor_to_im(q))
        cv2.moveWindow('q', 100, 100)
 
        cv2.imshow('p', tensor_to_im(p))
        cv2.moveWindow('p', 200, 200)

        cv2.imshow('n', tensor_to_im(n))
        cv2.moveWindow('n', 300, 300)
        cv2.waitKey()

    ds.end()
