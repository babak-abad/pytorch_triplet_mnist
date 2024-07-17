import os
import random

import cv2
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import numpy as np
from torch import nn
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt
import torchvision


class MNIST_DS(torch.utils.data.Dataset):
    def __init__(self, dir):
        self._dir = dir
        self._classes = os.listdir(dir)

    def __len__(self):
        return len(self._classes)

    def _convert_idx_to_path(self, idx):
        '''
        converts an int number to dir path of a class
        :param idx:
        :return:
        '''
        return os.path.join(self._dir, str(idx))

    def __getitem__(self, idx):
        dir1 = self._convert_idx_to_path(idx)
        sub_dirs1 = os.listdir(dir1)

        r = random.randint(0, len(sub_dirs1) - 1)
        query = os.path.join(dir1, sub_dirs1[r])

        r = random.randint(0, len(sub_dirs1) - 1)
        pos = os.path.join(dir1, sub_dirs1[r])

        r = random.randint(0, self.__len__() - 1)
        # we choose random dir index until it is unequal to query
        # (otherwise we have chosen and object of same class as negative sample)
        while r == idx:
            r = random.randint(0, self.__len__() - 1)

        dir2 = self._convert_idx_to_path(r)
        sub_dirs2 = os.listdir(dir2)
        r = random.randint(0, len(sub_dirs2) - 1)
        neg = os.path.join(dir2, sub_dirs2[r])

        q = cv2.imread(query)
        p = cv2.imread(pos)
        n = cv2.imread(neg)

        return q, p, n


def train(
        mdl,
        train_dataloader,
        valid_dataloader,
        n_epoch,
        opt,
        criterion,
        valid_step,
        save_path):
    min_valid_loss = float('inf')

    best_state = 0

    # save training and validation losses to draw them later
    trn_losses = []
    vld_losses = []

    train_loss = 0.0
    valid_loss = 0.0

    for e in range(0, n_epoch):
        mdl.train()
        train_loss = 0.0
        for _, (inp, out) in enumerate(train_dataloader):
            opt.zero_grad()
            pred = mdl(inp)
            loss = criterion(pred, out)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        trn_losses.append(train_loss)

        # if it is not time to calculate validation, continue
        if e % valid_step != 0 and e != 0:
            vld_losses.append(valid_loss)
            continue

        mdl.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, (inp, out) in enumerate(valid_dataloader):
                pred = mdl(inp)
                ls = criterion(pred, out)
                valid_loss += ls.item()

        vld_losses.append(valid_loss)

        # if they are better weights save them (less loss = better weights)
        if valid_loss < min_valid_loss:
            best_state = mdl.state_dict()
            if save_path != '':
                p = save_path.format(
                    en=n_epoch,
                    vl=valid_loss,
                    tl=train_loss)
                torch.save(mdl, p)

        print('epoch: ' + str(e + 1))
        print('train loss: ' + str(train_loss / len(train_dataloader)))
        print('valid loss: ' + str(valid_loss / len(valid_dataloader)))
        print('\n')

    return trn_losses, vld_losses, best_state


# if idx = 0 then it prints x
# if idx = 1 then it prints y
def draw_dataloader(dt, idx):
    res = []
    for v in dt.dataset:
        print(v[0])
        print(v[1])
        res.extend(v[idx].tolist())
    res = np.array(res)
    plt.plot(res)
    plt.show()


def normalize(data, rng=(0, 1)):
    mms = MinMaxScaler(feature_range=rng)
    scaler = mms.fit(data)
    return scaler.transform(data)


#
# def create_dataloader(data, win_sz):
#     x = []
#     y = []
#
#     for i in range(0, len(data) - win_sz - 1):
#         xx = data[i:win_sz + i, 1].astype('float32')
#         yy = data[win_sz + i, 1].astype('float32')
#         x.append(xx)
#         y.append(yy)
#
#     x = np.array(x)
#     y = np.array(y).reshape(-1, 1)
#     return x, y


class Data_Provider:
    def __init__(self, data, win_sz):
        self.x = []
        self.y = []

        dt = data.reshape(-1, 1)

        self.mms = MinMaxScaler(feature_range=(0, 1))
        s = self.mms.fit(dt)
        norm_data = s.transform(dt)
        norm_data = norm_data.reshape(1, -1)[0]
        for i in range(0, len(norm_data) - win_sz - 1):
            xx = norm_data[i:win_sz + i].astype('float32')
            yy = norm_data[win_sz + i].astype('float32')
            self.x.append(xx)
            self.y.append(yy)

        self.x = np.array(self.x)
        self.y = np.array(self.y).reshape(-1, 1)

    def denormalize(self, x):
        t = x.reshape(-1, 1)
        t = self.mms.inverse_transform(t)
        t = t.reshape(1, -1)[0]
        return t

    def get_dataloader(self, batch_sz):
        return create_dataloader(
            self.x,
            self.y,
            batch_sz,
            True)

    # mms = MinMaxScaler(feature_range=(0, 1))
    # scaler =  mms.fit(trn)
    # normalized_trn = scaler.transform(trn)
    #
    # scaler = mms.fit(vld)
    # normalized_vld = scaler.transform(vld)
    #
    # scaler = mms.fit(tst)
    # normalized_tst = scaler.transform(tst)
