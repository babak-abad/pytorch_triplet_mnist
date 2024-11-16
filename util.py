import glob
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

class Categ():
    def __init__(self):
        self._name = ''


class MNIST_DS(torch.utils.data.Dataset):
    def __init__(self, dir):
        self._ds_dir = dir
        self._samples = [] # {class_name, [sample_1, sample_2, ...]}

        dirs = next(os.walk(dir))[1]

        os.listdir()

        pattern = os.path.join(dir, '*', '*.bmp')
        pathes = glob.glob(pattern)

        p = pathes[0]
        p = os.path.relpath(p, dir)

        for p in pathes:
            parts = p.split(os.path.sep)
            fn = parts[-1]
            parrent = parts[-2]
            sample = (parrent, fn)
            self._samples.append(sample)


    def _list2dic(self, ls):
        dic = {}
        for p in ls:
            parts = p.split(os.path.sep)
            fn = parts[-1]
            parrent = parts[-2]
            if not (parrent in dic):
                dic[parrent] = []
            dic[parrent].append(fn)

        return dic

    def __len__(self):
        return len(self._samples)

    def _get_class(self, abs_path_to_file: str) -> str:
        return abs_path_to_file.split(os.path.sep)[-1]

    def _select_random_file(self, path_to_dir: str) -> str:
        return ''
    def _convert_idx_to_path(self, idx):
        '''
        converts an int number to dir path of a class
        :param idx:
        :return:
        '''
        return os.path.join(self._ds_dir, str(idx))

    def __getitem__(self, idx):
        query_class = self._samples[idx][0]
        query_path = os.path.join(self._ds_dir, self._samples[idx][0], self._samples[idx][1])

        # choose a positive sample from
        dirs = os.listdir(os.path.join(self._ds_dir, self._samples[idx][0]))
        positive_path = os.path.join(self._ds_dir, self._samples[idx][0], dirs[random.randint(0, len(dirs)-1)])

        # we choose random dir index until it is unequal to query
        # (otherwise we have chosen and object of same class as negative sample)
        r = random.randint(0, len(self._samples)-1)
        negative_class = self._samples[r]
        while negative_class == query_class:
            r = random.randint(0, self.__len__() - 1)
            negative_class = self._samples[r]

        negative_path = os.path.join(self._ds_dir, self._samples[r][0], self._samples[r][1])
        # print(os.path.exists(query_path))
        # cv2.imshow('a', cv2.resize(cv2.imread(negative_path, cv2.IMREAD_GRAYSCALE), (100, 100)))
        # cv2.moveWindow('a', 100, 100)
        # cv2.waitKey()
        q = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE).astype('double')
        p = cv2.imread(positive_path, cv2.IMREAD_GRAYSCALE).astype('double')
        n = cv2.imread(negative_path, cv2.IMREAD_GRAYSCALE).astype('double')



        q = torch.from_numpy(q).unsqueeze(0)
        p = torch.from_numpy(p).unsqueeze(0)
        n = torch.from_numpy(n).unsqueeze(0)

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


def train_triplet(
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
        for _, (q, p, n) in enumerate(train_dataloader):
            # opt.zero_grad()
            # pred = mdl(inp)
            # loss = criterion(pred, out)
            # loss.backward()
            # opt.step()
            # train_loss += loss.item()

            opt.zero_grad()
            q = q.float()
            p = p.float()
            n = n.float()
            pred_q = mdl(q)
            pred_p = mdl(p)
            pred_n = mdl(n)

            loss = criterion(pred_q, pred_p, pred_n)
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
            for i, (q, p, n) in enumerate(valid_dataloader):
                q = q.float()
                p = p.float()
                n = n.float()

                pred_q = mdl(q)
                pred_p = mdl(p)
                pred_n = mdl(n)
                ls = criterion(pred_q, pred_p, pred_n)
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


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

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
