import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.nn import Module
from torch import nn
from torch.optim import SGD
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

# providing dataset
class Csv_Loader(Dataset):
    def __init__(self, data_frame):
        self.x = data_frame.values[:, :-1]
        self.y = data_frame.values[:, -1]
        self.x = self.x.astype('float32')
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        print('x')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

    def get_splits(self, train_sz):
        train_sz = round(train_sz * len(self.x))
        test_sz = len(self.x) - train_sz

        trn, tst = random_split(self, [train_sz, test_sz])

        trn = DataLoader(trn, batch_size=32, shuffle=True)
        tst = DataLoader(tst, batch_size=32, shuffle=False)

        return trn, tst

# making model
class Slp(Module):
    def __init__(self, n_input):
        super(Slp, self).__init__()
        self.h1 = nn.Linear(n_input, 1)
        self.a1 = nn.Sigmoid()


    # propagate forward
    def forward(self, x):
        x = self.h1(x)
        x = self.a1(x)
        return x


# training
def train(model, train_data, n_epoch):
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss = nn.BCELoss()

    for i in range(0, n_epoch):
        for j, (inp, out) in enumerate(train_data):
            opt.zero_grad()
            pred = model(inp)
            ls = loss(pred, out)
            ls.backward()
            opt.step()

# evaluation
def eval(model, test_ds):
    predicteds = []
    actuals = []

    for i, (inputs, targets) in enumerate(test_ds):
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))

        pred = model(inputs)\
            .detach()\
            .numpy()\
            .round()

        predicteds.append(pred)
        actuals.append(actual)

    predictions = np.vstack(predicteds)
    actuals = np.vstack(actuals)

    acc = accuracy_score(actuals, predictions)
    return acc

#main
path = 'sonar.all-data'
train_sz = 0.5

# providing dataset
df = pd.read_csv(path, header=None)
n_input = len(df.columns) - 1
ds = Csv_Loader(df)
trn, tst = ds.get_splits(train_sz)

# making model
md = Slp(n_input)
print('model: ')
print(md)
print("#"*50)

# training
print('training...')
train(model=md, train_data=trn, n_epoch=100)
print("#"*50)

# evaluation
acc = eval(model=md, test_ds=tst)
print('acc = ' + str(acc))