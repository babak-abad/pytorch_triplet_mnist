import glob
import os
import sys
import util

from tqdm import tqdm
from time import sleep
for i in tqdm(range(0, 100), desc='desc', colour='GREEN'):
    for j in tqdm(range(0, 100), desc='desc', colour='GREEN'):
        sleep(0.1)


ds_path = os.path.join(os.getcwd(), 'data', 'dataset', 'MNIST_1')
# ls = os.listdir(ds_path)
# ls = [x[0] for x in os.walk(ds_path)]
# ls = next(os.walk(ds_path))[1]
print(len ('baba'))
#print(ls)
ds = util.MNIST_DS(ds_path)
ds.__getitem__(5)
#print(files)



