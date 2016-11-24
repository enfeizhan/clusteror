import sys
sys.path.append('..')
from datacycle.core import Clusteror

clusteror = Clusteror.from_csv('train.csv', nrows=10000)
clusteror.cleaned_dat = (clusteror.raw_dat.iloc[:, 1:] - 128) / 256
clusteror.reduce_dim(verbose=True)
