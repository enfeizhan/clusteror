import sys
sys.path.append('..')
from datacycle.core import Clusteror

clusteror = Clusteror.from_csv('train.csv', nrows=10000)
clusteror.cleaned_dat = (clusteror.raw_dat.iloc[:, 1:] - 128) / 256
# clusteror.reduce_dim(min_epochs=10, verbose=True)
clusteror.train_dim_reducer(
    approach='sda',
    hidden_layers_sizes=[20],
    corruption_levels=[0.1],
    min_epochs=70,
    verbose=True
)
