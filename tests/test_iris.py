import sys
sys.path.append('..')
import numpy as np
from clusteror.core import Clusteror

clusteror_valley = Clusteror.from_csv('data/iris.csv')
clusteror_valley.cleaned_data = np.tanh(
    clusteror_valley.raw_data.iloc[:, :-1]
    - clusteror_valley.raw_data.iloc[:, :-1].median(axis=0)
)
clusteror_valley.train_sda_dim_reducer(
    hidden_layers_sizes=[20],
    corruption_levels=[0.1],
    min_epochs=70,
    improvement_threshold=0.9,
    verbose=True
)
clusteror_valley.save_dim_reducer(filename='sda.pk')
clusteror_valley.reduce_to_one_dim()
clusteror_valley.train_valley()
clusteror_valley.save_valley(filename='valley.json')
clusteror_valley.add_cluster()

clusteror_kmeans = Clusteror.from_csv('data/iris.csv')
clusteror_kmeans.one_dim_data = clusteror_kmeans.one_dim_data
clusteror_kmeans.train_kmains(10)
clusteror_kmeans.save_kmeans(filename='km.pk')
clusteror_kmeans.add_cluster()
