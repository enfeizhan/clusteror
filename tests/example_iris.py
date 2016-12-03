import sys
sys.path.append('..')
import numpy as np
from clusteror.core import Clusteror
from clusteror.plot import scatter_plot_two_dim_group_data
from clusteror.plot import hist_plot_one_dim_group_data

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
clusteror_valley.save_dim_reducer(filepath='sda.pk')
clusteror_valley.reduce_to_one_dim()
clusteror_valley.train_valley(bins=20, contrast=0.5)
clusteror_valley.save_valley(filepath='valley.json')
clusteror_valley.add_cluster()

clusteror_kmeans = Clusteror.from_csv('data/iris.csv')
clusteror_kmeans.one_dim_data = clusteror_valley.one_dim_data
clusteror_kmeans.train_kmeans(10)
clusteror_kmeans.save_kmeans(filepath='km.pk')
clusteror_kmeans.add_cluster()

scatter_plot_two_dim_group_data(
    clusteror_valley.raw_data.iloc[:, :2],
    clusteror_valley.raw_data.cluster,
    colors=['red', 'blue', 'black'],
    show=False,
    filepath='two.png'
)
hist_plot_one_dim_group_data(
    clusteror_valley.one_dim_data,
    clusteror_valley.raw_data.Name,
    bins=50,
    colors=['red', 'blue', 'black'],
    show=False,
    filepath='name.png'
)
hist_plot_one_dim_group_data(
    clusteror_valley.one_dim_data,
    clusteror_valley.raw_data.cluster,
    bins=50,
    colors=['red', 'blue', 'black'],
    show=False,
    filepath='cluster.png'
)
