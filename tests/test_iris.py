import sys
sys.path.append('..')
import numpy as np
from clusteror.core import Clusteror

clusteror_tagger = Clusteror.from_csv('data/iris.csv')
clusteror.cleaned_data = np.tanh(
    clusteror.raw_data.iloc[:, :-1]
    - clusteror.raw_data.iloc[:, :-1].median(axis=0)
)
# clusteror.reduce_dim(min_epochs=10, verbose=True)
approach = 'sda'
clusteror_tagger.train_dim_reducer(
    approach=approach,
    hidden_layers_sizes=[20],
    corruption_levels=[0.1],
    min_epochs=70,
    improvement_threshold=0.9,
    verbose=True
)
clusteror_tagger.save_dim_reducer(approach=approach, filename='sda.pk')
clusteror_tagger.get_one_dim_data(approach=approach)
clusteror_tagger.train_tagger()
clusteror_tagger.save_tagger(filename='tagger.json')
clusteror_tagger.add_cluster_with_tagger()

clusteror_kmeans = Clusteror.from_csv('data/iris.csv')
clusteror_kmeans.one_dim_data = clusteror_kmeans.one_dim_data
clusteror_kmeans.train_kmains(10)
clsuteror_kmeans.save_kmeans(filename='km.pk')
clusteror_kmeans.add_cluster_with_kmeans()
