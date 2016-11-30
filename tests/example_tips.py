# import ipdb
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from clusteror.core import Clusteror
from clusteror.plot import scatter_plot_two_dim_group_data
from clusteror.plot import group_occurance_plot

clusteror_valley = Clusteror.from_csv('data/tips.csv')
# float number columns
dat_float = clusteror_valley.raw_data.select_dtypes(include=[np.float])
dat_float_median = dat_float.median()
# store median
dat_float_median.to_csv('float_median.csv')
# float number subtracted by median and taken tanh
clusteror_valley.cleaned_data = np.tanh(dat_float - dat_float_median)
# Now work on ordinal and categorical columns
# Caveat: need to retrain a model when a new value appears in these
# columns if you want them be a clustering feature

# ordinal columns can be processed similar to float number columns
# after being numericalised. It's already in numbers in this example
dat_ord = clusteror_valley.raw_data.loc[:, ['size']]
dat_ord_median = dat_ord.median()
# store median
dat_ord_median.to_csv('ord_median.csv')
# append to cleaned data
clusteror_valley.cleaned_data = pd.concat(
    [clusteror_valley.cleaned_data, np.tanh(dat_ord - dat_ord_median)],
    axis=1
)
# create dummy values for categorical columns
dat_cat = pd.get_dummies(
    clusteror_valley.raw_data.loc[:, ['sex', 'smoker', 'day', 'time']],
    drop_first=True
)
# make it equally away from 0
dat_cat = dat_cat - 0.5
# add to cleaned data
clusteror_valley.cleaned_data = pd.concat(
    [clusteror_valley.cleaned_data, dat_cat],
    axis=1
)
clusteror_valley.field_importance = {'total_bill': 100, 'tip': 1000}
# train neural networks
clusteror_valley.train_sda_dim_reducer(
    hidden_layers_sizes=[20],
    corruption_levels=[0.1],
    field_importance=clusteror_valley.field_importance,
    min_epochs=70,
    improvement_threshold=0.9,
    verbose=True
)
clusteror_valley.save_dim_reducer(filename='sda.pk')
clusteror_valley.reduce_to_one_dim()
clusteror_valley.train_valley(bins=20, contrast=0.5)
clusteror_valley.save_valley(filename='valley.json')
clusteror_valley.add_cluster()

scatter_plot_two_dim_group_data(
    clusteror_valley.raw_data.iloc[:, :2],
    clusteror_valley.raw_data.cluster,
    show=False,
    filename='two.png'
)
group_occurance_plot(
    clusteror_valley.raw_data.sex,
    'Sex',
    clusteror_valley.raw_data.cluster,
    'Segment',
    bbox_to_anchor=(0.5, 1),
    show=False,
    filename='sex_dist.png',
    rot=0
)
