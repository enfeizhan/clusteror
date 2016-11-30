'''
This module contains ``Clusteror`` class capsulating raw data to discover
clusters from, the cleaned data for a clusteror to run on, as well as
methods to training neural networks and delimiting occurances into clusters.
'''
# `h`_
# Example
# -------
# Examples can be given using either the ``Example`` or ``Examples``
# sections. Sections support any reStructuredText formatting, including
# literal blocks::
# 
#         $ python example_numpy.py
# 
# 
# Section breaks are created with two blank lines. Section breaks are also
# implicitly created anytime a new section starts. Section bodies *may* be
# indented:
# 
# Notes
# -----
#     This is an example of an indented section. It's like any other section,
#         but the body is indented to help it stand out from surrounding text.
# 
# If a section is indented, then a section break is created by
# resuming unindented text.
# 
# Attributes
# ----------
# module_level_variable1 : int
#     Module level variables may be documented in either the ``Attributes``
#     section of the module docstring, or in an inline docstring immediately
#     following the variable.
# 
#     Either form is acceptable, but the two should not be mixed. Choose
#     one convention to document module level variables and be consistent
#     with it.
# 
# .. _NumPy Documentation HOWTO:
#     https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
# import ipdb
import os
import sys
import json
import timeit
import warnings
import numpy as np
import pandas as pd
import pickle as pk
import theano
import theano.tensor as T
from sklearn.cluster import KMeans
from theano import function
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from .dA import dA
from .SdA import SdA
from .settings import numpy_random_seed
from .settings import theano_random_seed
from .utils import find_local_extremes


class OutRangeError(Exception):
    '''
    Exceptions thrown as cleaned data go beyond range ``[-1, 1]``.
    '''
    pass


class Clusteror(object):
    '''
    ``Clusteror`` class can train neural networks *denoising autoencoder* or
    *Stached Denoising Autoencoder*, train taggers, or load saved models
    from files.

    Parameters
    ----------
    raw_data : Pandas DataFrame
        Dataframe read from data source. It can be original dataset without
        any preprocessing or with a certain level of manipulation for
        future analysis.

    Attributes
    ----------
    _raw_data : Pandas DataFrame
        Stores the original dataset. It's the dataset that later
        post-clustering performance analysis will be based on.
    _cleaned_data : Pandas DataFrame
        Preprocessed data. Not necessarily has same number of columns with
        ``_raw_data`` as a categorical column can derive multiple columns.
        As the ``tanh`` function is used as activation function for symmetric
        consideration. All columns should have values in range ``[-1, 1]``,
        otherwise an ``OutRangeError`` will be raised.
    _network : str
        **da** for *Denoising Autoencoder*; **sda** for *Stacked Denoising
        Autoencoder*. Facilating functions called with one or the other
        algorithm.
    '''
    def __init__(self, raw_data):
        self._raw_data = raw_data

    @classmethod
    def from_csv(cls, filepath, **kwargs):
        '''
        Class method for directly reading .csv file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file
        **kwargs : keyword arguments
            Other keyword arguments passed to ``pandas.read_csv``
        '''
        raw_data = pd.read_csv(filepath, **kwargs)
        return cls(raw_data)

    @property
    def raw_data(self):
        '''
        Pandas DataFrame: For assgining new values to ``_raw_data``.
        '''
        return self._raw_data

    @raw_data.setter
    def raw_data(self, raw_data):
        self._raw_data = raw_data

    @property
    def cleaned_data(self):
        '''
        Pandas DataFrame: For assgining cleaned dataframe to ``_cleaned_dat``.
        '''
        return self._cleaned_data

    @cleaned_data.setter
    def cleaned_data(self, cleaned_data):
        self._cleaned_data = cleaned_data

    @property
    def da_dim_reducer(self):
        '''
        Theano function: Function that reduces dataset dimension. Attribute
            ``_network`` is given **da** to designate the method of the
            autoencoder as ``Denoising Autocoder``.
        '''
        return self._da_dim_reducer

    @da_dim_reducer.setter
    def da_dim_reducer(self, da_dim_reducer):
        self._da_dim_reducer = da_dim_reducer
        self._network = 'da'

    @property
    def sda_dim_reducer(self):
        return self._sda_dim_reducer

    @sda_dim_reducer.setter
    def sda_dim_reducer(self, sda_dim_reducer):
        self._sda_dim_reducer = sda_dim_reducer
        self._network = 'sda'

    @property
    def one_dim_data(self):
        return self._one_dim_data

    @one_dim_data.setter
    def one_dim_data(self, one_dim_data):
        self._one_dim_data = one_dim_data

    @property
    def valley(self):
        return self._valley

    @valley.setter
    def valley(self, valley):
        self._valley = valley
        self._tagger = 'valley'

    @property
    def kmeans(self):
        return self._kmeans

    @kmeans.setter
    def kmeans(self, kmeans):
        self._kmeans = kmeans
        self._tagger = 'kmeans'

    @property
    def field_importance(self):
        return self._field_importance

    @field_importance.setter
    def field_importance(self, field_importance):
        n_fields = self._cleaned_data.shape[1]
        if isinstance(field_importance, list):
            assert len(field_importance) == n_fields
            self._field_importance = field_importance
        elif isinstance(field_importance, dict):
            self._field_importance = [1] * n_fields
            columns = self._cleaned_data.columns.tolist()
            for field, importance in field_importance.items():
                try:
                    index = columns.index(field)
                    self._field_importance[index] = importance
                except ValueError:
                    msg = '{} isn\'t in fields'.format(field)
                    warnings.warn(msg)

    def _check_cleaned_data(self):
        '''
        Use various methods to reduce the dimension for further analysis.
        Early stops if updates change less than a threshold.
        '''
        assert self._cleaned_data is not None, 'Need cleaned data'
        if (self._cleaned_data.max() > 1).any():
            raise OutRangeError('Maximum should be less equal than 1.')
        if (self._cleaned_data.min() < -1).any():
            raise OutRangeError('Minimum should be greater equal than -1')

    def _prepare_network_training(self, batch_size):
        self.np_rs = np.random.RandomState(numpy_random_seed)
        self.theano_rs = RandomStreams(self.np_rs.randint(theano_random_seed))
        # compute number of minibatches for training, validation and testing
        self.data = np.asarray(self._cleaned_data, dtype=theano.config.floatX)
        self.train_set = shared(value=self.data, borrow=True)
        # compute number of minibatches for training
        # needs one more batch if residual is non-zero
        # e.g. 5 rows with batch size 2 needs 5 // 2 + 1
        self.n_train_batches = (
            self.data.shape[0] // batch_size +
            int(self.data.shape[0] % batch_size > 0)
        )

    def _pretraining_early_stopping(
            self,
            train_fun,
            n_train_batches,
            min_epochs,
            patience,
            patience_increase,
            improvement_threshold,
            verbose,
            **kwargs
            ):
        '''
        min_epochs is the minimum iterations that need to run.
        patience is possible to go beyond min_epochs.
        Must run max(min_epochs, patience).
        '''
        n_epochs = 0
        done_looping = False
        check_frequency = min(min_epochs, patience // 3)
        best_cost = np.inf
        assert improvement_threshold > 0 and improvement_threshold < 1
        start_time = timeit.default_timer()
        while (n_epochs < min_epochs) or (not done_looping):
            n_epochs += 1
            # go through training set
            c = []
            for minibatch_index in range(n_train_batches):
                c.append(train_fun(minibatch_index, **kwargs))
            cost = np.mean(c)
            if verbose:
                print(
                    'Training epoch {n_epochs}, '.format(n_epochs=n_epochs) +
                    'cost {cost}.'.format(cost=cost)
                )
            if n_epochs % check_frequency == 0:
                # check cost every check_frequency
                if cost < best_cost:
                    benchmark_better_cost = best_cost * improvement_threshold
                    if cost < benchmark_better_cost:
                        # increase patience if cost improves a lot
                        # the increase is a multiplicity of epochs that
                        # have been run
                        patience = max(patience,  n_epochs * patience_increase)
                        if verbose:
                            print(
                                'Epoch {n_epochs},'.format(n_epochs=n_epochs) +
                                ' patience increased to {patience}'.format(
                                    patience=patience
                                )
                            )
                    best_cost = cost
            if n_epochs > patience:
                done_looping = True
        end_time = timeit.default_timer()
        if verbose:
            training_time = (end_time - start_time)
            sys.stderr.write(
                os.path.split(__file__)[1] +
                ' ran for {time:.2f}m\n'.format(time=training_time / 60.))

    def train_da_dim_reducer(
        self,
        field_importance=None,
        batch_size=50,
        corruption_level=0.3,
        learning_rate=0.002,
        min_epochs=200,
        patience=60,
        patience_increase=2,
        improvement_threshold=0.98,
        verbose=False,
    ):
        '''
        Reduces the dimension of each record down to a dimension.
        verbose: boolean, default True
          If true, printing out the progress of pretraining.
        '''
        self._network = 'da'
        self._check_cleaned_data()
        self._prepare_network_training(batch_size=batch_size)
        # allocate symbolic variables for the dat
        # index to a [mini]batch
        index = T.lscalar('index')
        x = T.matrix('x')
        da = dA(
            n_visible=self.data.shape[1],
            n_hidden=1,
            np_rs=self.np_rs,
            theano_rs=self.theano_rs,
            field_importance=field_importance,
            input_data=x,
        )
        cost, updates = da.get_cost_updates(
            corruption_level=corruption_level,
            learning_rate=learning_rate
        )
        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: self.train_set[index * batch_size: (index + 1) * batch_size]
            }
        )
        self._pretraining_early_stopping(
            train_fun=train_da,
            n_train_batches=self.n_train_batches,
            min_epochs=min_epochs,
            patience=patience,
            patience_increase=patience_increase,
            improvement_threshold=improvement_threshold,
            corruption_level=corruption_level,
            verbose=verbose,
        )
        self.da = da
        self._da_dim_reducer = function([x], da.get_hidden_values(x))
        self.da_reconstruct = function(
            [x],
            da.get_reconstructed_input(da.get_hidden_values(x))
        )

    def train_sda_dim_reducer(
        self,
        field_importance=None,
        batch_size=50,
        hidden_layers_sizes=[20],
        corruption_levels=[0.3],
        learning_rate=0.002,
        min_epochs=200,
        patience=60,
        patience_increase=2,
        improvement_threshold=0.98,
        verbose=False
    ):
        '''
        Reduce the dimension of each record down to a dimension.
        '''
        assert hidden_layers_sizes is not None
        assert isinstance(corruption_levels, list)
        assert len(hidden_layers_sizes) == len(corruption_levels)
        self._network = 'sda'
        self._check_cleaned_data()
        self._prepare_network_training(batch_size=batch_size)
        hidden_layers_sizes.append(1)
        corruption_levels.append(0)
        x = T.matrix('x')
        sda = SdA(
            n_ins=self.data.shape[1],
            hidden_layers_sizes=hidden_layers_sizes,
            np_rs=self.np_rs,
            theano_rs=self.theano_rs,
            field_importance=field_importance,
            input_data=x
        )
        pretraining_fns = sda.pretraining_functions(
            train_set=self.train_set,
            batch_size=batch_size
        )
        for ind in range(sda.n_layers):
            self._pretraining_early_stopping(
                train_fun=pretraining_fns[ind],
                n_train_batches=self.n_train_batches,
                min_epochs=min_epochs,
                patience=patience,
                patience_increase=patience_increase,
                improvement_threshold=improvement_threshold,
                verbose=verbose,
                corruption_level=corruption_levels[ind],
                learning_rate=learning_rate
            )
        self.sda = sda
        self._sda_dim_reducer = function([x], sda.get_final_hidden_layer(x))
        self.sda_reconstruct = function(
            [x],
            sda.get_first_reconstructed_input(sda.get_final_hidden_layer(x))
        )

    def save_dim_reducer(
        self,
        filename='dim_reducer.pk',
        include_network=False
    ):
        if include_network:
            filename = self._network + '_' + filename
        with open(filename, 'wb') as f:
            if self._network == 'da':
                pk.dump(self._da_dim_reducer, f)
            elif self._network == 'sda':
                pk.dump(self._sda_dim_reducer, f)

    def load_dim_reducer(self, filename='dim_reducer.pk'):
        with open(filename, 'rb') as f:
            if self._network == 'da':
                self._da_to_lower_dim = pk.load(f)
            elif self._network == 'sda':
                self._sda_to_lower_dim = pk.load(f)

    def reduce_to_one_dim(self):
        assert self._cleaned_data is not None
        if self._network == 'da':
            self._one_dim_data = self._da_dim_reducer(self._cleaned_data)
        elif self._network == 'sda':
            self._one_dim_data = self._sda_dim_reducer(self._cleaned_data)
        self._one_dim_data = self._one_dim_data[:, 0]

    def train_valley(self, bins=100, contrast=0.3):
        bins = np.linspace(-1, 1, bins+1)
        # use the left point of bins to name the bin
        left_points = np.asarray(bins[:-1])
        cuts = pd.cut(self._one_dim_data, bins=bins)
        # ipdb.set_trace()
        bin_counts = cuts.describe().reset_index().loc[:, 'counts']
        local_min_inds, local_mins, local_max_inds, local_maxs = (
            find_local_extremes(bin_counts, contrast)
        )
        self.trained_bins = left_points[local_min_inds].tolist() + [1]
        if self.trained_bins[0] != -1:
            self.trained_bins = [-1] + self.trained_bins

        def valley(one_dim_data):
            cuts = pd.cut(
                one_dim_data,
                bins=self.trained_bins,
                labels=list(range(len(self.trained_bins) - 1))
            )
            return cuts.get_values()
        self._valley = valley
        self._tagger = 'valley'

    def save_valley(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.trained_bins, f)

    def load_valley(self, filename):
        with open(filename, 'r') as f:
            self.trained_bins = json.load(f)

        def valley(one_dim_data):
            cuts = pd.cut(
                one_dim_data,
                bins=self.trained_bins,
                labels=list(range(len(self.trained_bins) - 1))
            )
            return cuts.get_values()
        self._valley = valley
        self._tagger = 'valley'

    def train_kmeans(self, n_clusters=None, **kwargs):
        self._kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        self._kmeans.fit(self._one_dim_data.reshape(-1, 1))
        self._tagger = 'kmeans'

    def save_kmeans(self, filename):
        with open(filename, 'wb') as f:
            pk.dump(self._kmeans, f)

    def load_kmeans(self, filename):
        with open(filename, 'rb') as f:
            self._kmeans = pk.load(f)
        self._tagger = 'valley'

    def add_cluster(self):
        if self._tagger == 'valley':
            self.raw_data.loc[:, 'cluster'] = self._valley(self._one_dim_data)
        elif self._tagger == 'kmeans':
            self.raw_data.loc[:, 'cluster'] = (
                self._kmeans.predict(self._one_dim_data.reshape(-1, 1))
            )
