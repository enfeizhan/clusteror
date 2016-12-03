'''
This module contains ``Clusteror`` class capsulating raw data to discover
clusters from, the cleaned data for a clusteror to run on.

The clustering model encompasses two parts:

1. Neural network:
   Pre-training (often encountered in Deep Learning context)
   is implemented to achieve a goal that the neural network maps the input
   data of higher dimension to a one dimensional representation. Ideally this
   mapping is one-to-one.
   A Denoising Autoencoder (DA) or Stacked Denoising Autoencoder (SDA) is
   implemented for this purpose.
2. One dimensional clustering model:
   A separate model segments the samples against the one dimensional
   representation. Two models are available in this class definition:
       * K-Means
       * Valley model

The pivot idea here is given the neural network is a good one-to-one mapper
the separate clustering model on one dimensional representation is equivalent
to a clustering model on the original high dimensional data.

Note
----
Valley model is explained in details in module ``clusteror.utils``.
'''
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
from .nn import dA
from .nn import SdA
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
    ``Clusteror`` class can train neural networks *DA* or
    *SDA*, train taggers, or load saved models
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
        **da** for *DA*; **sda** for *SDA*.
        Facilating functions called with one or the other algorithm.
    _da_dim_reducer: Theano function
        Keeps the Theano function that is from trained DA model. Reduces
        the dimension of the cleaned data down to one.
    _sda_dim_reducer: Theano function
        Keeps the Theano function that is from trained SDA model. Reduces
        the dimension of the cleaned data down to one.
    _one_dim_data: Numpy Array
        The dimension reduced one dimensional data.
    _valley: Python function
        Trained valley model tagging sample with their one dimensional
        representation.
    _kmeans: Scikit-Learn K-Means model
        Trained K-Means model tagging samples with their one dimensional
        representation.
    _tagger: str
        Keeps records of which tagger implemented.
    _field_importance: List
        Keeps the list of coefficiences that influence the clustering
        emphasis.
    '''
    def __init__(self, raw_data):
        self._raw_data = raw_data.copy()

    @classmethod
    def from_csv(cls, filepath, **kwargs):
        '''
        Class method for directly reading CSV file.

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
            autoencoder as ``DA``.
        '''
        return self._da_dim_reducer

    @da_dim_reducer.setter
    def da_dim_reducer(self, da_dim_reducer):
        self._da_dim_reducer = da_dim_reducer
        self._network = 'da'

    @property
    def sda_dim_reducer(self):
        '''
        Theano function: Function that reduces dataset dimension. Attribute
            ``_network`` is given **sda** to designate the method of the
            autoencoder as ``SDA``.
        '''
        return self._sda_dim_reducer

    @sda_dim_reducer.setter
    def sda_dim_reducer(self, sda_dim_reducer):
        self._sda_dim_reducer = sda_dim_reducer
        self._network = 'sda'

    @property
    def one_dim_data(self):
        '''
        Numpy Array: Stores the output of neural network that has dimension
        one.
        '''
        return self._one_dim_data

    @one_dim_data.setter
    def one_dim_data(self, one_dim_data):
        self._one_dim_data = one_dim_data

    @property
    def valley(self):
        '''
        Python function: Trained on the dimension reduced one dimensional
        data that segregates subjects into concentration of existence in a
        subset of ``[-1, 1]``, by locating the "valley" in the distribution
        landscape. ``_tagger`` is given **valley** to facilitate
        follow-up usages.
        '''
        return self._valley

    @valley.setter
    def valley(self, valley):
        self._valley = valley
        self._tagger = 'valley'

    @property
    def kmeans(self):
        '''
        Python function: Trained on the dimension reduced one dimensional
        data that segregates subjects into concentration of existence in a
        subset of ``[-1, 1]`` with K-Means algorithm.  ``_tagger`` is
        given **valley** to facilitate follow-up usages.
        '''
        return self._kmeans

    @kmeans.setter
    def kmeans(self, kmeans):
        self._kmeans = kmeans
        self._tagger = 'kmeans'

    @property
    def tagger(self):
        '''
        str: Name the tagger if necessary to do so, which will facilitate, e.g.
        prefixing the filepath.
        '''
        return self._tagger

    @tagger.setter
    def tagger(self, tagger):
        self._tagger = tagger

    @property
    def field_importance(self):
        '''
        List: Significance that given to fields when training of neural
        network is done. Fields with a large number will be given more
        attention.

        Note
        ----
        The importance is only meaningful relatively between fields. If no
        values are specified, all fields are treated equally.

        Parameters
        ----------
        field_importance : List or Dict, default None (List of Ones)
            * If a list is designated, all fields should be assigned an
            importance, viz, the length of the list should be equal to the
            length of the features training the neural network.

            * It can also be given in a dict. In such a case, the fields can
            be selectively given a value. Dict key is for field name and value
            is for the importance. Fields not included will be initiated with
            the default value one. A warning will be issued when a key is
            not on the list of field names, mostly because of a typo.
        '''
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
        Checks on cleaned data before any work is done. This list of checks
        can be extended when more checks should be included.
        '''
        cleaned_data_info = (
            'Need first assign your cleaned data to attribute "_cleaned_data"'
        )
        assert self._cleaned_data is not None, cleaned_data_info
        if (self._cleaned_data.max() > 1).any():
            raise OutRangeError('Maximum should be less equal than 1.')
        if (self._cleaned_data.min() < -1).any():
            raise OutRangeError('Minimum should be greater equal than -1')

    def _check_network(self):
        '''
        Check if network has been correctly setup.
        '''
        network_info = (
            'Clusteror needs to know which network to use in'
            'attribute "_network"'
        )
        assert self._network is not None, network_info
        info = 'Train {} with {} or load it first!'
        if self._network == 'da':
            info = info.format('DA', '"train_da_dim_reducer"')
            assert self._da_dim_reducer is not None, info
        elif self._network == 'sda':
            info = info.format('SDA', '"train_sda_dim_reducer"')
            assert self._sda_dim_reducer is not None, info

    def _prepare_network_training(self, batch_size):
        '''
        Preparations needed to kick off training neural networks.

        Parameters
        ----------
        batch_size: int
            Size of each training batch. Necessary to derive the number
            of batches.
        '''
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
            train_func,
            n_train_batches,
            min_epochs,
            patience,
            patience_increase,
            improvement_threshold,
            verbose,
            **kwargs
            ):
        '''
        Scheme of early stopping if no substantial improvement can be
        observed.

        Parameters
        ----------
        train_func: Theano Function
            Function that takes in training set and updates internal
            parameters, in this case the weights and biases in neural network,
            and returns the evaluation of the cost function after each
            training step.
        n_train_batches: int
            Number of training batches derived from the total number of
            training samples and the batch size.
        min_epochs: int
            The mininum number of training epoch to run. It can be exceeded
            depending on the setup of patience and ad-hoc training progress.
        patience: int
            True number of training epochs to run if larger than
            ``min_epochs``. Note it is potentially increased during the
            training if the cost is better than the expectation from
            current cost.
        patience_increase: int
            Coefficient used to increase patience against epochs that
            have been run.
        improvement_threshold: float, between 0 and 1
            Minimum improvement considered as substantial improvement, i.e.
            new cost over existing lowest cost lower than this value.
        verbose: boolean
            Prints out training at each epoch if true.
        **kwargs: keyword arguments
            All keyword arguments pass on to ``train_func``.
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
                c.append(train_func(minibatch_index, **kwargs))
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
        Trains a ``DA`` neural network.

        Parameters
        ----------
        field_importance : List or Dict, default None (List of Ones)
            * If a list is designated, all fields should be assigned an
            importance, viz, the length of the list should be equal to the
            length of the features training the neural network.

            * It can also be given in a dict. In such a case, the fields can
            be selectively given a value. Dict key is for field name and value
            is for the importance. Fields not included will be initiated with
            the default value one. A warning will be issued when a key is
            not on the list of field names, mostly because of a typo.
        batch_size: int
            Size of each training batch. Necessary to derive the number
            of batches.
        corruption_level: float, between 0 and 1
            Dropout rate in reading input, typical pratice in deep learning
            to avoid overfitting.
        learning_rate: float
            Propagating step size for gredient descent algorithm.
        min_epochs: int
            The mininum number of training epoch to run. It can be exceeded
            depending on the setup of patience and ad-hoc training progress.
        patience: int
            True number of training epochs to run if larger than
            ``min_epochs``. Note it is potentially increased during the
            training if the cost is better than the expectation from
            current cost.
        patience_increase: int
            Coefficient used to increase patience against epochs that
            have been run.
        improvement_threshold: float, between 0 and 1
            Minimum improvement considered as substantial improvement, i.e.
            new cost over existing lowest cost lower than this value.
        verbose: boolean, default False
            Prints out training at each epoch if true.
        '''
        self._network = 'da'
        # note .field_importance indicates the magic of the property
        # decorator is played to transform the format the input
        self.field_importance = field_importance
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
            train_func=train_da,
            n_train_batches=self.n_train_batches,
            min_epochs=min_epochs,
            patience=patience,
            patience_increase=patience_increase,
            improvement_threshold=improvement_threshold,
            verbose=verbose
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
        Trains a ``SDA`` neural network.

        Parameters
        ----------
        field_importance : List or Dict, default None (List of Ones)
            * If a list is designated, all fields should be assigned an
            importance, viz, the length of the list should be equal to the
            length of the features training the neural network.

            * It can also be given in a dict. In such a case, the fields can
            be selectively given a value. Dict key is for field name and value
            is for the importance. Fields not included will be initiated with
            the default value one. A warning will be issued when a key is
            not on the list of field names, mostly because of a typo.
        batch_size: int
            Size of each training batch. Necessary to derive the number
            of batches.
        hidden_layers_sizes: List of ints
            Number of neurons in the hidden layers (all but the input layer).
        corruption_levels: List of floats, between 0 and 1
            Dropout rate in reading input, typical pratice in deep learning
            to avoid overfitting.
        learning_rate: float
            Propagating step size for gredient descent algorithm.
        min_epochs: int
            The mininum number of training epoch to run. It can be exceeded
            depending on the setup of patience and ad-hoc training progress.
        patience: int
            True number of training epochs to run if larger than
            ``min_epochs``. Note it is potentially increased during the
            training if the cost is better than the expectation from
            current cost.
        patience_increase: int
            Coefficient used to increase patience against epochs that
            have been run.
        improvement_threshold: float, between 0 and 1
            Minimum improvement considered as substantial improvement, i.e.
            new cost over existing lowest cost lower than this value.
        verbose: boolean, default False
            Prints out training at each epoch if true.
        '''
        # note .field_importance indicates the magic of the property
        # decorator is played to transform the format the input
        self.field_importance = field_importance
        assert hidden_layers_sizes is not None
        assert isinstance(corruption_levels, list)
        assert len(hidden_layers_sizes) == len(corruption_levels)
        self._network = 'sda'
        self._check_cleaned_data()
        self._prepare_network_training(batch_size=batch_size)
        # for the purpose of this excercise, restrict the final layer 1d
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
                train_func=pretraining_fns[ind],
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

    def _prefix_filepath(self, prefix_type, filepath):
        '''
        Prefixes a filepath with the type stored in the file.

        Examples
        --------
            >> clusteror._prefix_filepath('network', 'a/b')
            'a/da_b'

        Note
        ----
        Only the filename part is prefixed if there are directories in the
        path.

        Parameters
        ----------
        prefix_type: str
            The type to prefixing the filepath.
        filepath: str
            Filepath to be prefixed.

        Returns
        -------
            Prefixed filepath.
        '''
        filepath_list = list(os.path.split(filepath))
        filepath_list[-1] = (
            getattr(self, prefix_type) +
            '_' +
            filepath_list[-1]
        )
        filepath = os.path.join(tuple(filepath_list))
        return filepath

    def save_dim_reducer(
        self,
        filepath='dim_reducer.pk',
        include_network=False
    ):
        '''
        Save dimension reducer from the neural network training.

        Parameters
        ----------
        filepath: str
            Filename to store the dimension reducer.
        include_network: boolean
            If true, prefix the filepath with the network type.
        '''
        self._check_network()
        if include_network:
            filepath = self._prefix_filepath('network', filepath)
        with open(filepath, 'wb') as f:
            if self._network == 'da':
                pk.dump(self._da_dim_reducer, f)
            elif self._network == 'sda':
                pk.dump(self._sda_dim_reducer, f)

    def load_dim_reducer(self, filepath='dim_reducer.pk'):
        '''
        Loads saved dimension reducer. Need to first name the network type.

        Parameters
        ----------
        filepath: str
        '''
        assert self._network is not None
        with open(filepath, 'rb') as f:
            if self._network == 'da':
                self._da_dim_reducer = pk.load(f)
            elif self._network == 'sda':
                self._sda_dim_reducer = pk.load(f)

    def reduce_to_one_dim(self):
        '''
        Reduces the dimension of input dataset to one before the tagging
        in the next step.

        Input of the Theano function is the cleaned data and output is a
        one dimensional data stored in ``_one_dim_data``.
        '''
        self._check_cleaned_data()
        self._check_network()
        if self._network == 'da':
            self._one_dim_data = self._da_dim_reducer(self._cleaned_data)
        elif self._network == 'sda':
            self._one_dim_data = self._sda_dim_reducer(self._cleaned_data)
        self._one_dim_data = self._one_dim_data[:, 0]

    def _check_one_dim_data(self):
        '''
        Check if one_dim_data exists. Give error info if not.
        '''
        one_dim_data_info = 'Get reduced one dimensional data first!'
        assert self._one_dim_data is not None, one_dim_data_info

    def train_valley(self, bins=100, contrast=0.3):
        '''
        Trains the ability to cut the universe of samples into clusters based
        how the dimension reduced dataset assembles in a histogram. Unlike
        the K-Means, no need to preset the number of clusters.

        Parameters
        ----------
        bins: int
            Number of bins to aggregate the one dimensional data.
        contrast: float, between 0 and 1
            Threshold used to define local minima and local maxima. Detailed
            explanation in ``utils.find_local_extremes``.

        Note
        ----
        When getting only one cluster, check the distribution of
        ``one_dim_data``. Likely the data points flock too close to each other.
        Try increasing ``bins`` first. If not working, try different
        neural networks with more or less layers with more or less neurons.
        '''
        bins = np.linspace(-1, 1, bins+1)
        # use the left point of bins to name the bin
        left_points = np.asarray(bins[:-1])
        self._check_one_dim_data()
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

    def _check_tagger(self):
        '''
        Check tagger existence. Give error info if not.
        '''
        tagger_info = 'Clusteror needs to know which tagger to use'
        assert self._tagger is not None, tagger_info
        info = 'Train {} with {} or load it first' 
        if self._tagger == 'valley':
            info = info.format('"valley"', '"train_valley"')
            assert self._valley is not None, info
        elif self._tagger == 'kmeans':
            info = info.format('"kmeans"', '"train_kmeans"')
            assert self._kmeans is not None, info

    def save_valley(self, filepath, include_taggername=False):
        '''
        Saves valley tagger.

        Parameters
        ----------
        filepath: str
            File path to save the tagger.
        include_taggername: boolean, default False
            Include the **valley_** prefix in filename if true.
        '''
        self.check_tagger()
        if include_taggername:
            filepath = self._prefix_filepath('tagger', filepath)
        with open(filepath, 'w') as f:
            json.dump(self.trained_bins, f)

    def load_valley(self, filepath):
        '''
        Loads a saved valley tagger from a file. Create the valley function
        from the saved parameters.

        Parameter
        ---------
        filepath: str
            File path to the file saving the valley tagger.
        '''
        with open(filepath, 'r') as f:
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

    def train_kmeans(self, n_clusters=10, **kwargs):
        '''
        Trains K-Means model on top of the one dimensional data derived from
        dimension reducers.

        Parameters
        ----------
        n_clusters: int
            The number of clusters required to start a K-Means learning.
        **kwargs: keyword arguments
            Any other keyword arguments passed on to Scikit-Learn K-Means
            model.
        '''
        self._check_one_dim_data()
        self._kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        self._kmeans.fit(self._one_dim_data.reshape(-1, 1))
        self._tagger = 'kmeans'

    def save_kmeans(self, filepath, include_taggername=False):
        '''
        Saves K-Means model to the named file path. Can add a prefix to
        indicate this saves a K-Means model.

        Parameters
        ----------
        filepath: str
           File path for saving the model.
        include_taggername: boolean, default False
           Include the **kmean_** prefix in filename if true.
        '''
        self._check_tagger()
        if include_taggername:
            filepath = self._prefix_filepath('tagger', filepath)
        with open(filepath, 'wb') as f:
            pk.dump(self._kmeans, f)

    def load_kmeans(self, filepath):
        '''
        Loads a saved K-Means tagger from a file.

        Parameter
        ---------
        filepath: str
            File path to the file saving the K-Means tagger.
        '''
        with open(filepath, 'rb') as f:
            self._kmeans = pk.load(f)
        self._tagger = 'kmeans'

    def add_cluster(self):
        '''
        Tags each sample regarding their reduced one dimensional value. Adds
        an extra column **'cluster'** to ``raw_data``, seggesting a
        zero-based cluster ID.
        '''
        self._check_tagger()
        if self._tagger == 'valley':
            self.raw_data.loc[:, 'cluster'] = self._valley(self._one_dim_data)
        elif self._tagger == 'kmeans':
            self.raw_data.loc[:, 'cluster'] = (
                self._kmeans.predict(self._one_dim_data.reshape(-1, 1))
            )
