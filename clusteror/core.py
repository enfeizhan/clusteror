import os
import sys
import json
import timeit
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
from .utils import check_local_extremity


class OutRangeError(Exception):
    pass


class Clusteror(object):
    def __init__(self, raw_data):
        self._raw_data = raw_data
        self.np_rs = np.random.RandomState(numpy_random_seed)
        self.theano_rs = RandomStreams(self.np_rs.randint(theano_random_seed))

    @classmethod
    def from_csv(cls, filepath, **kwargs):
        raw_data = pd.read_csv(filepath, **kwargs)
        return cls(raw_data)

    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, raw_data):
        self._raw_data = raw_data

    @property
    def cleaned_data(self):
        return self._cleaned_data

    @cleaned_data.setter
    def cleaned_data(self, cleaned_data):
        self._cleaned_data = cleaned_data

    @property
    def one_dim_data(self):
        return self._one_dim_data

    @one_dim_data.setter
    def one_dim_data(self, one_dim_data):
        self._one_dim_data = one_dim_data

    def train_dim_reducer(
        self,
        approach='da',
        field_importance=None,
        to_dim=1,
        batch_size=50,
        hidden_layers_sizes=None,
        corruption_levels=0.3,
        learning_rate=0.002,
        min_epochs=200,
        patience=60,
        patience_increase=2,
        improvement_threshold=0.98,
        verbose=False,
    ):
        '''
        Use various methods to reduce the dimension for further analysis.
        Early stops if updates change less than a threshold.
        '''
        assert self._cleaned_data is not None, 'Need cleaned data'
        if (self._cleaned_data.max() > 1).any():
            raise OutRangeError('Maximum should be less equal than 1.')
        if (self._cleaned_data.min() < -1).any():
            raise OutRangeError('Minimum should be greater equal than -1')
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
        if approach == 'da':
            self._train_da_dim_reducer(
                field_importance=field_importance,
                to_dim=to_dim,
                batch_size=batch_size,
                corruption_level=corruption_levels,
                learning_rate=learning_rate,
                min_epochs=min_epochs,
                patience=patience,
                patience_increase=patience_increase,
                improvement_threshold=improvement_threshold,
                verbose=verbose,
            )
        elif approach == 'sda':
            assert hidden_layers_sizes is not None
            assert isinstance(corruption_levels, list)
            assert len(hidden_layers_sizes) == len(corruption_levels)
            self._train_sda_dim_reducer(
                field_importance=field_importance,
                batch_size=batch_size,
                hidden_layers_sizes=hidden_layers_sizes,
                corruption_levels=corruption_levels,
                learning_rate=learning_rate,
                min_epochs=min_epochs,
                patience=patience,
                patience_increase=patience_increase,
                improvement_threshold=improvement_threshold,
                verbose=verbose,
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

    def _train_da_dim_reducer(
            self,
            field_importance,
            to_dim,
            batch_size,
            corruption_level,
            learning_rate,
            min_epochs,
            patience,
            patience_increase,
            improvement_threshold,
            verbose,
            ):
        '''
        Reduces the dimension of each record down to a dimension.
        verbose: boolean, default True
          If true, printing out the progress of pretraining.
        '''
        # allocate symbolic variables for the dat
        # index to a [mini]batch
        index = T.lscalar('index')
        x = T.matrix('x')
        da = dA(
            n_visible=self.data.shape[1],
            n_hidden=to_dim,
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
        self.da_lower_dim = function([x], da.get_hidden_values(x))
        self.da_reconstruct = function(
            [x],
            da.get_reconstructed_input(da.get_hidden_values(x))
        )

    def _train_sda_dim_reducer(
            self,
            field_importance,
            batch_size,
            hidden_layers_sizes,
            corruption_levels,
            learning_rate,
            min_epochs,
            patience,
            patience_increase,
            improvement_threshold,
            verbose,
            ):
        '''
        Reduce the dimension of each record down to a dimension.
        '''
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
        self.sda_lower_dim = function([x], sda.get_final_hidden_layer(x))
        self.sda_reconstruct = function(
            [x],
            sda.get_first_reconstructed_input(sda.get_final_hidden_layer(x))
        )

    def save_dim_reducer(self, approach='da', filename='dim_reducer.pk'):
        with open(approach+'_'+filename, 'wb') as f:
            if approach == 'da':
                pk.dump(self.da_to_lower_dim, f)
            elif approach == 'sda':
                pk.dump(self.sda_to_lower_dim, f)

    def load_dim_reducer(self, approach='da', filename='dim_reducer.pk'):
        with open(filename, 'rb') as f:
            if approach == 'da':
                self.da_to_lower_dim = pk.load(f)
            elif approach == 'sda':
                self.sda_to_lower_dim = pk.load(f)

    def train_tagger(self, bins=100, contrast=0.3):
        bins = np.linspace(0, 1, bins+1)
        left_points = bins[:-1]
        cuts = pd.cut(self._one_dim_data, bins=bins)
        bin_counts = cuts.reset_index().loc[:, 'counts']
        local_min_inds = []
        for ind, value in bin_counts.iteritems():
            is_local_min = check_local_extremity(
                bin_counts,
                ind,
                contrast=contrast,
                kind='min'
            )
            if is_local_min:
                local_min_inds.append(left_points[ind])
        self.trained_bins = [0] + local_min_inds + [1]

        def tagger(one_dim_data):
            cuts = pd.cut(
                one_dim_data,
                bins=self.trained_bins,
                labels=list(range(len(self.trained_bins) - 1))
            )
            return cuts.get_values()
        self.tagger = tagger

    def save_tagger(self, filename):
        with open(filename, 'wb') as f:
            json.dump(self.trained_bins, f)

    def load_tagger(self, filename):
        with open(filename, 'rb') as f:
            self.trained_bins = json.load(f)

        def tagger(one_dim_data):
            cuts = pd.cut(
                one_dim_data,
                bins=self.trained_bins,
                labels=list(range(len(self.trained_bins) - 1))
            )
            return cuts.get_values()
        self.tagger = tagger

    def get_one_dim_data(self, approach='da'):
        assert self._cleaned_data is not None
        if approach == 'da':
            self._one_dim_data = self.da_to_lower_dim(self._cleaned_data)
        elif approach == 'sda':
            self._one_dim_data = self.sda_to_lower_dim(self._cleaned_data)

    def train_kmeans(self, n_clusters=None, **kwargs):
        self.km = KMeans(n_clusters=n_clusters, **kwargs)
        self.km.fit(self._one_dim_data)

    def save_kmeans(self, filename):
        with open(filename, 'wb') as f:
            pk.dump(self.km, f)

    def load_kmeans(self, filename):
        with open(filename, 'rb') as f:
            self.km = pk.load(f)

    def add_cluster_with_tagger(self):
        self.raw_data.loc[:, 'cluster'] = self.tagger(self._one_dim_data)

    def add_cluster_with_kmeans(self):
        self.raw_data.loc[:, 'cluster'] = self.km.predict(self._one_dim_data)
