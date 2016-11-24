import os
import sys
import timeit
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano import function
from theano.tensor.shared_randomstreams import RandomStreams
from .dA import dA
from .settings import numpy_random_seed
from .settings import theano_random_seed


class OutRangeError(Exception):
    pass


class Clusteror(object):
    def __init__(self, raw_dat):
        self._raw_dat = raw_dat

    @classmethod
    def from_csv(cls, filepath, **kwargs):
        raw_dat = pd.read_csv(filepath, **kwargs)
        return cls(raw_dat)

    @property
    def raw_dat(self):
        return self._raw_dat

    @raw_dat.setter
    def raw_dat(self, raw_dat):
        self._raw_dat = raw_dat

    @property
    def cleaned_dat(self):
        return self._cleaned_dat

    @cleaned_dat.setter
    def cleaned_dat(self, cleaned_dat):
        self._cleaned_dat = cleaned_dat

    def reduce_dim(
        self,
        approach='da',
        field_importance=None,
        to_dim=1,
        batch_size=50,
        corruption_level=0.3,
        learning_rate=0.002,
        min_epochs=200,
        verbose=False,
        patience=60,
        patience_increase=2,
        improvement_threshold=0.9995,
    ):
        '''
        Use various methods to reduce the dimension for further analysis.
        Early stops if updates change less than a threshold.
        '''
        if approach == 'da':
            self._da_reduce_dim(
                field_importance=field_importance,
                to_dim=to_dim,
                batch_size=batch_size,
                corruption_level=corruption_level,
                learning_rate=learning_rate,
                min_epochs=min_epochs,
                verbose=verbose,
                patience=patience,
                patience_increase=patience_increase,
                improvement_threshold=improvement_threshold,
            )
        # elif approach == 'sda':
        #     self._sda_reduce_dim()

    def _pretraining_early_stopping(
            self,
            train_model,
            n_train_batches,
            min_epochs,
            patience,
            patience_increase,
            improvement_threshold,
            corruption_level,
            verbose,
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
                c.append(train_model(minibatch_index))
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
                'The {:2.1f}%'.format(corruption_level*100) +
                ' corruption code for file ' +
                os.path.split(__file__)[1] +
                ' ran for {time:.2f}m\n'.format(time=training_time / 60.))

    def _da_reduce_dim(
            self,
            field_importance,
            to_dim,
            batch_size,
            corruption_level,
            learning_rate,
            min_epochs,
            verbose,
            patience,
            patience_increase,
            improvement_threshold,
            ):
        '''
        Reduces the dimension of each record down to a dimension.
        verbose: boolean, default True
          If true, printing out the progress of pretraining.
        '''
        assert self.cleaned_dat is not None, 'Need cleaned dat'
        if (self.cleaned_dat.max() > 1).any():
            raise OutRangeError('Maximum should be less equal than 1.')
        if (self.cleaned_dat.min() < -1).any():
            raise OutRangeError('Minimum should be greater equal than -1')
        dat = np.asarray(self.cleaned_dat, dtype=theano.config.floatX)

        np_rs = np.random.RandomState(numpy_random_seed)
        theano_rs = RandomStreams(np_rs.randint(theano_random_seed))
        train_set_x = theano.shared(value=dat, borrow=True)

        # compute number of minibatches for training
        # needs one more batch if residual is non-zero
        # e.g. 5 rows with batch size 2 needs 5 // 2 + 1
        n_train_batches = (
            dat.shape[0] // batch_size + int(dat.shape[0] % batch_size > 0)
        )

        # allocate symbolic variables for the dat
        # index to a [mini]batch
        index = T.lscalar('index')
        x = T.matrix('x')
        #################################
        # BUILDING THE MODEL CORRUPTION #
        #################################
        da = dA(
            n_visible=dat.shape[1],
            n_hidden=to_dim,
            np_rs=np_rs,
            theano_rs=theano_rs,
            field_importance=field_importance,
            input_dat=x,
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
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )
        self._pretraining_early_stopping(
            train_model=train_da,
            n_train_batches=n_train_batches,
            min_epochs=min_epochs,
            patience=patience,
            patience_increase=patience_increase,
            improvement_threshold=improvement_threshold,
            corruption_level=corruption_level,
            verbose=verbose,
        )
        self.denoising_autoencoder = da
        self.to_lower_dim = function([x], da.get_hidden_values(x))
        self.reconstruct = function(
            [x],
            da.get_reconstructed_input(da.get_hidden_values(x))
        )
