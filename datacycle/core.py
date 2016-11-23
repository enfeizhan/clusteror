import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from .dA import dA
from .settings import numpy_random_seed
from .settings import theano_random_seed


class OutRangeError(Exception):
    pass


class Clusteror(object):
    def __init__(self, raw_data):
        self._raw_data = raw_data

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

    def _pretraining_early_stopping(
            self,
            train_model,
            n_train_batches,
            n_epochs,
            patience,
            patience_increase,
            improvement_threshold,
            verbose,
            ):
        epoch = 0
        done_looping = False
        check_frequency = min(n_epochs, patience / 2)
        best_cost = np.inf

        start_time = timeit.default_timer()

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            # go through training set
            c = []
            for minibatch_index in range(n_train_batches):
                c.append(train_model(minibatch_index))
            cost = np.mean(c)
            if verbose is True:
                print(
                    'Training epoch {epoch}, cost {cost}.'
                    .format(epoch=epoch, cost=cost))
            if epoch % check_frequency == 0:
                if cost < best_cost:
                    much_better_cost = best_cost * improvement_threshold
                    if cost < much_better_cost:
                        patience = max(patience,  (epoch + 1) * patience_increase)
                        print(
                            'Epoch {epoch}, patience increased to {patience}'.
                            format(epoch=epoch, patience=patience))
                    best_cost = cost
            if epoch > patience:
                done_looping = True
        end_time = timeit.default_timer()
        training_time = (end_time - start_time)
        sys.stderr.write(
            'The 30% corruption code for file ' +
            os.path.split(__file__)[1] +
            ' ran for {time:.2f}m'.format(time=training_time / 60.))

    def reduce_dim(
        self,
        approach='da'
    ):
        if approach == 'da':
            self._da_reduce_dim()
        elif approach == 'sda':
            self._sda_reduce_dim()

    def _da_reduce_dim(
            self,
            field_weights=None,
            to_dim=1,
            batch_size=50,
            corruption_level=0.3,
            learning_rate=0.002,
            training_epochs=200,
            verbose=False,
            patience=60,
            patience_increase=2,
            improvement_threshold=0.9995,
            random_state=123):
        '''
        Reduces the dimension of each record down to a dimension.
        verbose: boolean, default True
          If true, printing out the progress of pretraining.
        '''
        assert self.cleaned_dat is not None, 'Need cleaned data'
        if (self.cleaned_dat.max() > 1).any():
            raise OutRangeError('Maximum should be less equal than 1.')
        if (self.cleaned_dat.min() < -1).any():
            raise OutRangeError('Minimum should be greater equal than -1')
        dat = np.asarray(self.cleaned_dat, dtype=theano.config.floatX)

        np_rs = np.random.RandomState(numpy_random_seed)
        theano_rs = RandomStreams(np_rs.randint(theano_random_seed))
        train_set_x = theano.shared(value=dat, borrow=True)

        # compute number of minibatches for training
        n_train_batches = (
            dat.shape[0] // batch_size + int(dat.shape[0] % batch_size > 0)
        )

        # allocate symbolic variables for the data
        # index to a [mini]batch
        index = T.lscalar('index')
        x = T.matrix('x')
        #################################
        # BUILDING THE MODEL CORRUPTION #
        #################################
        da = dA(
            n_visible=dat.shape[1],
            n_hidden=1,
            np_rs=np_rs,
            theano_rs=theano_rs,
            field_weights=field_weights,
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
        _pretraining_early_stopping(
            train_model=train_da,
            n_train_batches=n_train_batches,
            n_epochs=training_epochs,
            patience=patience,
            patience_increase=patience_increase,
            improvement_threshold=improvement_threshold,
            verbose=verbose)
        self.denoising_autoencoder = da
        x = T.dmatrix('x')
        self.to_lower_dim = theano.function([x], da.get_hidden_values(x))
        self.reconstruct = theano.function([x], da.get_reconstructed_input(x))
