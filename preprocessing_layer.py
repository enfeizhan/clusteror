import os
import sys
import timeit
import theano
import theano.tensor as T
import numpy as np
import pandas as pd
import pickle as pk
from scipy.special import expit
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.preprocessing import StandardScaler
from discovery_layer import DataDiscovery
from dA import dA
# from SdA import SdA

ss = StandardScaler()


def _pretraining_early_stopping(
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


class DataPreprocessing(DataDiscovery):
    def da_reduce_dim(
            self,
            dat,
            field_weights=None,
            to_dim=1,
            batch_size=50,
            corruption_level=0.3,
            learning_rate=0.002,
            training_epochs=200,
            verbose=True,
            patience=60,
            patience_increase=2,
            improvement_threshold=0.9995,
            random_state=123):
        '''
        Reduce the dimension of each record down to a dimension.
        verbose: boolean, default True
        If true, printing out the progress of pretraining.
        '''
        rng = np.random.RandomState(random_state)
        if isinstance(dat, pd.core.frame.DataFrame):
            dat = dat.values
        sys.stderr.write(
            'Squeezing the data into [0 1] recommended!')
        dat = np.asarray(dat, dtype=theano.config.floatX)
        train_set_x = theano.shared(value=dat, borrow=True)

        # compute number of minibatches for training, validation and testing
        n_train_batches = (
            train_set_x.get_value(borrow=True).shape[0] // batch_size)

        # start-snippet-2
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')
        # end-snippet-2

        #################################
        # BUILDING THE MODEL CORRUPTION #
        #################################
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input_dat=x,
            field_weights=field_weights,
            n_visible=dat.shape[1],
            n_hidden=1
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

    def save_da_reduce_dim(self, filename):
        f = open(filename, 'wb')
        pk.dump(self.to_lower_dim, f)
        print(
            'Denoising autoencoder model saved in {}.'.format(filename)
            )

    def sda_reduce_dim(
            self,
            dat,
            to_dim=1,
            batch_size=50,
            corruption_level=0.3,
            learning_rate=0.002,
            training_epochs=100,
            compress_to_zero_one=True,
            random_state=123):
        '''
        Reduce the dimension of each record down to a dimension.
        '''
        rng = np.random.RandomState(random_state)
        if isinstance(dat, pd.core.frame.DataFrame):
            dat = dat.values
        if compress_to_zero_one:
            print('Transform data ...')
            dat = expit(ss.fit_transform(dat))
            print('Transform completed ...')
            self.standard_scaler = ss
        else:
            sys.stderr.write('Better squeeze data into [0 1] range!')
        dat = np.asarray(dat, dtype=theano.config.floatX)
        train_set_x = theano.shared(value=dat, borrow=True)

        # compute number of minibatches for training, validation and testing
        n_train_batches = (
            train_set_x.get_value(borrow=True).shape[0] // batch_size)

        # start-snippet-2
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')
        # end-snippet-2

        #####################################
        # BUILDING THE MODEL CORRUPTION 30% #
        #####################################
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=dat.shape[1],
            n_hidden=1
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
        start_time = timeit.default_timer()
        # ###########
        # TRAINING #
        # ###########
        # go through training epochs
        for epoch in range(training_epochs):
            # go through trainng set
            c = []
            for batch_index in range(n_train_batches):
                c.append(train_da(batch_index))
            print('Training epoch {}, cost '.format(epoch) +
                  '{cost}.'.format(cost=np.mean(c)))
        end_time = timeit.default_timer()
        training_time = (end_time - start_time)
        sys.stderr.write(
            'The 30% corruption code for file ' +
            os.path.split(__file__)[1] +
            ' ran for {time:.2f}m'.format(time=training_time / 60.))
        self.denoising_autoencoder = da
        x = T.dmatrix('x')
        self.to_lower_dim = theano.function([x], da.get_hidden_values(x))
        self.reconstruct = theano.function([x], da.get_reconstructed_input(x))

# running this model will test the method defined here
if __name__ == '__main__':
    dat = pd.read_csv('complete_cols_for_clustering.csv', index_col=0)
    cols_to_read = [
        'bookings',
        'passengers',
        'luggage_fee',
        'infant_fee',
        'unique_passengers']
    dp = DataPreprocessing()
    dp.da_reduce_dim(
        dat.loc[:, cols_to_read],
        to_dim=1,
        batch_size=100000,
        corruption_level=0.3,
        learning_rate=0.02,
        )
