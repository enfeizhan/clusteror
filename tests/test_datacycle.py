# import ipdb
# import os
import sys
sys.path.append('..')
# import timeit
import unittest
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano import function
from datacycle.dA import dA
from datacycle.SdA import SdA
from datacycle.settings import decimal_places


def test_SdA(
        dat,
        pretrain_epochs=15,
        pretrain_lr=0.01,
        hidden_layers_sizes=[20],
        corruption_levels=[.3],
        batch_size=2,
        ):
    dat = np.asarray(dat, dtype=theano.config.floatX)
    train_set = theano.shared(value=dat, borrow=True)
    # compute number of minibatches for training, validation and testing
    n_rows = dat.shape[0]
    # there should be one more batch if residual is greater than 0
    n_train_batches = n_rows // batch_size + int((n_rows % batch_size) > 0)
    # numpy random generator
    # construct the stacked denoising autoencoder class
    sda = SdA(
        n_ins=dat.shape[1],
        hidden_layers_sizes=hidden_layers_sizes,
        corruption_levels=corruption_levels,
    )
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(
        train_set=train_set,
        batch_size=batch_size
    )
    print('... pre-training the model')
    # Pre-train layer-wise
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretrain_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                assert corruption_levels[i] >= 0
                assert corruption_levels[i] < 1
                c.append(
                    pretraining_fns[i](
                        index=batch_index,
                        corruption_level=corruption_levels[i],
                        learning_rate=pretrain_lr)
                )
            print(
                'Pre-training layer {}, '.format(i) +
                'epoch {epoch}, '.format(epoch=epoch) +
                'cost {cost:f}'.format(cost=np.mean(c))
            )
    return sda


def tanh_cross_entropy(field_importance, dat_in, dat_rec):
    cost = -np.sum(
        field_importance * (
            0.5 * (1 + dat_in) * np.log(0.5 * (1 + dat_rec)) +
            0.5 * (1 - dat_in) * np.log(0.5 * (1 - dat_rec))
        ),
        axis=1
    )
    return np.mean(cost)


class TestDA(unittest.TestCase):
    def setUp(self):
        # prepare testing data
        self.dat = pd.read_csv(
            'makeup_test_data.csv',
            dtype=theano.config.floatX
        )
        self.field_importance = [1, 5, 10]
        self.initial_W = np.asarray(
            [[1], [2], [3]],
            dtype=theano.config.floatX
        )
        self.initial_bvis = np.asarray([1, 2, 3], dtype=theano.config.floatX)
        self.initial_bhid = np.asarray([1], dtype=theano.config.floatX)
        self.corruption_level = 0
        self.learning_rate = 0.1
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.da = dA(
            n_visible=self.dat.shape[1],
            n_hidden=1,
            field_importance=self.field_importance,
            initial_W=self.initial_W,
            initial_bvis=self.initial_bvis,
            initial_bhid=self.initial_bhid,
            input_dat=self.x
        )
        # calculate cost in a sequential way
        self.y = np.tanh(np.dot(self.dat, self.initial_W) + self.initial_bhid)
        self.z = np.tanh(np.dot(self.y, self.initial_W.T) + self.initial_bvis)
        self.seq_cost = tanh_cross_entropy(
            np.asarray(self.field_importance, dtype=theano.config.floatX),
            self.dat,
            self.z
        )

    def test_dA_conrrupted_input(self):
        corrupted_input = self.da.get_corrupted_input(
            self.x,
            self.corruption_level
        )
        get_da_corrupted_input = function([self.x], corrupted_input)
        da_corrupted_input = get_da_corrupted_input(self.dat)
        test_almost_equal = np.testing.assert_array_almost_equal(
            self.dat,
            da_corrupted_input,
            decimal=decimal_places
        )
        self.assertTrue(test_almost_equal is None)

    def test_dA_hidden_values(self):
        hidden_values = self.da.get_hidden_values(self.x)
        get_da_hidden_values = function([self.x], hidden_values)
        da_hidden_values = get_da_hidden_values(self.dat)
        test_almost_equal = np.testing.assert_array_almost_equal(
            self.y,
            da_hidden_values,
            decimal=decimal_places
        )
        self.assertTrue(test_almost_equal is None)

    def test_dA_reconstructed_input(self):
        reconstructed_input = self.da.get_reconstructed_input(
            self.da.get_hidden_values(self.x)
        )
        get_da_reconstructed_input = function([self.x], reconstructed_input)
        da_reconstructed_input = get_da_reconstructed_input(self.dat)
        test_almost_equal = np.testing.assert_array_almost_equal(
            self.z,
            da_reconstructed_input,
            decimal=decimal_places
        )
        self.assertTrue(test_almost_equal is None)

    def test_dA_cost(self):
        # calculate cost from dA
        cost, updates = self.da.get_cost_updates(
            corruption_level=self.corruption_level,
            learning_rate=self.learning_rate
        )
        train_da = function(
            [self.x],
            cost,
            updates=updates,
        )
        da_cost = np.mean(train_da(self.dat))
        # confirm equal
        self.assertAlmostEqual(da_cost, self.seq_cost, places=decimal_places)


if __name__ == '__main__':
    unittest.main()
    # test_SdA(dat)
