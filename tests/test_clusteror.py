import sys
sys.path.append('..')
import unittest
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano import shared
from theano import function
from clusteror.nn import dA
from clusteror.nn import SdA
from clusteror.settings import decimal_places


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
            'tests/data/makeup_test_data.csv',
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
            input_data=self.x
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


class TestSdA(unittest.TestCase):
    def setUp(self):
        # prepare testing data
        self.dat = pd.read_csv(
            'tests/data/makeup_test_data.csv',
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
        self.sda = SdA(
            n_ins=self.dat.shape[1],
            hidden_layers_sizes=[1],
            field_importance=self.field_importance,
            input_data=self.x
        )
        # monkey patch the weights and biases
        self.sda.dA_layers[0].W.set_value(self.initial_W)
        self.sda.dA_layers[0].bhid.set_value(self.initial_bhid)
        self.sda.dA_layers[0].bhid_prime.set_value(self.initial_bvis)
        # calculate cost in a sequential way
        self.y = np.tanh(np.dot(self.dat, self.initial_W) + self.initial_bhid)
        self.z = np.tanh(np.dot(self.y, self.initial_W.T) + self.initial_bvis)
        self.seq_cost = tanh_cross_entropy(
            np.asarray(self.field_importance, dtype=theano.config.floatX),
            self.dat,
            self.z
        )

    def test_SdA_final_hidden_layer(self):
        final_hidden_layer = self.sda.get_final_hidden_layer(self.x)
        get_sda_final_hidden_layer = function(
            [self.x],
            final_hidden_layer
        )
        sda_final_hidden_layer = get_sda_final_hidden_layer(self.dat)
        test_almost_equal = np.testing.assert_array_almost_equal(
            self.y,
            sda_final_hidden_layer,
            decimal=decimal_places
        )
        self.assertTrue(test_almost_equal is None)

    def test_SdA_first_reconstructed_layer(self):
        first_reconstructed_input = self.sda.get_first_reconstructed_input(
            self.sda.get_final_hidden_layer(self.x)
        )
        get_sda_first_reconstructed_input = function(
            [self.x],
            first_reconstructed_input
        )
        sda_first_reconstructed_input = get_sda_first_reconstructed_input(
            self.dat
        )
        test_almost_equal = np.testing.assert_array_almost_equal(
            self.z,
            sda_first_reconstructed_input,
            decimal=decimal_places
        )
        self.assertTrue(test_almost_equal is None)

    def test_SdA_pretraining_functions(self):
        train_set = shared(value=self.dat.values, borrow=True)
        pretraining_fns = self.sda.pretraining_functions(
            train_set=train_set,
            batch_size=self.dat.shape[0]
        )
        for i in range(self.sda.n_layers):
            c = []
            c.append(
                pretraining_fns[i](
                    index=0,
                    corruption_level=0,
                    learning_rate=0.1)
            )
        sda_cost = np.mean(c)
        self.assertAlmostEqual(sda_cost, self.seq_cost, places=decimal_places)


if __name__ == '__main__':
    unittest.main()
