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


def test_dA(dat, learning_rate=0.1, training_epochs=15,
            corruption_level=0.3, batch_size=200, initial_W=None,
            initial_bvis=None, initial_bhid=None):
    x = T.matrix('x')  # the data is presented as rasterized images
    da = dA(
        n_visible=dat.shape[1],
        n_hidden=1,
        initial_W=initial_W,
        initial_bvis=initial_bvis,
        initial_bhid=initial_bhid,
        input_dat=x,
    )
    cost, updates = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )
    train_da = function(
        [x],
        cost,
        updates=updates,
    )
    return np.mean(train_da(dat))


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


class TestDatacycle(unittest.TestCase):
    def test_dA_cost(self):
        dat = pd.read_csv('makeup_test_data.csv')
        dat = np.asarray(dat, dtype=theano.config.floatX)
        initial_W = np.asarray([[1], [2], [3]], dtype=theano.config.floatX)
        initial_bvis = np.asarray([0, 0, 0], dtype=theano.config.floatX)
        initial_bhid = np.asarray([0], dtype=theano.config.floatX)
        da_cost = test_dA(
            dat,
            learning_rate=0.02,
            corruption_level=0,
            training_epochs=1,
            batch_size=4,
            initial_W=initial_W,
            initial_bvis=initial_bvis,
            initial_bhid=initial_bhid
        )
        y = np.tanh(np.dot(dat, initial_W) + initial_bhid)
        z = np.tanh(np.dot(y, initial_W.T) + initial_bvis)
        cost = - np.sum(
            0.5 * (1 + dat) * np.log(0.5 * (1 + z)) +
            0.5 * (1 - dat) * np.log(0.5 * (1 - z)),
            axis=1
        )
        cost = np.mean(cost)
        self.assertAlmostEqual(da_cost, cost, places=decimal_places)


if __name__ == '__main__':
    # unittest.main()
    dat = pd.read_csv('makeup_test_data.csv')
    test_SdA(dat)
