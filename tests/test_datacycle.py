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
    unittest.main()

    # x = T.dmatrix('x')
    # output = da.get_hidden_values(x)
    # fewd = function([x], output)
    # recon = da.get_reconstructed_input(x)
    # multid = function([x], recon)

# def test_SdA_without_finetune(
#         dat,
#         pretrain_epochs=15,
#         pretrain_lr=0.001,
#         hidden_layers_sizes=[40, 20, 2],
#         corruption_levels=[.3, .3, .3, .3, .3],
#         batch_size=1,
#         ):
#     '''
#     Demonstrates how to train and test a stochastic denoising autoencoder.
# 
#     This is demonstrated on MNIST.
# 
#     :type learning_rate: float
#     :param learning_rate: learning rate used in the finetune stage
#     (factor for the stochastic gradient)
# 
#     :type pretrain_epochs: int
#     :param pretrain_epochs: number of epoch to do pretraining
# 
#     :type pretrain_lr: float
#     :param pretrain_lr: learning rate to be used during pre-training
# 
#     :type n_iter: int
#     :param n_iter: maximal number of iterations ot run the optimizer
# 
#     :type dataset: string
#     :param dataset: path the the pickled dataset
# 
#     '''
# 
#     dat = numpy.asarray(dat, dtype=theano.config.floatX)
#     train_set_x = theano.shared(value=dat, borrow=True)
#     # compute number of minibatches for training, validation and testing
#     n_train_batches = train_set_x.get_value(borrow=True).shape[0]
#     n_train_batches //= batch_size
#     n_train_batches += 1
# 
#     # numpy random generator
#     # start-snippet-3
#     numpy_rs = numpy.random.RandomState(89677)
#     print('... building the model')
#     # construct the stacked denoising autoencoder class
#     sda = SdA(
#         numpy_rs=numpy_rs,
#         n_ins=dat.shape[1],
#         hidden_layers_sizes=hidden_layers_sizes,
#     )
#     # end-snippet-3 start-snippet-4
#     #########################
#     # PRETRAINING THE MODEL #
#     #########################
#     print('... getting the pretraining functions')
#     pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
#                                                 batch_size=batch_size)
# 
#     print('... pre-training the model')
#     start_time = timeit.default_timer()
#     # Pre-train layer-wise
#     for i in range(sda.n_layers):
#         # go through pretraining epochs
#         for epoch in range(pretrain_epochs):
#             # go through the training set
#             c = []
#             for batch_index in range(n_train_batches):
#                 c.append(
#                     pretraining_fns[i](
#                         index=batch_index,
#                         corruption=corruption_levels[i],
#                         lr=pretrain_lr))
#             print(
#                 'Pre-training layer {}, '.format(i) +
#                 'epoch {epoch}, '.format(epoch=epoch) +
#                 'cost {cost:f}'.format(cost=numpy.mean(c))
#                 )
# 
#     end_time = timeit.default_timer()
# 
#     sys.stderr.write('The pretraining code for file ' +
#                      os.path.split(__file__)[1] +
#                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
#     return sda
# 
# 
# def test_SdA_with_finetune(
#         dataset,
#         finetune_lr=0.1,
#         pretrain_epochs=15,
#         learning_rate=0.02,
#         hidden_layers_sizes=[200, 100, 50],
#         corruption_levels=[0.1, 0.1, 0.1],
#         training_epochs=1000,
#         batch_size=1,
#         method='logistic_regression',
#         scoring='mean_absolute_error'):
#     '''
#     Demonstrates how to train and test a stochastic denoising autoencoder.
# 
#     This is demonstrated on MNIST.
# 
#     :type learning_rate: float
#     :param learning_rate: learning rate used in the finetune stage
#     (factor for the stochastic gradient)
# 
#     :type pretrain_epochs: int
#     :param pretrain_epochs: number of epoch to do pretraining
# 
#     :type pretrain_lr: float
#     :param pretrain_lr: learning rate to be used during pre-training
# 
#     :type n_iter: int
#     :param n_iter: maximal number of iterations ot run the optimizer
# 
#     :type dataset: string
#     :param dataset: path the the pickled dataset
# 
#     '''
# 
#     # datasets = load_data(dataset)
#     datasets = load_kaggle_winton_data()
# 
#     train_set_x, train_set_y = datasets[0]
#     valid_set_x, valid_set_y = datasets[1]
#     test_set_x, test_set_y = datasets[2]
# 
#     # compute number of minibatches for training, validation and testing
#     n_train_batches = train_set_x.get_value(borrow=True).shape[0]
#     n_train_batches //= batch_size
#     n_train_batches += 1
# 
#     # numpy random generator
#     # start-snippet-3
#     numpy_rs = numpy.random.RandomState(89677)
#     print('... building the model')
#     # construct the stacked denoising autoencoder class
#     sda = SdA(
#         numpy_rs=numpy_rs,
#         n_ins=train_set_x.get_value(borrow=True).shape[1],
#         hidden_layers_sizes=hidden_layers_sizes,
#         n_outs=train_set_y.get_value(borrow=True).shape[1],
#         corruption_levels=[0.1, 0.1, 0.1],
#         method=method,
#         scoring=scoring,
#     )
#     # end-snippet-3 start-snippet-4
#     #########################
#     # PRETRAINING THE MODEL #
#     #########################
#     print('... getting the pretraining functions')
#     pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
#                                                 batch_size=batch_size)
# 
#     print('... pre-training the model')
#     start_time = timeit.default_timer()
#     # Pre-train layer-wise
#     corruption_levels = [.1, .2, .3]
#     for i in range(sda.n_layers):
#         # go through pretraining epochs
#         for epoch in range(pretrain_epochs):
#             # go through the training set
#             c = []
#             for batch_index in range(n_train_batches):
#                 c.append(pretraining_fns[i](index=batch_index,
#                          corruption=corruption_levels[i],
#                          lr=learning_rate))
#             print(
#                 'Pre-training layer {i}, epoch {epoch}, cost {cost}'
#                 .format(i=i, epoch=epoch, cost=numpy.mean(c)))
# 
#     end_time = timeit.default_timer()
# 
#     sys.stderr.write('The pretraining code for file ' +
#                      os.path.split(__file__)[1] +
#                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
#     # end-snippet-4
#     ########################
#     # FINETUNING THE MODEL #
#     ########################
# 
#     # get the training, validation and testing function for the model
#     print('... getting the finetuning functions')
#     train_fn, validate_model, test_model = sda.build_finetune_functions(
#         datasets=datasets,
#         batch_size=batch_size,
#         learning_rate=finetune_lr
#     )
# 
#     print('... finetunning the model')
#     # early-stopping parameters
#     # look as this many examples regardless
#     patience = 10 * n_train_batches
#     # wait this much longer when a new best is found
#     patience_increase = 2
#     # a relative improvement of this much is considered significant
#     improvement_threshold = 0.995
#     # go through this many
#     # minibatche before checking the network
#     # on the validation set; in this case we
#     # check every epoch
#     validation_frequency = min(n_train_batches, patience//2)
# 
#     best_validation_loss = numpy.inf
#     test_score = 0.
#     start_time = timeit.default_timer()
# 
#     done_looping = False
#     epoch = 0
# 
#     while (epoch < training_epochs) and (not done_looping):
#         epoch = epoch + 1
#         for minibatch_index in range(n_train_batches):
#             train_fn(minibatch_index)
#             iter_num = (epoch - 1) * n_train_batches + minibatch_index
# 
#             if (iter_num + 1) % validation_frequency == 0:
#                 validation_losses = validate_model()
#                 this_validation_loss = numpy.mean(validation_losses)
#                 if method == 'linear_regression':
#                     print('epoch %i, minibatch %i/%i, validation error %f' %
#                           (epoch, minibatch_index + 1, n_train_batches,
#                            this_validation_loss))
#                 elif method == 'logistic_regression':
#                     print('epoch %i, minibatch %i/%i, validation error %f %%' %
#                           (epoch, minibatch_index + 1, n_train_batches,
#                            this_validation_loss * 100))
# 
#                 # if we got the best validation score until now
#                 if this_validation_loss < best_validation_loss:
# 
#                     # improve patience if loss improvement is good enough
#                     if (
#                         this_validation_loss < best_validation_loss *
#                         improvement_threshold
#                     ):
#                         patience = max(patience, iter_num * patience_increase)
# 
#                     # save best validation score and iteration number
#                     best_validation_loss = this_validation_loss
# 
#                     # test it on the test set
#                     test_losses = test_model()
#                     test_score = numpy.mean(test_losses)
#                     if method == 'linear_regression':
#                         print(('     epoch %i, minibatch %i/%i, test error of '
#                                'best model %f') %
#                               (epoch, minibatch_index + 1, n_train_batches,
#                                test_score))
#                     elif method == 'logistic_regression':
#                         print(('     epoch %i, minibatch %i/%i, test error of '
#                                'best model %f %%') %
#                               (epoch, minibatch_index + 1, n_train_batches,
#                                test_score * 100))
# 
#             if patience <= iter_num:
#                 done_looping = True
#                 break
# 
#     end_time = timeit.default_timer()
#     if method == 'linear_regression':
#         print(
#             'Optimization complete with best validation' +
#             'score of {:f}, '.format(best_validation_loss) +
#             'with test performance {:f}'.format(test_score)
#             )
#     elif method == 'logistic_regression':
#         print(
#             'Optimization complete with best validation' +
#             'score of {:f} %%, '.format(best_validation_loss*100) +
#             'with test performance {:f} %%'.format(test_score*100)
#             )
#     sys.stderr.write('The training code for file ' +
#                      os.path.split(__file__)[1] +
#                      ' ran for %.2fm\n' % ((end_time - start_time) / 60.))
#     return sda
# 
# 
# if __name__ == '__main__':
#     # import pandas as pd
#     # from scipy.special import expit
#     # from sklearn.preprocessing import StandardScaler
#     # # test the sda without regression layer
#     # ss = StandardScaler()
#     # dat = pd.read_csv('clustering2013_2014.csv', index_col=0, nrows=1000)
#     # sda = test_SdA_without_finetune(
#     #     expit(
#     #         ss.fit_transform(
#     #             dat.loc[:, ['bookings', 'passengers',
#     #                         'luggage_fee', 'infant_fee',
#     #                         'unique_passengers']])),
#     #     pretrain_lr=0.02,
#     #     pretrain_epochs=100,
#     #     hidden_layers_sizes=[2, 1],
#     #     batch_size=100)
#     sda = test_SdA_with_finetune(
#         'dataset',
#         finetune_lr=0.01,
#         pretrain_epochs=100,
#         learning_rate=0.01,
#         hidden_layers_sizes=[200, 100, 50],
#         corruption_levels=[0.1, 0.1, 0.1],
#         training_epochs=500,
#         batch_size=10000,
#         method='linear_regression',
#         scoring='mean_absolute_error')
# 
# 
# def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
#              dataset='mnist.pkl.gz', batch_size=200, n_hidden=500):
#     """
#     Demonstrate stochastic gradient descent optimization for a multilayer
#     perceptron
# 
#     This is demonstrated on MNIST.
# 
#     :type learning_rate: float
#     :param learning_rate: learning rate used (factor for the stochastic
#     gradient
# 
#     :type L1_reg: float
#     :param L1_reg: L1-norm's weight when added to the cost (see
#     regularization)
# 
#     :type L2_reg: float
#     :param L2_reg: L2-norm's weight when added to the cost (see
#     regularization)
# 
#     :type n_epochs: int
#     :param n_epochs: maximal number of epochs to run the optimizer
# 
#     :type dataset: string
#     :param dataset: the path of the MNIST dataset file from
#                  http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
# 
# 
#    """
#     datasets = load_kaggle_data(dataset)
# 
#     train_set_x, train_set_y = datasets[0]
#     valid_set_x, valid_set_y = datasets[1]
#     test_set_x, test_set_y = datasets[2]
# 
#     # compute number of minibatches for training, validation and testing
#     n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
#     n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
#     n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
# 
#     ######################
#     # BUILD ACTUAL MODEL #
#     ######################
#     print('... building the model')
# 
#     # allocate symbolic variables for the data
#     index = T.lscalar()  # index to a [mini]batch
#     x = T.matrix('x')  # the data is presented as rasterized images
#     # the labels are presented as 1D vector of [int] labels
#     y = T.ivector('y')
# 
#     rs = np.random.RandomState(1234)
# 
#     # construct the MLP class
#     classifier = MLP(
#         rs=rs,
#         input_dat=x,
#         n_in=28 * 28,
#         n_hidden=n_hidden,
#         n_out=10
#     )
# 
#     # start-snippet-4
#     # the cost we minimize during training is the negative log likelihood of
#     # the model plus the regularization terms (L1 and L2); cost is expressed
#     # here symbolically
#     cost = (
#         classifier.negative_log_likelihood(y)
#         + L1_reg * classifier.L1
#         + L2_reg * classifier.L2_sqr
#     )
#     # end-snippet-4
# 
#     # compiling a Theano function that computes the mistakes that are made
#     # by the model on a minibatch
#     test_model = theano.function(
#         inputs=[index],
#         outputs=classifier.errors(y),
#         givens={
#             x: test_set_x[index * batch_size:(index + 1) * batch_size],
#             y: test_set_y[index * batch_size:(index + 1) * batch_size]
#         }
#     )
# 
#     validate_model = theano.function(
#         inputs=[index],
#         outputs=classifier.errors(y),
#         givens={
#             x: valid_set_x[index * batch_size:(index + 1) * batch_size],
#             y: valid_set_y[index * batch_size:(index + 1) * batch_size]
#         }
#     )
# 
#     # start-snippet-5
#     # compute the gradient of cost with respect to theta (sotred in params)
#     # the resulting gradients will be stored in a list gparams
#     gparams = [T.grad(cost, param) for param in classifier.params]
# 
#     # specify how to update the parameters of the model as a list of
#     # (variable, update expression) pairs
# 
#     # given two lists of the same length, A = [a1, a2, a3, a4] and
#     # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
#     # element is a pair formed from the two lists :
#     #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
#     updates = [
#         (param, param - learning_rate * gparam)
#         for param, gparam in zip(classifier.params, gparams)
#     ]
# 
#     # compiling a Theano function `train_model` that returns the cost, but
#     # in the same time updates the parameter of the model based on the rules
#     # defined in `updates`
#     train_model = theano.function(
#         inputs=[index],
#         outputs=cost,
#         updates=updates,
#         givens={
#             x: train_set_x[index * batch_size: (index + 1) * batch_size],
#             y: train_set_y[index * batch_size: (index + 1) * batch_size]
#         }
#     )
#     # end-snippet-5
# 
#     ###############
#     # TRAIN MODEL #
#     ###############
#     print('... training')
# 
#     # early-stopping parameters
#     patience = 100  # look as this many examples regardless
#     # wait this much longer when a new best is found
#     patience_increase = 2
#     # a relative improvement of this much is considered significant
#     improvement_threshold = 0.995
#     # go through this many
#     # minibatche before checking the network
#     # on the validation set; in this case we
#     # check every epoch
#     validation_frequency = min(n_train_batches, patience / 2)
# 
#     best_validation_loss = np.inf
#     best_iter = 0
#     test_score = 0.
#     start_time = timeit.default_timer()
# 
#     epoch = 0
#     done_looping = False
# 
#     while (epoch < n_epochs) and (not done_looping):
#         epoch = epoch + 1
#         for minibatch_index in range(n_train_batches):
# 
#             train_model(minibatch_index)
#             # iteration number
#             iter_num = (epoch - 1) * n_train_batches + minibatch_index
# 
#             if (iter_num + 1) % validation_frequency == 0:
#                 # compute zero-one loss on validation set
#                 validation_losses = [validate_model(i) for i
#                                      in range(n_valid_batches)]
#                 this_validation_loss = np.mean(validation_losses)
# 
#                 print(
#                     'epoch %i, minibatch %i/%i, validation error %f %%' %
#                     (
#                         epoch,
#                         minibatch_index + 1,
#                         n_train_batches,
#                         this_validation_loss * 100.
#                     )
#                 )
# 
#                 # if we got the best validation score until now
#                 if this_validation_loss < best_validation_loss:
#                     # improve patience if loss improvement is good enough
#                     if (
#                         this_validation_loss < best_validation_loss *
#                         improvement_threshold
#                     ):
#                         patience = max(patience, iter_num * patience_increase)
# 
#                     best_validation_loss = this_validation_loss
#                     best_iter = iter_num
# 
#                     # test it on the test set
#                     test_losses = [test_model(i) for i
#                                    in range(n_test_batches)]
#                     test_score = np.mean(test_losses)
# 
#                     print(('     epoch %i, minibatch %i/%i, test error of '
#                            'best model %f %%') %
#                           (epoch, minibatch_index + 1, n_train_batches,
#                            test_score * 100.))
# 
#             if patience <= iter_num:
#                 done_looping = True
#                 break
# 
#     end_time = timeit.default_timer()
#     print(('Optimization complete. Best validation score of %f %% '
#            'obtained at iteration %i, with test performance %f %%') %
#           (best_validation_loss * 100., best_iter + 1, test_score * 100.))
#     print(sys.stderr, ('The code for file ' +
#           os.path.split(__file__)[1] +
#           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
#     return classifier
# 
# 
# if __name__ == '__main__':
#     classifier = test_mlp()
