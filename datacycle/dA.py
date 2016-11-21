import numpy as np
import theano
from theano import tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from .settings import numpy_random_seed
from .settings import theano_random_seed


class dA(object):
    """
    Denoising Auto-Encoder class (dA).
    """
    def __init__(self, n_visible, n_hidden,
                 theano_rs=None, field_weights=None,
                 initial_W=None, initial_bvis=None,
                 initial_bhid=None, input_dat=None):
        '''
        theano_rs:  Theano random generator that gives symbolic random values
        field_weights:  put on each field when calculating the cost
                        if not given, all fields given equal weight ones
        '''
        # set theano random state if not given
        if not theano_rs:
            np_rs = np.random.RandomState(numpy_random_seed)
            theano_rs = RandomStreams(np_rs.randint(theano_random_seed))
        self.theano_rs = theano_rs
        # set equal field weights if not given
        if not field_weights:
            field_weights = np.ones(n_visible, dtype=theano.config.floatX)
        else:
            field_weights = np.asarray(
                field_weights,
                dtype=theano.config.floatX
            )
        # store in a shared variable
        self.field_weights = shared(
            value=field_weights,
            name='field_weights',
            borrow=True
        )
        # note : W' was written as `W_prime` and b' as `b_prime`
        if initial_W is None:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(
                np_rs.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
        self.W = shared(value=initial_W, name='W', borrow=True)
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        if initial_bvis is None:
            initial_bvis = np.zeros(n_visible, dtype=theano.config.floatX)
        # b_prime corresponds to the bias of the visible
        self.bhid_prime = shared(value=initial_bvis, name='bvis', borrow=True)
        if initial_bhid is None:
            initial_bhid = np.zeros(n_hidden, dtype=theano.config.floatX)
        # b corresponds to the bias of the hidden
        self.bhid = shared(value=initial_bhid, name='bhid', borrow=True)
        # if no input_dat is given, generate a variable representing the input
        if input_dat is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input_dat')
        else:
            self.x = input_dat
        self.params = [self.W, self.bhid, self.bhid_prime]

    def get_corrupted_input(self, input_dat, corruption_level):
        corrup_info = 'Must be between 0 and 1.'
        assert corruption_level >= 0 and corruption_level < 1, corrup_info
        return self.theano_rs.binomial(
            size=input_dat.shape,
            n=1,
            p=1 - corruption_level,
            dtype=theano.config.floatX
        ) * input_dat

    def get_hidden_values(self, input_dat):
        """
        Computes the values of the hidden layer.
        """
        return T.tanh(T.dot(input_dat, self.W) + self.bhid)

    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed input given the values of the
        hidden layer.
        """
        return T.tanh(T.dot(hidden, self.W_prime) + self.bhid_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """
        This function computes the cost and the updates for one trainng
        step of the dA.
        """
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # need this cross entropy because now the x and z are in the
        # range [-1, 1]
        L = - T.sum(
            self.field_weights * (
                .5 * (1 + self.x) * T.log(.5 * (1 + z)) +
                .5 * (1 - self.x) * T.log(.5 * (1 - z))
            ),
            axis=1
        )
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return cost, updates
