import numpy as np
import theano
from theano import tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams


class dA(object):
    """
    Denoising Auto-Encoder class (dA).
    """

    def __init__(
        self,
        np_rds,
        n_visible,
        n_hidden,
        theano_rds=None,
        input_dat=None,
        field_weights=None,
        W=None,
        bhid=None,
        bvis=None
    ):
        '''
        theano_rds:  Theano random generator that gives symbolic random values
        field_weights:  put on each field when calculating the cost
                        if not given, all fields given equal weight ones
        '''
        if not theano_rds:
            theano_rds = RandomStreams(np_rds.randint(2 ** 30))
        self.theano_rds = theano_rds
        if not field_weights:
            field_weights = np.ones(
                n_visible,
                dtype=theano.config.floatX
            )
        else:
            field_weights = np.asarray(
                field_weights,
                dtype=theano.config.floatX
            )
        self.field_weights = shared(
            value=field_weights,
            name='field_weights',
            borrow=True
        )
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(
                np_rds.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = shared(value=initial_W, name='W', borrow=True)
        self.W = W
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        if not bvis:
            initial_bvis = np.zeros(n_visible, dtype=theano.config.floatX)
            bvis = shared(value=initial_bvis, name='bvis', borrow=True)
        if not bhid:
            initial_bhid = np.zeros(n_hidden, dtype=theano.config.floatX)
            bhid = shared(value=initial_bhid, name='bhid', borrow=True)
        # b corresponds to the bias of the hidden
        self.bhid = bhid
        # b_prime corresponds to the bias of the visible
        self.bhid_prime = bvis
        # if no input_dat is given, generate a variable representing the input
        if input_dat is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input_dat')
        else:
            self.x = input_dat
        self.params = [self.W, self.bhid, self.bhid_prime]

    def get_corrupted_input(self, input_dat, corruption_level):
        assert corruption_level >= 0 and corruption_level < 1
        return self.theano_rds.binomial(
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
