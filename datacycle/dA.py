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
        np_rng,
        theano_rng=None,
        input_dat=None,
        n_visible=None,
        n_hidden=None,
        field_weights=None,
        W=None,
        bhid=None,
        bvis=None
    ):
        '''
        theano_rng:  Theano random generator that gives symbolic random values
        field_weights:  put on each field when calculating the cost
                        if not given, all fields given equal weight ones
        '''
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        self.theano_rng = theano_rng
        if not field_weights:
            field_weights = np.ones(
                self.n_visible,
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
                np_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(self.n_visible, self.n_hidden)
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
        """
        This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        assert corruption_level >= 0 and corruption_level < 1
        return self.theano_rng.binomial(
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
