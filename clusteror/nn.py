'''
This module comprises of classes for neural networks.
'''
# import ipdb
import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import In
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from .settings import numpy_random_seed
from .settings import theano_random_seed


class dA(object):
    '''
    Denoising Autoencoder (DA) class.

    Parameters
    ----------

    n_visible: int
        Input dimension.
    n_hidden: int
        Output dimension.
    np_rs: Numpy function
        Numpy random state.
    theano_rs: Theano function
        Theano random generator that gives symbolic random values.
    field_importance:  list or Numpy array
        Put on each field when calculating the cost.  If not given,
        all fields given equal weight ones.
    initial_W:  Numpy matrix
        Initial weight matrix. Dimension (n_visible, n_hidden).
    initial_bvis: Numpy array
        Initial bias on input side. Dimension n_visible.
    initial_bhid: Numpy arry
        Initial bias on output side. Dimension n_hidden.
    input_data: Theano symbolic variable
        Variable for input data.

    Attributes
    ----------
    theano_rs: Theano function
        Theano random generator that gives symbolic random values.
    field_importance:  list or Numpy array
        Put on each field when calculating the cost.  If not given,
        all fields given equal weight ones.
    W: Theano shared variable
        Weight matrix. Dimension (n_visible, n_hidden).
    W_prime: Theano shared variable
        Transposed weight matrix. Dimension (n_hidden, n_visible).
    bhid: Theano shared variable
        Bias on output side. Dimension n_hidden.
    bvis: Theano shared variable
        Bias on input side. Dimension n_visible.
    x: Theano symbolic variable
        Used as input to build graph.
    params: list
        List packs neural network paramters.
    '''
    def __init__(self, n_visible, n_hidden,
                 np_rs=None, theano_rs=None, field_importance=None,
                 initial_W=None, initial_bvis=None,
                 initial_bhid=None, input_data=None):
        if np_rs is None:
            np_rs = np.random.RandomState(numpy_random_seed)
        # set theano random state if not given
        if theano_rs is None:
            theano_rs = RandomStreams(np_rs.randint(theano_random_seed))
        self.theano_rs = theano_rs
        # set equal field weights if not given
        if not field_importance:
            field_importance = np.ones(n_visible, dtype=theano.config.floatX)
        else:
            field_importance = np.asarray(
                field_importance,
                dtype=theano.config.floatX
            )
        # store in a shared variable
        self.field_importance = shared(
            value=field_importance,
            name='field_importance',
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
        self.bvis = shared(
            value=initial_bvis,
            name='bvis',
            borrow=True
        )
        if initial_bhid is None:
            initial_bhid = np.zeros(n_hidden, dtype=theano.config.floatX)
        # b corresponds to the bias of the hidden
        self.bhid = shared(value=initial_bhid, name='bhid', borrow=True)
        # if no input_data is given, generate a variable representing the input
        if input_data is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.matrix(name='input_data')
        else:
            self.x = input_data
        self.params = [self.W, self.bhid, self.bvis]

    def get_corrupted_input(self, input_data, corruption_level):
        '''
        Corrupts the input by multiplying input with an array of zeros and
        ones that is generated by binomial trials.

        Parameters
        ----------
        input_data: Theano symbolic variable
            Data input to neural network.
        corruption_level: float or Theano symbolic variable
            Probability to corrupt a bit in the input data. Between 0 and 1.

        Returns
        -------
        Theano graph
            A graph with output as the corrupted input.
        '''
        corrupted_input = self.theano_rs.binomial(
            size=input_data.shape,
            n=1,
            p=1 - corruption_level,
            dtype=theano.config.floatX
        ) * input_data
        return corrupted_input

    def get_hidden_values(self, input_data):
        '''
        Computes the values of the hidden layer.

        Parameters
        ----------
        input_data: Theano symbolic variable
            Data input to neural network.

        Returns
        -------
        Theano graph
            A graph with output as the hidden layer values.
        '''
        return T.tanh(T.dot(input_data, self.W) + self.bhid)

    def get_reconstructed_input(self, hidden):
        '''
        Computes the reconstructed input given the values of the
        hidden layer.

        Parameters
        ----------
        hidden: Theano symbolic variable
            Data input to neural network at the hidden layer side.

        Returns
        -------
        Theano graph
            A graph with output as the reconstructed data at the visible side.
        '''
        return T.tanh(T.dot(hidden, self.W_prime) + self.bvis)

    def get_cost_updates(self, corruption_level, learning_rate):
        '''
        This function computes the cost and the updates for one trainng
        step of the dA.

        Parameters
        ----------
        corruption_level: float or Theano symbolic variable
            Probability to corrupt a bit in the input data. Between 0 and 1.
        learning_rate: float or Theano symbolic variable
            Step size for Gradient Descent algorithm.

        Returns
        -------
        cost: Theano graph
            A graph with output as the cost.
        updates: List of tuples
            Instructions of how to update parameters. Used in training stage
            to update parameters.
        '''
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # need this cross entropy because now the x and z are in the
        # range [-1, 1]
        L = - T.sum(
            self.field_importance * (
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


class SdA(object):
    '''
    Stacked Denoising Autoencoder (SDA) class.

    A SdA model is obtained by stacking several DAs.
    The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.

    Parameters
    ----------

    n_ins: int
        Input dimension.
    hidden_layers_sizes: list of int
        Each int will be assgined to each hidden layer. Same number of hidden
        layers will be created.
    np_rs: Numpy function
        Numpy random state.
    theano_rs: Theano function
        Theano random generator that gives symbolic random values.
    field_importance:  list or Numpy array
        Put on each field when calculating the cost.  If not given,
        all fields given equal weight ones.
    input_data: Theano symbolic variable
        Variable for input data.

    Attributes
    ----------
    theano_rs: Theano function
        Theano random generator that gives symbolic random values.
    field_importance:  list or Numpy array
        Put on each field when calculating the cost.  If not given,
        all fields given equal weight ones.
    W: Theano shared variable
        Weight matrix. Dimension (n_visible, n_hidden).
    W_prime: Theano shared variable
        Transposed weight matrix. Dimension (n_hidden, n_visible).
    bhid: Theano shared variable
        Bias on output side. Dimension n_hidden.
    bvis: Theano shared variable
        Bias on input side. Dimension n_visible.
    x: Theano symbolic variable
        Used as input to build graph.
    params: list
        List packs neural network paramters.
    dA_layers: list
        List that keeps dA instances.
    n_layers: int
        Number of hidden layers, len(dA_layers).
    '''

    def __init__(self, n_ins, hidden_layers_sizes,
                 np_rs=None, theano_rs=None, field_importance=None,
                 input_data=None):
        # set theano random state if not given
        if np_rs is None:
            np_rs = np.random.RandomState(numpy_random_seed)
        if theano_rs is None:
            theano_rs = RandomStreams(np_rs.randint(theano_random_seed))
        self.theano_rs = theano_rs
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0
        if input_data is None:
            input_data = T.matrix(name='input_data')
        self.x = input_data
        outputs = []
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
                dA_layer = dA(
                    n_visible=n_ins,
                    n_hidden=hidden_layers_sizes[i],
                    np_rs=np_rs,
                    theano_rs=theano_rs,
                    field_importance=field_importance,
                    input_data=layer_input,
                )
            else:
                layer_input = outputs[-1]
                dA_layer = dA(
                    n_visible=hidden_layers_sizes[i - 1],
                    n_hidden=hidden_layers_sizes[i],
                    np_rs=np_rs,
                    theano_rs=theano_rs,
                    input_data=layer_input
                )
            # ipdb.set_trace()
            outputs.append(dA_layer.get_hidden_values(layer_input))
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)

    def get_final_hidden_layer(self, input_data):
        '''
        Computes the values of the last hidden layer.

        Parameters
        ----------
        input_data: Theano symbolic variable
            Data input to neural network.

        Returns
        -------
        Theano graph
            A graph with output as the hidden layer values.
        '''
        assert len(self.dA_layers) > 0
        h_values = []
        h_values.append(input_data)
        for da in self.dA_layers:
            h_values.append(da.get_hidden_values(h_values[-1]))
        return h_values[-1]

    def get_first_reconstructed_input(self, hidden):
        '''
        Computes the reconstructed input given the values of the last
        hidden layer.

        Parameters
        ----------
        hidden: Theano symbolic variable
            Data input to neural network at the hidden layer side.

        Returns
        -------
        Theano graph
            A graph with output as the reconstructed data at the visible side.
        '''
        assert len(self.dA_layers) > 0
        v_values = []
        v_values.append(hidden)
        for da_layer in reversed(self.dA_layers):
            v_values.append(da_layer.get_reconstructed_input(v_values[-1]))
        return v_values[-1]

    def pretraining_functions(self, train_set, batch_size):
        '''
        This function computes the cost and the updates for one trainng
        step of the dA.

        Parameters
        ----------
        train_set: Theano shared variable
            The complete training dataset.
        batch_size: int
            Number of rows for each mini-batch.

        Returns
        -------
        List
            Theano functions that run one step training on each dA layers.
        '''
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        # fraction of corruption to use
        corruption_level = T.scalar('corruption_level')
        # learning rate to use
        learning_rate = T.scalar('learning_rate')
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
        pretrain_fns = []
        for da in self.dA_layers:
            # get the cost and the updates list
            cost, updates = da.get_cost_updates(
                corruption_level,
                learning_rate
            )
            # compile the theano function
            fn = function(
                inputs=[
                    index,
                    In(corruption_level, value=0.2),
                    In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns
