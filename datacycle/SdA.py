# import ipdb
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from theano import In

from .dA import dA
from .settings import numpy_random_seed
from .settings import theano_random_seed


class SdA(object):
    '''Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    '''

    def __init__(self, n_ins, hidden_layers_sizes,
                 np_rs=None, theano_rs=None, field_weights=None,
                 input_dat=None):
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
        if input_dat is None:
            input_dat = T.matrix(name='input_dat')
        self.x = input_dat
        outputs = []
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
                dA_layer = dA(
                    n_visible=n_ins,
                    n_hidden=hidden_layers_sizes[i],
                    np_rs=np_rs,
                    theano_rs=theano_rs,
                    field_weights=field_weights,
                    input_dat=layer_input,
                )
            else:
                layer_input = outputs[-1]
                dA_layer = dA(
                    n_visible=hidden_layers_sizes[i - 1],
                    n_hidden=hidden_layers_sizes[i],
                    np_rs=np_rs,
                    theano_rs=theano_rs,
                    input_dat=layer_input
                )
            # ipdb.set_trace()
            outputs.append(dA_layer.get_hidden_values(layer_input))
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)

    def get_final_hidden_layer(self, input_dat):
        '''
        Given input, gets the final output.
        '''
        assert len(self.dA_layers) > 0
        h_values = []
        h_values.append(input_dat)
        for da in self.dA_layers:
            h_values.append(da.get_hidden_values(h_values[-1]))
        return h_values[-1]

    def get_first_reconstructed_input(self, hidden):
        '''
        Given output, reconstructs the input from the last layer.
        '''
        assert len(self.dA_layers) > 0
        v_values = []
        v_values.append(hidden)
        for da_layer in reversed(self.dA_layers):
            v_values.append(da_layer.get_reconstructed_input(v_values[-1]))
        return v_values[-1]

    def pretraining_functions(self, train_set, batch_size):
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
