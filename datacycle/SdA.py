import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from theano import Param

from mlp import HiddenLayer
from .dA import dA


class SdA(object):
    '''Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    '''

    def __init__(
        self,
        np_rng,
        hidden_layers_sizes,
        theano_rng=None,
        n_ins=None,
        n_outs=None,
        corruption_levels=None,
    ):
        self.tanh_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0
        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        # end-snippet-1
        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in range(self.n_layers):
            # construct the sigmoidal layer
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.tanh_layers[-1].output
            tanh_layer = HiddenLayer(
                np_rng=np_rng,
                input_dat=layer_input,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.tanh
                )
            # add the layer to our list of layers
            self.tanh_layers.append(tanh_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # tanh_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(tanh_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(
                np_rng=np_rng,
                theano_rng=theano_rng,
                input_dat=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                W=tanh_layer.W,
                bhid=tanh_layer.b
            )
            self.dA_layers.append(dA_layer)
        self.y_pred = self.top_layer.y_pred

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

    def pretraining_functions(self, train_set_x, batch_size):
        '''
        Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
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
                    Param(corruption_level, default=0.2),
                    Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_valid_batches += 1
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size
        n_test_batches += 1

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score
