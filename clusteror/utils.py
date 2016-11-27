""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import numpy
import pandas as pd
import theano
import theano.tensor as T
from scipy.special import expit


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(
        numpy.asarray(
            data_x,
            dtype=theano.config.floatX),
        borrow=borrow)
    shared_y = theano.shared(
        numpy.asarray(
            data_y,
            dtype=theano.config.floatX),
        borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def shared_dataset_all_float(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(
        numpy.asarray(
            data_x,
            dtype=theano.config.floatX),
        borrow=borrow)
    shared_y = theano.shared(
        numpy.asarray(
            data_y,
            dtype=theano.config.floatX),
        borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y


def load_kaggle_data(train_data):
    data = pd.read_csv('train.csv')
    data = data.copy()
    random_num = numpy.random.uniform(size=42000)
    train_set = (
        data.loc[(random_num < .6), 'pixel0':'pixel783'].values / 255,
        data.loc[(random_num < .6), 'label'].values)
    valid_set = (
        data
        .loc[
            ((random_num > .6) & (random_num < .8)),
            'pixel0':'pixel783']
        .values / 255,
        data
        .loc[
            ((random_num > .6) & (random_num < .8)),
            'label'])
    test_set = (
        data.loc[(random_num > .8), 'pixel0':'pixel783'].values / 255,
        data.loc[(random_num > .8), 'label'].values)
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_kaggle_winton_data():
    data = pd.read_csv('train.csv')
    data = data.fillna(0).copy()
    length = data.shape[0]
    random_num = numpy.random.uniform(size=length)
    train_set = (
        expit(data.loc[(random_num < .8), 'Ret_MinusTwo':'Ret_120'].values),
        data.loc[(random_num < .8), 'Ret_121':'Ret_PlusTwo'].values)
    valid_set = (
        expit(data
              .loc[
                  ((random_num > .8) & (random_num < .9)),
                  'Ret_MinusTwo':'Ret_120'].values),
        data
        .loc[
            ((random_num > .8) & (random_num < .9)),
            'Ret_121':'Ret_PlusTwo'])
    test_set = (
        expit(data.loc[(random_num > .9), 'Ret_MinusTwo':'Ret_120'].values),
        data.loc[(random_num > .9), 'Ret_121':'Ret_PlusTwo'])
    test_set_x, test_set_y = shared_dataset_all_float(test_set)
    valid_set_x, valid_set_y = shared_dataset_all_float(valid_set)
    train_set_x, train_set_y = shared_dataset_all_float(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def confirm_proceed(to_proc, no_message):
    if to_proc == 'y' or to_proc == '':
        pass
    elif to_proc == 'n':
        raise SystemExit('{}'.format(no_message))
    else:
        raise SystemExit('Unknown answer!')


def datetime_to_string(dt):
    if isinstance(dt, pd.tslib.NaTType):
        return None
    year = int(dt.year)
    month = int(dt.month)
    day = int(dt.day)
    hour = (
        int(dt.hour) if dt.hour is not numpy.nan else int(0))
    minute = (
        int(dt.minute) if dt.minute is not numpy.nan else int(0))
    second = (
        int(dt.second) if dt.second is not numpy.nan else int(0))
    dt_string = (
        str(year) +
        '-' +
        '{:02d}'.format(month) +
        '-' +
        '{:02d}'.format(day) +
        ' ' +
        '{:02d}'.format(hour) +
        ':' +
        '{:02d}'.format(minute) +
        ':' +
        '{:02d}'.format(second))
    return dt_string


def not_null(series):
        return series.notnull().value_counts()


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)
        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def check_local_extremity(series, ind, contrast=0.15, kind='min'):
    if kind == 'min':
        value = series.iloc[ind]
        upper_benchmark = value * (1 + contrast)

        left_sub = series.iloc[:ind]
        if left_sub.empty:
            smallest_to_left = True
        else:
            left_larger = left_sub > upper_benchmark
            left_trues = left_larger.loc[left_larger.values]
            if left_trues.empty:
                smallest_to_left = False
            else:
                left_ind = left_trues.index[-1]
                left_smallest = series.iloc[left_ind:ind].min()
                smallest_to_left = left_smallest > value

        right_sub = series.iloc[ind+1:]
        if right_sub.empty:
            smallest_to_right = True
        else:
            right_larger = right_sub > upper_benchmark
            right_trues = right_larger.loc[right_larger.values]
            if right_trues.empty:
                smallest_to_right = False
            else:
                right_ind = right_trues.index[0]
                right_smallest = series.iloc[ind+1:right_ind+1].min()
                smallest_to_right = right_smallest > value
        return smallest_to_left and smallest_to_right
    if kind == 'max':
        value = series.iloc[ind]
        lower_benchmark = value * (1 - contrast)

        left_sub = series.iloc[:ind]
        if left_sub.empty:
            largest_to_left = True
        else:
            left_smaller = left_sub < lower_benchmark
            left_trues = left_smaller.loc[left_smaller.values]
            if left_trues.empty:
                largest_to_left = False
            else:
                left_ind = left_trues.index[-1]
                left_largest = series.iloc[left_ind:ind].max()
                largest_to_left = left_largest < value

        right_sub = series.iloc[ind+1:]
        if right_sub.empty:
            largest_to_right = True
        else:
            right_smaller = right_sub < lower_benchmark
            right_trues = right_smaller.loc[right_smaller.values]
            if right_trues.empty:
                largest_to_right = False
            else:
                right_ind = right_trues.index[0]
                right_largest = series.iloc[ind+1:right_ind+1].max()
                largest_to_right = right_largest < value
        return largest_to_left and largest_to_right
