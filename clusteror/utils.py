'''
This module works as a transient store of useful functions. New standalone
functions will be first placed here. As they grow in number and can be
consolidated into an independent class, module, or even a new package.
'''


def find_local_extremes(series, contrast):
    '''
    Finds local minima and maxima according to ``contrast``. In theory,
    they can be determined by first derivative and second derivative. The
    result derived this way is of no value in dealing with a very noisy,
    zig-zag data as too many local extremes would be found for any turn-around.
    The method presented here compares the point currently looked at and the
    opposite potential extreme that is updated as scanning through the
    data sequence. For instance, a potential maximum is 10, then a data point
    of value smaller than 10 / (1 + contrast) is written down as a local
    minimum.

    Parameters
    ----------
    series: Pandas Series
        One dimenional data to find local extremes in.
    contrast: float
        A value between 0 and 1 as a threshold between minimum and maximum.

    Returns
    -------
    local_min_inds: list
        List of indices for local minima.
    local_mins: list
        List of minimum values.
    local_max_inds: list
        List of indices for local maxima.
    local_maxs: list
        List of maximum values.
    '''
    state = {
        'pmin': None,
        'pmin_ind': None,
        'pmax': None,
        'pmax_ind': None,
        'lmin': None,
        'lmax': None
    }
    # initialise, all starting points are potential local min and local max
    state['pmin_ind'] = series.index.tolist()[0]
    state['pmin'] = series.iat[0]
    state['pmax_ind'] = series.index.tolist()[0]
    state['pmax'] = series.iat[0]
    # store true local mins and maxes
    local_min_inds = []
    local_mins = []
    local_max_inds = []
    local_maxs = []
    # walk through all rows
    for ind, value in series.iteritems():
        if state['pmin'] is not None and state['pmax'] is not None:
            # when just starts out or find a potential extreme after
            # confirming a local extreme
            if value <= state['pmin']:
                # value is smaller then update potential min
                state['pmin'] = value
                state['pmin_ind'] = ind
                if value * (1 + contrast) <= state['pmax']:
                    # if the gap between current point and potential max is
                    # larger than contrast
                    # then confirm the last potential max is a true local max
                    state['lmax'] = state['pmax']
                    local_max_inds.append(state['pmax_ind'])
                    local_maxs.append(state['pmax'])
                    # so for a moment no potential max
                    state['pmax'] = None
                    state['pmax_ind'] = None
            elif value >= state['pmax']:
                # value is larger then update potential max
                state['pmax'] = value
                state['pmax_ind'] = ind
                if value > state['pmin'] * (1 + contrast):
                    # if the gap between current point and potential min is
                    # larger than contrast
                    # then confirm the last potential min is a true local min
                    state['lmin'] = state['pmin']
                    local_min_inds.append(state['pmin_ind'])
                    local_mins.append(state['pmin'])
                    # so for a moment no potential min
                    state['pmin'] = None
                    state['pmin_ind'] = None
            else:
                # point is between potenital min and potential max
                # just pass without updating
                pass
        elif state['pmax'] is not None and state['lmin'] is not None:
            # when just found a local min, trying to find next local max
            if value >= state['pmax']:
                # update if value is larger
                state['pmax'] = value
                state['pmax_ind'] = ind
            elif value <= state['lmin']:
                # this is where it just after a sharp blip
                # confirm the last point is a local max
                state['lmax'] = state['pmax']
                local_max_inds.append(state['pmax_ind'])
                local_maxs.append(state['pmax'])
                # so for a moment no potential max
                state['pmax'] = None
                state['pmax_ind'] = None
                # the current point becomes a potential min
                state['pmin'] = value
                state['pmin_ind'] = ind
            else:
                # smaller than the last point, so this is a potential min
                state['lmin'] = None
                state['pmin'] = value
                state['pmin_ind'] = ind
        elif state['pmin'] is not None and state['lmax'] is not None:
            # when just found a local max, trying to find the next local min
            if value <= state['pmin']:
                # update if value is smaller
                state['pmin'] = value
                state['pmin_ind'] = ind
            elif value >= state['lmax']:
                # this is where just after a deep dip
                # confirm the last point is a local min
                state['lmin'] = state['pmin']
                local_min_inds.append(state['pmin_ind'])
                local_mins.append(state['pmin'])
                # so for a moment no potential min
                state['pmin'] = None
                state['pmin_ind'] = None
                # the current point becomes a potential max
                state['pmax'] = value
                state['pmax_ind'] = ind
            else:
                # larger than the last point, so this is a potential max
                state['lmax'] = None
                state['pmax'] = value
                state['pmax_ind'] = ind
        else:
            print('strange')
    return local_min_inds, local_mins, local_max_inds, local_maxs
