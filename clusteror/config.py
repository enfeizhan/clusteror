import re
from collections import namedtuple
from contextlib import contextmanager
import warnings

DeprecatedOption = namedtuple('DeprecatedOption', 'key msg rkey removal_ver')
RegisteredOption = namedtuple(
    'RegisteredOption',
    'key msg rkey removal_ver'
)

_deprecated_options = {}  # holds deprecated option metdata
_registered_options = {}  # holds registered option metdata
_global_config = {}  # holds the current values for registered options
_reserved_keys = ['all']  # keys which have a special meaning


class OptionError(AttributeError, KeyError):
    """Exception for pandas.options, backwards compatible with KeyError
    checks
    """


def _select_options(pat):
    """returns a list of keys matching `pat`

    if pat=="all", returns all registered options
    """

    # short-circuit for exact key
    if pat in _registered_options:
        return [pat]

    # else look through all of them
    keys = sorted(_registered_options.keys())
    if pat == 'all':  # reserved key
        return keys

    return [k for k in keys if re.search(pat, k, re.I)]


def _get_deprecated_option(key):
    """
    Retrieves the metadata for a deprecated option, if `key` is deprecated.

    Returns
    -------
    DeprecatedOption (namedtuple) if key is deprecated, None otherwise
    """

    try:
        d = _deprecated_options[key]
    except KeyError:
        return None
    else:
        return d


def _warn_if_deprecated(key):
    """
    Checks if `key` is a deprecated option and if so, prints a warning.

    Returns
    -------
    bool - True if `key` is deprecated, False otherwise.
    """

    d = _get_deprecated_option(key)
    if d:
        if d.msg:
            print(d.msg)
            warnings.warn(d.msg, DeprecationWarning)
        else:
            msg = "'%s' is deprecated" % key
            if d.removal_ver:
                msg += ' and will be removed in %s' % d.removal_ver
            if d.rkey:
                msg += ", please use '%s' instead." % d.rkey
            else:
                msg += ', please refrain from using it.'

            warnings.warn(msg, DeprecationWarning)
        return True
    return False


def _translate_key(key):
    """
    if key is deprecated and a replacement key defined, will return the
    replacement key, otherwise returns `key` as - is
    """

    d = _get_deprecated_option(key)
    if d:
        return d.rkey or key
    else:
        return key


def _get_single_key(pat, silent):
    keys = _select_options(pat)
    if len(keys) == 0:
        if not silent:
            _warn_if_deprecated(pat)
        raise OptionError('No such keys(s): %r' % pat)
    if len(keys) > 1:
        raise OptionError('Pattern matched multiple keys')
    key = keys[0]
    if not silent:
        _warn_if_deprecated(key)
    key = _translate_key(key)
    return key


def _get_root(key):
    path = key.split('.')
    cursor = _global_config
    for p in path[:-1]:
        cursor = cursor[p]
    return cursor, path[-1]


def _get_option(pat, silent=False):
    key = _get_single_key(pat, silent)

    # walk the nested dict
    root, k = _get_root(key)
    return root[k]


def _get_registered_option(key):
    """
    Retrieves the option metadata if `key` is a registered option.

    Returns
    -------
    RegisteredOption (namedtuple) if key is deprecated, None otherwise
    """
    return _registered_options.get(key)


def _set_option(*args, **kwargs):
    # must at least 1 arg deal with constraints later
    nargs = len(args)
    if not nargs or nargs % 2 != 0:
        raise ValueError("Must provide an even number of non-keyword "
                         "arguments")

    # default to false
    silent = kwargs.pop('silent', False)

    if kwargs:
        raise TypeError('_set_option() got an unexpected keyword '
                        'argument "{0}"'.format(list(kwargs.keys())[0]))

    for k, v in zip(args[::2], args[1::2]):
        key = _get_single_key(k, silent)

        o = _get_registered_option(key)
        if o and o.validator:
            o.validator(v)

        # walk the nested dict
        root, k = _get_root(key)
        root[k] = v

        if o.cb:
            if silent:
                with warnings.catch_warnings(record=True):
                    o.cb(key)
            else:
                o.cb(key)


def _build_option_description(k):
    """ Builds a formatted description of a registered option and prints it """

    o = _get_registered_option(k)
    d = _get_deprecated_option(k)

    # s = u('%s ') % k
    s = '%s ' % k

    if o.doc:
        s += '\n'.join(o.doc.strip().split('\n'))
    else:
        s += 'No description available.'

    if o:
        # s += u('\n    [default: %s] [currently: %s]') % (o.defval,
        #                                                _get_option(k, True))
        s += '\n    [default: %s] [currently: %s]' % (
            o.defval,
            _get_option(k, True)
        )

    if d:
        # s += u('\n    (Deprecated')
        s += '\n    (Deprecated'
        # s += (u(', use `%s` instead.') % d.rkey if d.rkey else '')
        s += (', use `%s` instead.' % d.rkey if d.rkey else '')
        # s += u(')')
        s += ')'

    s += '\n\n'
    return s


def _describe_option(pat='', _print_desc=True):

    keys = _select_options(pat)
    if len(keys) == 0:
        raise OptionError('No such keys(s)')

    s = u('')
    for k in keys:  # filter by pat
        s += _build_option_description(k)

    if _print_desc:
        print(s)
    else:
        return s
