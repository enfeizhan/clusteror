# import ipdb
import os
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

config = configparser.ConfigParser()
path_list = list(os.path.split(__file__))
path_list = path_list[:-2]
path_list.append('setup.cfg')
path_tuple = tuple(path_list)
cfg_path = os.path.join(*path_tuple)
config.read(cfg_path)
# ipdb.set_trace()
numpy_random_seed = int(config['RANDOMNESS']['NumpyRandomSeed'])
theano_random_seed = int(config['RANDOMNESS']['TheanoRandomSeed'])
decimal_places = int(config['TEST']['DecimalPlaces'])
