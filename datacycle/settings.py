try:
    import configparser
except ImportError:
    import ConfigParser as configparser

config = configparser.ConfigParser()
config.read('setup.cfg')
numpy_random_seed = config['RANDOMNESS']['NumpyRandomSeed']
theano_random_seed = config['RANDOMNESS']['TheanoRandomSeed']
