__version__ = '3.1.1'

print('using donkey v{} ...'.format(__version__))

import sys
print('\n'.join(sys.path))

if sys.version_info.major < 3:
    msg = 'Donkey Requires Python 3.4 or greater. You are using {}'.format(sys.version)
    raise ValueError(msg)

from . import parts
from .vehicle import Vehicle
from .memory import Memory
from . import utils
from . import config
from . import contrib
from .config import load_config

import tensorflow as tf
tf_config=tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)
