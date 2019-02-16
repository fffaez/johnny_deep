"""A minimal toy framework for (not very) deep learning."""

# the library was highly influenced (with some pieces of code copied and pasted from):
# https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

__version__ = '0.0.1'
__author__ = 'Marco Faedo <marcofaedo@gmail.com>'
__all__ = ['models', 'activations', 'utils']

from . import models
from . import activations
from . import utils
