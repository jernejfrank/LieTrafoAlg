import numpy as np

import sympy
from sympy import Symbol,sqrt

import matplotlib.pyplot as plt


import tensorflow as tf

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential


from keras.layers import Dense, LSTM
import sys
sys.path.insert(0,'..')

from DeepLie_mods import *

from LieOperator import *
from Poisson import PoissonBracket
# 
import Factorization
from Factorization import factorization
from Factorization import taylor_to_weight_mat
from Factorization import taylorize


