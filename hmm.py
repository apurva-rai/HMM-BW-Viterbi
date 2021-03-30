from numba import vectorize, jit
import numpy as np
import pickle
import np.random
from timeit import default_timer as timer
import re

class HiddenMarkovModel:
