from numba import vectorize, jit
import numpy as np
import pickle
from timeit import default_timer as timer
import re
import pandas as pd

class HiddenMarkovModel:

    '''
    Constructor for the Hidden Markov Model network. Takes as input the number of hidden states,
    transitition probabilites for the respective states in a two dimensional float array, a dictionary of emission probabilites
    and, an numpy array of initial probabilites.
    '''
    def __init__(self,hiddenStates,transProbs=None,emissionProbs=None,initialProbs=None):
        self.hiddenStates = hiddenStates
        self.transProbs = transProbs
        self.emissionProbs = emissionProbs
        self.initialProbs = initialProbs

        self.condProbs = None
        self.currentProbs = None
        self.iterations = 100

        if transProbs == None:
            self.transProbs = np.ones((self.hiddenStates, self.hiddenStates))
            self.transProbs = self.transProbs + np.random.uniform(size=(self.hiddenStates,self.hiddenStates))
            self.transProbs = self.transProbs/self.transProbs.sum(axis=1,keepdims=true)

        if initialProbs == None:
            self.initialProbs = np.ones((self.hiddenStates, self.hiddenStates))
            self.initialProbs = self.transProbs + np.random.uniform(size=(self.hiddenStates,self.hiddenStates))
            self.initialProbs = self.transProbs/self.transProbs.sum(axis=1,keepdims=true)
            self.initialProbs = self.initialProbs[np.random.randint(0,self.hiddenStates-1)]

        if emissionProbs == None:
            self.emissionProbs = []
