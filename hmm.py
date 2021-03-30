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

        self.a = None
        self.b = None
        self.r = None
        self.x = None
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

    #Used to normalize the data set in the forward-backward algorithms
    @staticmethod
    def normalizer(datum):
        return [float(x)/np.sum(datum) for x in datum]

    #The forward part of the BW algorithm. This calculates the forward probabilites
    def Baum_Welch_Forward(self,datum):
        self.a = np.zeros((self.hiddenStates,len(datum)))

        for i in range(self.hiddenStates):
            self.a[i][0] = self.initialProbs[i] * self.emissionProbs[i][datum[0]]

        self.a[:,0] = normalizer(self.a[:,0])

        for i in range(len(datum)-1):
            for j in range(self.hiddenStates):
                sum = 0

                for k in range(self.hiddenStates):
                    sum = sum + self.a[k][i] * self.transProbs[k][j]

                self.a[j][i+1] = self.emissionProbs[j][datum[i+1]] * sigma

            self.a[:,i+1] = noramlizer(self.a[:,i+1])
