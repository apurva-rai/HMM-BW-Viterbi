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

        self.rList = []
        self.xList = []

        if transProbs == None:
            self.transProbs = np.ones((self.hiddenStates, self.hiddenStates))
            self.transProbs = self.transProbs + np.random.uniform(size=(self.hiddenStates,self.hiddenStates))
            self.transProbs = self.transProbs/self.transProbs.sum(axis=1,keepdims=True)

        if initialProbs == None:
            self.initialProbs = np.ones((self.hiddenStates, self.hiddenStates))
            self.initialProbs = self.transProbs + np.random.uniform(size=(self.hiddenStates,self.hiddenStates))
            self.initialProbs = self.transProbs/self.transProbs.sum(axis=1,keepdims=True)
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

            self.a[:,i+1] = normalizer(self.a[:,i+1])

    #The backward part of the BW algorithm. Calculates the backward probabilites
    def Baum_Welch_Backward(self,datum):
        self.b = np.zeros((self.hiddenStates,len(datum)))
        self.b[:,len(datum)-1] = 1

        for i in range(len(datum)-2, -1, -1):   #Going in reverse
            for j in range(self.hiddenStates):
                for k in range(self.hiddenStates):
                    self.b[j][i] = self.b[j][i] + self.b[k][i+1] * (self.transProbs[j][k] * self.emissionProbs[k][datum[i+1]])

    #Used to calculate the temporary conditional probabilites in the current state
    def currentProbability(self,datum):
        self.r = np.zeros((self.hiddenStates, len(datum)))
        self.x = np.zeros((self.hiddenStates, len(datum)))

        for i in range(len(datum)):
            sumR = 0
            sumX = 0
            for j in range(self.hiddenStates):
                self.r[j][i] = self.a[j][i] * self.b[j][i]
                sumR = sumR + self.r[j][i]
                if(i != len(datum) - 1):
                    for k in range(self.hiddenStates):
                        self.x[j][k][i] = self.a[j][i] * self.transProbs[j][k] * self.b[k][i+1] * self.emissionProbs[k][datum[i+1]]
                        sumX = sumX + self.x[j][k][i]

            self.r[:,i] = self.r[:,i] / sumR if (sumR != 0) else np.zeros(self.r[:,i].shape)
            if(i != len(datum) - 1):
                self.x[:,:,i] = self.x[:,:,i] / sumX if (sumX != 0) else np.zeros(self.x[:,:,i].shape)

        return self.r, self.x

    #The Expectation Maximizatiom part of the BW algorithm
    def Baum_Welch_Expectation(self,datum):
        for i in datum:
            line = re.split(r' +',i)    #Tokenize and parse by whitespace using simple regular expression
            self.Baum_Welch_Forward(line)
            self.Baum_Welch_Backward(line)
            self.r, self.x = self.currentProbability(line)
            self.rList.append(self.r)
            self.xList.append(self.x)

    def Baum_Welch_Maximization(self,datum):
        for i in range(self.hiddenStates):
            pi = 0

            for j in range(len(datum)):
                pi = pi + self.r[j][i][0]

            self.initialProbs[i] = pi

        self.initialProbs = HiddenMarkovModel.normalizer(self.initialProbs)

        for i in range(self.hiddenStates):
            sumR = 0

            for j in range(len(datum)):
                sumR = sumR + np.sum(self.rList[j][i][:-1])

            for j in range(len(datum)):
                sumX = 0

                for k in range(len(datum)):
                    sumX = sumX + np.sum(self.xList[k][i][j][:-1])

                self.transProbs[i][j] = sumX / sumR

            self.transProbs[i] = HiddenMarkovModel.normalizer(self.transProbs[i])

        for i in range(self.hiddenStates):
            pi = self.emissionProbs[0].copy()

            for j in pi:
                pi[j] = 0

            sum = 0

            for j in range(len(datum)):
                line = re.split(r' +',datum[j])
                sum = sum + np.sum(self.rList[j][i])

                for k in range(len(line)):
                    currentToken = line[k]
                    adder = {currentToken: pi[currentToken] + self.rList[j][i][k]}
                    pi.update(adder)

            for j in pi:
                pi[j] = pi[j] / sum

            self.emissionProbs[i].update(pi)

    #Open and save the pickle file in memory to run
    def opener(self,fileName):
        with open(fileName,'wb') as fileIn:
            pickle.dump((self.transProbs,self.emissionProbs,self.initialProbs),fileIn)

    #Builds emission probabilites matrix using a list of unique token which it derives in the first half of the funciton
    def buildMatrix(self,datum):
        tokenList = []

        for line in datum:
            currentToken = re.split(r' +',line)

            for tok in currentToken:
                if tok in tokenList:
                    continue
                tokenList.append(tok)

        temp = np.ones((self.hiddenStates,len(tokenList)))
        temp = temp + np.random.uniform(size=(self.hiddenStates),len(tokenList))
        temp = temp / temp.sum(axis=1,keepdims=True)

        for i in range(self.hiddenStates):
            tempDictionary = {j: k for j, k in zip(tokenList, temp[i])}
            self.emissionProbs.append(tempDictionary)
