from numba import vectorize, jit
import numpy as np
import pickle
from timeit import default_timer as timer
import re
import pandas as pd
import logging
import traceback

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
        self.iterations = 5

        self.rList = []
        self.xList = []

        if transProbs is None:
            self.transProbs = np.ones((self.hiddenStates, self.hiddenStates))
            self.transProbs = self.transProbs + np.random.uniform(size=(self.hiddenStates,self.hiddenStates))
            self.transProbs = self.transProbs/self.transProbs.sum(axis=1,keepdims=True)

        if initialProbs is None:
            self.initialProbs = np.ones((self.hiddenStates, self.hiddenStates))
            self.initialProbs = self.transProbs + np.random.uniform(size=(self.hiddenStates,self.hiddenStates))
            self.initialProbs = self.transProbs/self.transProbs.sum(axis=1,keepdims=True)
            self.initialProbs = self.initialProbs[np.random.randint(0,self.hiddenStates-1)]

        if emissionProbs is None:
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

        self.a[:,0] = HiddenMarkovModel.normalizer(self.a[:,0])

        for i in range(len(datum)-1):
            for j in range(self.hiddenStates):
                sum = 0

                for k in range(self.hiddenStates):
                    sum = sum + self.a[k][i] * self.transProbs[k][j]

                self.a[j][i+1] = self.emissionProbs[j][datum[i+1]] * sum

            self.a[:,i+1] = HiddenMarkovModel.normalizer(self.a[:,i+1])

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
        self.x = np.zeros((self.hiddenStates, self.hiddenStates, len(datum)))

        for k in range(len(datum)):
            sum = 0
            for i in range(self.hiddenStates):
                self.r[i][k] = self.a[i][k] * self.b[i][k]
                sum = sum + self.r[i][k]
            self.r[:, k] = self.r[:, k] / sum if sum != 0 else np.zeros(self.r[:, k].shape)

        for k in range(len(datum)-1):
            sum = 0
            for i in range(self.hiddenStates):
                for j in range(self.hiddenStates):
                    self.x[i][j][k] = self.a[i][k] * self.transProbs[i][j] * self.b[j][k+1] * self.emissionProbs[j][datum[k+1]]
                    sum = sum + self.x[i][j][k]
            self.x[:, :, k] = self.x[:, :, k] / sum if sum != 0 else np.zeros(self.x[:, :, k].shape)

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
                pi = pi + self.rList[j][i][0]

            self.initialProbs[i] = pi

        self.initialProbs = HiddenMarkovModel.normalizer(self.initialProbs)

        for i in range(self.hiddenStates):
            sumR = 0

            for j in range(len(datum)):
                sumR = sumR + np.sum(self.rList[j][i][:-1])

            for j in range(self.hiddenStates):
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

    def trainer(self,datum,filName):
        self.buildMatrix(datum)

        for i in range(self.iterations):
            self.Baum_Welch_Expectation(datum)
            self.Baum_Welch_Maximization(datum)

        self.opener(fileName = filName)

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
        temp = temp + np.random.uniform(size=(self.hiddenStates,len(tokenList)))
        temp = temp / temp.sum(axis=1,keepdims=True)

        for i in range(self.hiddenStates):
            tempDictionary = {j: k for j, k in zip(tokenList, temp[i])}
            self.emissionProbs.append(tempDictionary)

    #Viterbi algorithm used for text predicitons. Takes length of prediction and text to be predicted on as input
    def Viterbi(self,input,n):
        tokens = re.split(r' +',input)
        t_1 = np.zeros((self.hiddenStates,len(tokens)))
        t_2 = np.zeros((self.hiddenStates, len(tokens)))

        for i in range(self.hiddenStates):
            if tokens[0] in self.emissionProbs[i].keys():
                t_1[i][0] = self.initialProbs[i] * self.emissionProbs[i][tokens[0]]
                t_2[i][0] = 0

        for i in range(1, len(tokens)):
            for j in range(self.hiddenStates):
                if tokens[i] in self.emissionProbs[j].keys():
                    t_1[j][i] = np.max(t_1[:,i-1] * self.transProbs[j] * self.emissionProbs[j][tokens[i]])
                    t_2[j][i] = np.argmax(t_1[:,i-1] * self.transProbs[j] * self.emissionProbs[j][tokens[i]])

        chi = np.zeros(len(tokens))
        zeta = np.zeros(len(tokens))

        for i in range(len(tokens)-1,0,-1):
            zeta[i-1] = t_2[int(zeta[i]),i]
            chi[i-1] = zeta[i-1]

        currentState = int(chi[len(tokens)-1])
        soln = ''

        for i in range(n):
            currentState = np.random.choice(range(self.hiddenStates), p = self.transProbs[currentState])
            nextToken = np.random.choice(list(self.emissionProbs[currentState].keys()),p = list(self.emissionProbs[currentState].values()))
            soln = soln + nextToken + ' '

        print(input + ' ' + soln + '\n')

    #Make a string based on the analyzed data set from BW algorithm
    def generator(self,n):
        s = np.random.choice(range(self.hiddenStates), p = self.initialProbs)
        currentState = s
        soln = ''

        for i in range(n):
            currentToken = np.random.choice(list(self.emissionProbs[currentState].keys()), p = list(self.emissionProbs[currentState].values()))
            soln = soln + currentToken + ' '
            currentState = np.random.choice(range(self.hiddenStates), p = self.transProbs[currentState])

        print(soln + '\n')

    @staticmethod
    def load(fileName):
        with open(fileName, 'rb') as fileIn:
            return pickle.load(fileIn)
