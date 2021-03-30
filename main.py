from hmm import HiddenMarkovModel
from pandas import read_csv
import string
#Skip unnecessary delimiters in input
def textCleaner(data):
    data = "".join(_ for _ in data if _ not in string.punctuation).lower()
    data = data.encode("utf8").decode('ascii', 'ignore')

    return data

if __name__ == '__main__':
    data = read_csv("Shakespeare_data.csv")

    data.dropna(axis = 'columns', how = 'any', inplace = True)

    text = [_ for _ in data['PlayerLine']]

    corpus = [textCleaner(i) for i in text]
    '''
    Used to train model-
    hmm_model = HiddenMarkovModel(hiddenStates = 5)
    hmm_model.trainer(corpus,filName='model.pickle')
    print("Model complete")
    '''

    trans,ems,initials = HiddenMarkovModel.load('model.pickle')

    currentModel = HiddenMarkovModel(hiddenStates = 5 ,transProbs = trans, emissionProbs = ems, initialProbs = initials)

    print('1- Generate text\n2- Predict text')

    inNum = 0
    inNum = input()

    if inNum == '1':
        print("Number of words to be generated: ")
        currentModel.generator(int(input()))

    elif inNum == '2':
        print("Input sequences of words to predict on: ")
        textIn = str(input())

        print("Number of words to be predicted: ")
        currentModel.Viterbi(textIn,int(input()))

    else:
        print("Improper input. Exiting...")
