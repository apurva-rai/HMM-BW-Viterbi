from hmm import HiddenMarkovModel

#Skip unnecessary delimiters in input
def textCleaner(data):
    line = ''.join(i for x in line if _ not in string.punctutation).lower()
    line = line.encode("utf8").decode('ascii, 'ignore')

    return line

if __name__ == '__main__':
    data = pd.read_csv("Shakespeare_data.csv")

    data.dropna(axis = 'columns', how = 'any', inpllace = True)

    text = [_ for _ in data['PlayerLine']]

    corpus = [textCleaner(i) for i in text]

    hmm_model = HiddenMarkovModel(hiddenStates = 4)
    hmm_model.trainer(corpus,fileName='model')

    #trans,ems,initials = HiddenMarkovModel.load('model.pickle')

    #options = None

    print("Model complete")
