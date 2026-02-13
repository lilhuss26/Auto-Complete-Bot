from src.Core.Preprocess import Preprocess
from src.Core.DataSplit import DataSplit
from src.Core.NLPClass import NLPClass
from src.Core.Model import Model

class Pipeline:
    def __init__(self,path : str = r'data/en_US.twitter.txt'):
        self.path = path
        self.preprocess = Preprocess(self.path)
        self.preprocess.read_file()
        tokenized_sentences = self.preprocess.sentences_to_tokens()
        self.datasplit = DataSplit(tokenized_sentences)
        self.train, self.test = self.datasplit.split_data()
        self.nlp = NLPClass(self.train, self.test)
        self.vocabulary = self.nlp.handling_oov()
        self.final_train = self.nlp.unk_tokenize_train()
        self.final_test = self.nlp.unk_tokenize_test()
        self.model = Model(self.final_train, self.vocabulary)

    def get_suggestions(self, previous_tokens : str, k=1.0, start_with=None):
        return self.model.get_suggestions(previous_tokens, k, start_with)