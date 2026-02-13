
class NLPClass:
    def __init__(self, train_data, test_data, count_threshold = 50):
        self.train_data = train_data
        self.test_data = test_data
        self.count_threshold = count_threshold
        self.word_counts = {}
        self.vocabulary = []

    def _count_the_words(self) -> 'dict':
        for sentence in self.train_data:
            for token in sentence:
                if token not in self.word_counts.keys():
                    self.word_counts[token] = 1
                else:
                    self.word_counts[token] += 1
        return self.word_counts

    def handling_oov(self) -> 'list':
        self._count_the_words()
        for word, count in self.word_counts.items():
            if count >= self.count_threshold:
                self.vocabulary.append(word)
        return self.vocabulary

    def unk_tokenize_train(self):
        self._unk_tokenize(self.train_data)

    def unk_tokenize_test(self):
        self._unk_tokenize(self.test_data)

    def _unk_tokenize(self,data, unknown_token="<unk>") -> 'list':
        vocabulary = set(self.vocabulary)
        new_tokenized_sentences = []

        for sentence in data:
            new_sentence = []
            for token in sentence:
                if token in vocabulary:
                    new_sentence.append(token)
                else:
                    new_sentence.append(unknown_token)
            new_tokenized_sentences.append(new_sentence)

        return new_tokenized_sentences
