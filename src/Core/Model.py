from nltk.util import ngrams
from collections import Counter

class Model:
    def __init__(self, data, vocabulary):
        self.data = data
        self.n_gram_counts_list = []
        self.vocabulary = vocabulary

    def get_suggestions(self, previous_tokens : str, k=1.0, start_with=None):
        for n in range(1, 6):
            n_model_counts = self._count_n_grams_builtin(self.data, n)
            self.n_gram_counts_list.append(n_model_counts)

        previous_tokens = previous_tokens.split()
        count = len(self.n_gram_counts_list)
        suggestions = []
        for i in range(count - 1):
            n_gram_counts = self.n_gram_counts_list[i]
            nplus1_gram_counts = self.n_gram_counts_list[i + 1]
            suggestion = self._auto_complete(previous_tokens, n_gram_counts,
                                       nplus1_gram_counts, self.vocabulary,
                                       k=k, start_with=start_with)
            suggestions.append(suggestion)

        return suggestions

    def _auto_complete(self, previous_tokens, n_gram_counts, nplus1_gram_counts, vocabulary, k=1.0, start_with=None):

        n = len(list(n_gram_counts.keys())[0])
        previous_n_gram = previous_tokens[-n:]
        probabilities = self._probs(previous_n_gram, n_gram_counts, nplus1_gram_counts, vocabulary, k=k)
        suggestion = None
        max_prob = 0
        for word, prob in probabilities.items():
            if start_with != None:
                if not word.startswith(start_with):
                    continue
            if prob > max_prob:
                suggestion = word
                max_prob = prob

        return suggestion, max_prob
    def _probs(self, previous_n_gram, n_gram_counts, nplus1_gram_counts, vocabulary, k=1.0) -> 'dict':
        previous_n_gram = tuple(previous_n_gram)
        vocabulary = vocabulary + ["<e>", "<unk>"]
        vocabulary_size = len(vocabulary)
        probabilities = {}
        for word in vocabulary:
            probability = self._prob_for_single_word(word, previous_n_gram,
                                               n_gram_counts, nplus1_gram_counts,
                                               vocabulary_size, k=k)
            probabilities[word] = probability

        return probabilities
    def _prob_for_single_word(self, word, previous_n_gram, n_gram_counts, nplus1_gram_counts, vocabulary_size,
                             k=1.0) -> 'float':
        previous_n_gram = tuple(previous_n_gram)
        previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0
        denom = previous_n_gram_count + k * vocabulary_size
        nplus1_gram = previous_n_gram + (word,)
        nplus1_gram_count = nplus1_gram_counts[nplus1_gram] if nplus1_gram in nplus1_gram_counts else 0
        num = nplus1_gram_count + k
        prob = num / denom
        return prob

    def _count_n_grams_builtin(self, data, n, start_token="<s>", end_token="<e>") -> dict:
        all_words = [word for sentence in data for word in sentence]
        all_words = [start_token] * (n - 1) + all_words + [end_token]
        n_grams = list(ngrams(all_words, n))
        n_gram_counts = Counter(n_grams)

        return n_gram_counts
