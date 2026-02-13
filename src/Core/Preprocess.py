import re
import nltk

class Preprocess:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = None
        self.tokenized = None

    def read_file(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content_b = file.read()
            self.content = re.sub(r'[^a-zA-Z\s]', '', content_b)
        return self.content

    def sentences_to_tokens(self) -> 'list':

        sentences = self.content.split('\n')
        sentences = [s.strip() for s in sentences]
        sentences = [s for s in sentences if len(s) > 0]

        for sentence in sentences:
            sentence = sentence.lower()
            token = nltk.word_tokenize(sentence)
            self.tokenized.append(token)

        return self.tokenized