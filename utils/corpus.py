# -*- coding: utf-8 -*-

import os
import re
from nltk.corpus import stopwords
from utils import NLL2RDF_CLASSES


class WordInstance(object):
    def __init__(self, word, label=None, tag=None, dependency=None, head=None, position=None):
        self.word = word
        self.label = label
        self.tag = tag
        self.dependency = dependency
        self.head = head
        self.position = position

    def get_word_lemma(self):
        return self.word.lower()  # TODO: Use a lemmatizer?

    def __str__(self):
        if self.label is not None:
            return "%s\t%s" % (self.word, NLL2RDF_CLASSES[self.label])
        else:
            return "%s\t%s\t%s\t%s\t%s" % (self.position, self.word, self.tag, self.dependency, self.head)

    def __repr__(self):
        return str(self)


class CoNLLSentence(object):
    def __init__(self, words):
        self.words = words

    def __getitem__(self, item):
        return self.words[item]

    def get_head_of_word(self, position):
        if self[position].head == 0:
            return "root"
        else:
            return self[self[position].head - 1]

    def get_window(self, position, window_size=5):
        ws = int(window_size / 2)

        return self.words[position-ws:position+ws+1]

    def get_raw_sentence(self):
        return " ".join([word.word for word in self.words])

    def __str__(self):
        return "\n".join([str(word) for word in self.words])

    def __repr__(self):
        return str(self)


class CoNLLDocument(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, item):
        return self.sentences[item]

    def __str__(self):
        return "\n\n".join([str(sentence) for sentence in self.sentences])

    def __repr__(self):
        return str(self)


class UntaggedCorpusIterator(object):
    def __init__(self, corpus_directory, remove_numbers=False, remove_stop_words=False):
        self.corpus_directory = corpus_directory
        self.remove_stop_words = remove_stop_words
        self.remove_numbers = remove_numbers

    def __iter__(self):
        for fname in os.listdir(self.corpus_directory):
            with open(os.path.join(self.corpus_directory, fname), "r") as f:
                for line in f:
                    if line.strip() == "":
                        continue
                    else:
                        yield self.preprocess(line.strip())

    def preprocess(self, line):
        if self.remove_numbers:
            line = re.sub(r"[^a-zA-Z]", ' ', line)
        else:
            line = re.sub(r"[^a-zA-Z0-9]", ' ', line)

        line = line.lower()

        if self.remove_stop_words:
            return [w for w in line.split() if w not in set(stopwords.words("english"))]
        else:
            return line.split()
