# -*- coding: utf-8 -*-

import os
import re
from nltk.corpus import stopwords
from utils import NLL2RDF_CLASSES


class WordInstance(object):
    def __init__(self, word, label=None, tag=None, dependency=None):
        self.word = word
        self.label = label
        self.tag = tag
        self.dependency = dependency

    def __str__(self):
        if self.label is None:
            return self.word
        else:
            return "%s\t%s" % (self.word, NLL2RDF_CLASSES[self.label])

    def __repr__(self):
        return str(self)


class CoNLLSentence(object):
    def __init__(self, words):
        self.words = words

    def __getitem__(self, item):
        return self.words[item]

    def __iter__(self):
        for word in self.words:
            yield word

    def __str__(self):
        return "\n".join([str(word) for word in self.words])

    def __repr__(self):
        return str(self)

    def get_class_corpus(self, class_name, window_size=2):
        for position, token in enumerate(self, start=1):
            window = self.get_window(position, window_size)
            label = 0 if class_name != token.label else 1
            yield window, label

    def get_raw_sentence(self):
        return " ".join([word.word for word in self.words])

    def get_window(self, position, window_size=2):
        start = position-window_size-1
        end = position+window_size

        window = []

        for widx in xrange(start, end):
            if widx < 0:
                window.append(WordInstance("PADDING"))
            elif widx >= len(self.words):
                window.append(WordInstance("PADDING"))
            else:
                window.append(self[widx])

        return window


class CoNLLDocument(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence

    def __getitem__(self, item):
        return self.sentences[item]

    def __str__(self):
        return "\n\n".join([str(sentence) for sentence in self.sentences])

    def __repr__(self):
        return str(self)


class UntaggedCorpusIterator(object):
    def __init__(self, corpus_directory, remove_numbers=True, remove_stop_words=False):
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
