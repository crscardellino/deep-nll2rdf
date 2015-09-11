# -*- coding: utf-8 -*-

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
