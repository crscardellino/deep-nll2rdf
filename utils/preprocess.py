# -*- coding: utf-8 -*-

import re
import os
from corpus import *
from glob import glob
from utils import *


def parse_nll2rdf_conll_document(document_path):
    sentences = []
    words = []
    odrl_action = 0

    with open(document_path, "r") as input_file:
        for line in input_file.readlines():
            if re.match(r'^\s*$', line.strip()):
                sentences.append(CoNLLSentence(words))
                words = []

            elif re.match(r'^#id-[0-9]+$', line.strip()):
                continue

            else:
                token_data = line.split()

                if re.sub(r'\-RRB\-|\-LRB\-|\-LSB\-|\-RSB\-|[^a-zA-Z]', '', token_data[1]).strip() == '':
                    continue
                elif len(token_data) == 10:
                    label = None
                elif ODRL_CLASSES_MAPPINGS[token_data[10]] == 1:
                    label = NLL2RDF_CLASSES_MAPPINGS[(1, 0)]
                else:
                    odrl_action = ODRL_ACTIONS_MAPPINGS[token_data[11]] if len(token_data) == 12 else odrl_action

                    nll2rdf_class = (
                        ODRL_CLASSES_MAPPINGS[token_data[10]],
                        odrl_action
                    )

                    label = NLL2RDF_CLASSES_MAPPINGS[nll2rdf_class]

                for token in re.sub(r'[^a-zA-Z]', ' ', token_data[1]).split():
                    words.append(
                        WordInstance(
                            word=token.strip(),
                            label=label,
                            tag=token_data[3],
                            dependency=token_data[7],
                        )
                    )

    return CoNLLDocument(sentences)


class NLL2RDFCorpus(object):
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.documents = []
        self.parse_directory()

    def __iter__(self):
        for document in self.documents:
            yield document

    def parse_directory(self):
        documents_paths = [y for x in os.walk(self.directory_path) for y in glob(os.path.join(x[0], '*.conll'))]

        for document_path in documents_paths:
            self.documents.append(parse_nll2rdf_conll_document(document_path))

    def get_class_corpus(self, class_number, window_size=2):
        for document in self:
            for sentence in document:
                for instance, label in sentence.get_class_corpus(class_number, window_size):
                    yield instance, label
