# -*- coding: utf-8 -*-

import re
from conll import *
from utils import *


def parse_conll_document(document_path):
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

                if len(token_data) == 10:
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

                words.append(
                    WordInstance(
                        word=token_data[1],
                        label=label,
                        tag=token_data[3],
                        dependency=token_data[7],
                        head=token_data[6],
                        position=token_data[0]
                    )
                )

    return CoNLLDocument(sentences)
