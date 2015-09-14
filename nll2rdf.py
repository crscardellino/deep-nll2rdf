#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import shutil
import sys
import utils
from gensim.models import Word2Vec
from nn import NNPipeline
from utils.preprocess import NLL2RDFCorpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word2Vec algorithm")
    parser.add_argument("train", type=str, metavar="CORPUS_DIRECTORY",
                        help="Path to the corpus directory (CoNLL format).")
    parser.add_argument("model", type=str, metavar="MODEL", help="Path to Word2Vec model (.bin file).")
    parser.add_argument("--output_dir", type=str, metavar="OUTPUTDIR", help="Path to the output directory.",
                        default="results")
    parser.add_argument("--layers", type=int, nargs='+', metavar="LAYERS", help="Size of the layers (and number)",
                        default=[1500, 1200, 600, 300])
    parser.add_argument("--window", type=int, metavar="WINDOW", help="Size of the window to classify.", default=2)
    parser.add_argument("--activation", type=str, metavar="ACTIVATION_FUNCTION", help="Activation function type",
                        default='sigmoid')
    parser.add_argument("--epochs", type=int, metavar="EPOCHS", help="Number of epochs", default=1)
    parser.add_argument("--kfolds", type=int, metavar="KFOLDS", help="Number of KFolds (0 to deactivate)", default=3)
    parser.add_argument("--test_split", type=float, metavar="TEST_SPLIT",
                        help="Test split ratio. Only use if kfolds = 0.", default=0.0)

    args = parser.parse_args()

    print >> sys.stderr, "Parsing nll2rdf corpus"
    nll2rdf_corpus = NLL2RDFCorpus(args.train)

    print >> sys.stderr, "Creating output directories (will erase any existing data)"
    try:
        os.mkdir(args.output_dir)
    except OSError:
        shutil.rmtree(args.output_dir)
        os.mkdir(args.output_dir)

    print >> sys.stderr, "Loading word2vec model"
    word2vec = Word2Vec.load_word2vec_format(args.model, binary=True)

    # We have one model for each class
    print >> sys.stderr, "Training and testing the models"
    for class_number, class_name in utils.NLL2RDF_CLASSES:
        if class_number == 0:  # Ignore the NO-CLASS
            continue

        print >> sys.stderr, "Setting up training for class {}".format(class_name.lower())

        X = []
        y = []

        class_directory = os.path.join(args.output_dir, class_name.lower())
        os.mkdir(class_directory)

        for window, label in nll2rdf_corpus.get_class_corpus(class_name, args.window):
            vectors = []
            for token in window:
                vectors.append(word2vec[token.word])

            X.append(np.hstack(vectors))
            y.append(label)

        X = np.vstack(X)
        y = np.array(y)

        pipeline = NNPipeline(args.layers, args.activation, args.epochs, kfolds=args.kfolds,
                              test_split=args.test_split)

        print >> sys.stderr, "Training and saving model and results"
        pipeline.save_score(X, y, class_directory, True)