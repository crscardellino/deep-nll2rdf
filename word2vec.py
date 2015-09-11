# -*- coding: utf-8 -*-

import argparse
import os
import sys
from corpus import UntaggedCorpusIterator
from gensim.models import Word2Vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Word2Vec algorithm")
    parser.add_argument("train", type=str, metavar="CORPUS_DIRECTORY", help="Path to the corpus directory.")
    parser.add_argument("--output", type=str, metavar="OUTPUT", help="File to dump the word vectors.",
                        default=os.path.join(os.path.abspath(__file__), "vectors.bin"))
    parser.add_argument("--vocab", type=str, metavar="VOCAB_OUTPUT", help="File to dump the vocabulary",
                        default=os.path.join(os.path.abspath(__file__), "vocabulary"))
    parser.add_argument("--size", type=int, metavar="SIZE", help="Set size of word vectors.",
                        default=100)
    parser.add_argument("--window", type=int, metavar="WINDOW", help="Max skip length between words.",
                        default=10)
    parser.add_argument("--threads", type=int, metavar="THREADS", help="Set the number of threads for parallelizing.",
                        default=12)
    parser.add_argument("--min_count", type=int, metavar="MIN_COUNT",
                        help="Set the minimum number of occurrences for a word", default=5)
    parser.add_argument("--sample", type=float, metavar="SAMPLE",
                        help="Threshold for configuring which higher-frequency words are randomly downsampled",
                        default=1E-5)
    parser.add_argument("--alpha", type=float, metavar="ALPHA", help="Set the starting learning rate", default=0.025)
    parser.add_argument("--iter", type=int, metavar="ITERATIONS", help="number of iterations (epochs) over the corpus.",
                        default=5)
    parser.add_argument("--negative", type=int, metavar="NEGATIVE_SAMPLING",
                        help="If > 0, negative sampling will be used, the int for negative specifies how many " +
                             "\"noise words\" should be drawn (usually between 5-20).", default=5)
    parser.add_argument("--cbow", action="store_true",
                        help="If set it will use the cbow method to generate the vectors")
    parser.add_argument("--cbow_mean", action="store_true",
                        help="If set use the mean of the context word vectors. Else use the sum. " +
                             "Only applies when cbow is used.")
    parser.add_argument("--hs", action="store_true",
                        help="If set hierarchical sampling will be used for model training")
    parser.add_argument("--gensim_save", action="store_true",
                        help="If set, save the word vectors in gensim (pickle) format.")

    args = parser.parse_args()

    sentences = UntaggedCorpusIterator(args.train)

    model_config = {
        "size": args.size,
        "window": args.window,
        "workers": args.threads,
        "min_count": args.min_count,
        "sample": args.sample,
        "alpha": args.alpha,
        "iter": args.iter,
        "negative": None if args.negative == 0 else args.negative,
        "sg": 0 if args.cbow else 1,
        "hs": 1 if args.hs else 0,
        "cbow_mean": 1 if args.cbow_mean else 0
    }

    print >> sys.stderr, "Training Word2Vec model on %s corpus." % sentences.corpus_directory
    model = Word2Vec(sentences, **model_config)

    print >> sys.stderr, "Saving the model in %s." % args.train
    if args.gensim_save:
        model.save(args.output)
    else:
        model.save_word2vec_format(args.output, fvocab=args.vocab, binary=True)
