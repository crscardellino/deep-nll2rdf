# -*- coding: utf-8 -*-

import cPickle as pickle
import os
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder, Dropout
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import NLL2RDF_CLASSES

np.random.seed(1337)  # for reproducibility


class NNPipeline(object):
    def __init__(self, layers, activation, epochs, kfolds=3, test_split=0.0, batch_size=64, classes=2):
        assert bool(kfolds) != bool(test_split)  # Either cross_validation or test_split
        self.layers = layers
        self.activation = activation
        self.epochs = epochs
        self.kfolds = kfolds
        self.test_split = test_split
        self.model = None
        self.batch_size = batch_size
        self.classes = classes

    def _pretraining(self, X):
        """
        Pre-train the NN using stacked denoising autoencoders
        :param X: features
        :param y: labels
        :return: encoders
        """
        encoders = []
        for i, (n_in, n_out) in enumerate(zip(self.layers[:-1], self.layers[1:]), start=1):
            print >> sys.stderr, ('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
            # Create AE and training
            ae = Sequential()
            encoder = containers.Sequential([Dropout(0.3), Dense(n_in, n_out, activation=self.activation)])
            decoder = containers.Sequential([Dense(n_out, n_in, activation=self.activation)])
            ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
            ae.compile(loss='mean_squared_error', optimizer='sgd')
            ae.fit(X, X, batch_size=self.batch_size, nb_epoch=self.epochs)
            # Store trainined weight and update training data
            encoders.append(ae.layers[0].encoder)
            X = ae.predict(X)

        return encoders

    def _fine_tuning(self, X, Y, encoders):
        self.model = Sequential()

        print >> sys.stderr, 'Fine tuning of the neural network (with regularization)'

        for encoder in encoders:
            self.model.add(encoder)

        self.model.add(Dense(self.layers[-1], self.classes, activation='softmax',
                             W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))

        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

        self.model.fit(X, Y, batch_size=self.batch_size, show_accuracy=True, nb_epoch=self.epochs)

    def fit(self, X, y):
        Y = np_utils.to_categorical(y, self.classes)

        encoders = self._pretraining(np.copy(X))
        self._fine_tuning(X, Y, encoders)

    def save_score(self, X, y, directory, save_model=False):
        if self.kfolds > 0:
            accuracies = []
            precision_scores = []
            recall_scores = []
            f1_scores = []

            print >> sys.stderr, "\nStratified {}-Fold Cross-Validation".format(self.kfolds)

            for fold, (train_idx, test_idx) in enumerate(StratifiedKFold(y, self.kfolds, shuffle=True), start=1):
                print >> sys.stderr, "\nFold {}: {} train examples, {} test examples".format(
                    fold, train_idx.shape[0], test_idx.shape[0]
                )
                Xtrain, Xtest = X[train_idx], X[test_idx]
                ytrain, ytest = y[train_idx], y[test_idx]

                self.fit(Xtrain, ytrain)
                ypred = self.model.predict_classes(Xtest)

                accuracies.append(accuracy_score(ytest, ypred))
                prec, rec, f1, cls = precision_recall_fscore_support(ytest, ypred)
                precision_scores.append(prec)
                recall_scores.append(rec)
                f1_scores.append(f1)

            accuracies = np.mean(accuracies)
            precision_scores = np.mean(precision_scores, axis=0)
            recall_scores = np.mean(recall_scores, axis=0)
            f1_scores = np.mean(f1_scores, axis=0)

            with open(os.path.join(directory, "results.txt"), "a") as fobj:
                fobj.write("Stratified {}-Fold Cross-Validation Results\n".format(self.kfolds))
                fobj.write("Accuracy: {:.2f}\n".format(accuracies))

                fobj.write("Class\tPrec\tRec\tF1\n")
                for i in xrange(precision_scores.shape[0]):
                    fobj.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
                        NLL2RDF_CLASSES[i], precision_scores[i], recall_scores[i], f1_scores[i]
                    ))
        else:
            sys.stderr.write("\nTest {} Split: ".format(self.test_split))
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=self.test_split)
            print >> sys.stderr, "{} train examples, {} test examples".format(Xtrain.shape[0], Xtest.shape[0])

            self.fit(Xtrain, ytrain)
            ypred = self.model.predict_classes(Xtest)

            accuracy = accuracy_score(ytest, ypred)
            precision, recall, fscore, cls = precision_recall_fscore_support(ytest, ypred)

            with open(os.path.join(directory, "results.txt"), "a") as fobj:
                fobj.write("Test {} Split Results\n".format(self.test_split))
                fobj.write("Accuracy: {:.2f}\n".format(accuracy))

                fobj.write("Class\tPrec\tRec\tF1\n")
                for i in xrange(precision.shape[0]):
                    fobj.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
                        NLL2RDF_CLASSES[i], precision[i], recall[i], fscore[i])
                    )

        if save_model:
            self.fit(X, y)
            self.save_model(os.path.join(directory, "model.p"))

    def save_model(self, file_path):
        with open(file_path, "wb") as fobj:
            pickle.dump(self.model, fobj)
