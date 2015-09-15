#!/usr/bin/env bash

python word2vec.py corpus/unannotated/licenses-raw-format/ --size 200 --window 10 --min_count 1 --output corpus/licenses.sg.neg.10.bin --vocab corpus/licenses.sg.neg.10.voc
python word2vec.py corpus/unannotated/licenses-raw-format/ --size 200 --window 5 --min_count 1 --output corpus/licenses.sg.neg.5.bin --vocab corpus/licenses.sg.neg.5.voc
# python word2vec.py corpus/unannotated/licenses-raw-format/ --size 200 --window 10 --min_count 1 --output corpus/licenses.cbow.10.bin --vocab corpus/licenses.cbow.10.voc --hs --cbow
# python word2vec.py corpus/unannotated/licenses-raw-format/ --size 200 --window 5 --min_count 1 --output corpus/licenses.cbow.5.bin --vocab corpus/licenses.cbow.5.voc --hs --cbow

