#!/bin/bash

cat data/src_train_tokenized.txt data/src_train_tokenized.txt > data/en_teacher.txt
cat data/tgt_train_tokenized.txt data/backtranslated.txt > data/src_teacher.txt

paste data/src_teacher.txt data/en_teacher.txt > src-en.tsv

python shuffle_teacher_corpus.py
