#!/bin/bash

mv data/src_train_tokenized.txt backtrans/data/tgt_train_tokenized.txt
mv data/src_valid_tokenized.txt backtrans/data/tgt_valid_tokenized.txt
mv data/tgt_train_tokenized.txt backtrans/data/src_train_tokenized.txt
mv data/tgt_valid_tokenized.txt backtrans/data/src_valid_tokenized.txt
