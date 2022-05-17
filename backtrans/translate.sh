#!/bin/bash
nmt_translate --model data/average400000 --beam_size 4 --batch_type tokens --batch_size 32000 --replace_unk --src data/src_train_tokenized.txt --output data/backtranslated

