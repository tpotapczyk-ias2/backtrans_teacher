#!python

# Standard Library
import imp
import os
import subprocess

import pandas as pd
import numpy as np
import youtokentome as yttm
from collections import defaultdict

ENV = os.environ

config = imp.load_source('config', 'config.py')

data = defaultdict(lambda: defaultdict(dict))


#define paths for main files
for name in config.names:
    data['full'][name]['raw']= f'data/full_{name}.txt'
    data['full'][name]['base_tokenized'] = f'data/full_{name}_pre_tokenized.txt'
    data['full'][name]['bpe_tokenized'] = f'data/full_{name}_tokenized.txt'

    data[name]['bpe_train_data'] = f'data/BPE_{name}_train_data.txt'
    data[name]['bpe_model'] = f'data/BPE_{name}_model.model'
    
    for subset in config.subsets:
        data[subset][name]['tokenized'] = f'data/{name}_{subset}_tokenized.txt'

    temp_full_for_shuffle = 'data/full_for_shuffle.txt'


############## shuffle coropora
# run initail tokenization of corpora
for name in config.names:
    command_base_tokenize = f"python -m scripts.tokenize -i {data['full'][name]['raw']} -o {data['full'][name]['base_tokenized']}"
    print(command_base_tokenize)
    result = subprocess.check_output(command_base_tokenize, shell=True)

# prepare and read to df the tokenized corpora 
command_shuffle_datast = f"paste {data['full']['src']['base_tokenized']} {data['full']['tgt']['base_tokenized']} >  {temp_full_for_shuffle}"
print(command_shuffle_datast)
result = subprocess.check_output(command_shuffle_datast, shell=True)

full_shufle = pd.read_csv(f'{temp_full_for_shuffle}', sep="\t", header=None, names=['src','tgt'])
full_shufle = full_shufle[full_shufle['src'].apply(lambda x: len(str(x))>1)]
full_shufle = full_shufle[full_shufle['src'].apply(lambda x: len(str(x))<5000)]
full_shufle = full_shufle[full_shufle['src'].apply(lambda x: len(str(x).split())<500)]
full_shufle = full_shufle[full_shufle['tgt'].apply(lambda x: len(str(x))>1)]
full_shufle = full_shufle[full_shufle['tgt'].apply(lambda x: len(str(x))<5000)]
full_shufle = full_shufle[full_shufle['tgt'].apply(lambda x: len(str(x).split())<500)]
#try:
#    assert full_shufle.src.isnull().sum() == 0
#    assert full_shufle.tgt.isnull().sum() == 0
#except AssertionError:
#    print('Houston, we have a problem.')
#    raise

# shuffle the alligned corpora and save shuffled data to respective files
full_shufle = full_shufle.sample(frac=1)
sample_for_BPE = full_shufle.sample(n=config.vocab_text_sample_size)

for name in config.names:    
    np.savetxt(fname = data['full'][name]['base_tokenized']
            , X = full_shufle[name].values
            , fmt = "%s")
    
    # save daa for traninig BPE model
    np.savetxt(fname = data[name]['bpe_train_data'] 
        , X = sample_for_BPE[name].values 
        , fmt = "%s")
    
    # train BPE model
    yttm.BPE.train(data=data[name]['bpe_train_data'], vocab_size=config.vocab_size, model=data[name]['bpe_model'])


    command_BPE_tokenize = f"yttm encode --model {data[name]['bpe_model']} --output_type subword < {data['full'][name]['base_tokenized']} > {data['full'][name]['bpe_tokenized']}"
    print(command_BPE_tokenize)
    result = subprocess.check_output('mkdir backtrans', shell=True)
    subprocess.check_output(command_shuffle_datast, shell=True)
    for subset in config.subsets:

        if subset == 'train':
            command_train_valid = f"head -n -{config.valid_sentences} {data['full'][name]['bpe_tokenized']} > {backtrans/data[subset][name]['tokenized']}"
        elif subset == 'valid':
            command_train_valid = f"tail -n -{config.valid_sentences} {data['full'][name]['bpe_tokenized']} > {backtrans/data[subset][name]['tokenized']}"
        else:
            raise ValueError
        
        print(command_train_valid)
        result = subprocess.check_output(command_train_valid, shell=True)
