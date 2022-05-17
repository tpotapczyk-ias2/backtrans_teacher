import pandas as pd

full_shufle = pd.read_csv(f'src-en.tsv', sep="\t", header=None, names=['src','tgt'])
full_shufle = full_shufle[full_shufle['src'].apply(lambda x: len(str(x))>1)]
full_shufle = full_shufle[full_shufle['src'].apply(lambda x: len(str(x))<5000)]
full_shufle = full_shufle[full_shufle['src'].apply(lambda x: len(str(x).split())<500)]
full_shufle = full_shufle[full_shufle['tgt'].apply(lambda x: len(str(x))>1)]
full_shufle = full_shufle[full_shufle['tgt'].apply(lambda x: len(str(x))<5000)]
full_shufle = full_shufle[full_shufle['tgt'].apply(lambda x: len(str(x).split())<500)]

full_shufle = full_shufle.sample(frac=1)

for name in config.names:    
        np.savetxt(fname = data['full'][name]['base_tokenized']
                            , X = full_shufle[name].values
                            , fmt = "%s")
            
        # save daa for traninig BPE model
        np.savetxt(fname = data[name]['bpe_train_data'] 
                                , X = sample_for_BPE[name].values 
                                , fmt = "%s")
