language = 'sl'

names = ('src','tgt')
subsets = ('train', 'valid')

valid_sentences = 2000

log_file_level = 'INFO'

vocab_size = 2 * 75000
training_opts = {
    'train_from': 'data/checkpoints/checkpoint_step_8000.pt',
    'rnn_size': 128,
    'rnn_type': 'LSTM',
    'dropout': 0.1,
    'save_checkpoint_steps': 2000,
    'train_steps': 300000,
    'batch_type': 'tokens',
    'batch_size': 2000,
    'layers': 2,
    'word_vec_size': 128,
    'optim': 'adam',
    'learning_rate': 0.001
}

training_string = ' '.join([f'--{k} {v}' for k, v in training_opts.items()])
