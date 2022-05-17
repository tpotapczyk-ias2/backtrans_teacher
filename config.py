language = 'hu'

names = ('src', 'tgt')
subsets = ('train', 'valid')

valid_sentences = 2000

log_file_level = 'INFO'
vocab_size = 32000
vocab_text_sample_size=200000
valid_sentences = 2000
vocab_size = 32000
training_opts = {
    #'train_from': 'data/checkpoints/checkpoint_step_96000.pt',
    'batch_size': 6148,
    'batch_type': 'tokens',
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'optim': 'adam',
    'encoder_type': 'transformer',
    'decoder_type': 'transformer',
    'position_encoding': '',
    'enc_layers': 4,
    'dec_layers': 4,
    'heads': 8,
    'rnn_size': 512, 
    'word_vec_size': 512,
    'transformer_ff': 2048,
    'model_dtype': "fp32",
    'learning_rate': 1,
    'warmup_steps': 2000,
    'decay_method': "noam",
    'adam_beta1': 0.9,
    'adam_beta2': 0.997,
    'max_grad_norm': 0,
    #label_smoothing: 0.1
    'param_init': 0,
    'param_init_glorot': '',
    'normalization': "tokens",
    'save_checkpoint_steps': 2000,
    'train_steps': 300000,
    'valid_batch_size': 8,
    'valid_steps': 2000
}

training_string = ' '.join([f'--{k} {v}' for k, v in training_opts.items()])
