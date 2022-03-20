# config.py

config_seq2gmm = {}
config_seq2gmm['input_dim'] = 1

# Temporal Segmentation
config_seq2gmm['split_frag_num'] = 3

# seq2seq
config_seq2gmm['PAD'] = 0
config_seq2gmm['EOS'] = 1
config_seq2gmm['vocab_size'] = 1
config_seq2gmm['input_embedding_size'] = 1
config_seq2gmm['encoder_hidden_units'] = 15
config_seq2gmm['decoder_hidden_units'] = config_seq2gmm['encoder_hidden_units']

# Estimator
config_seq2gmm['num_mixture'] = 1
config_seq2gmm['es_FC_1'] = 10
config_seq2gmm['keep_prob'] = 0.5

# GMM
config_seq2gmm['GMM_input_dim'] = config_seq2gmm['encoder_hidden_units'] + 2
config_seq2gmm['num_dynamic_dim'] = config_seq2gmm['encoder_hidden_units']
config_seq2gmm['lambda_1'] = 0.0001 #0:0001

# Saver
config_seq2gmm['max_to_keep'] = None # the maximum number of recent checkpoint files to keep. As new files are created, older files are deleted.
config_seq2gmm['epoch_num'] = 2501
config_seq2gmm['save_every_epoch'] = 10
config_seq2gmm['learning_rate'] = 1e-3

config_seq2gmm['is_load_segnum_from_config'] = True # if load parameter 'split_frag_num' from this config file
config_seq2gmm['dataset_dir'] = 'UCRArchive_2018'
config_seq2gmm['dataset_name'] = 'TwoLeadECG'
config_seq2gmm['train_file'] = 'UCRArchive_2018/TwoLeadECG/TwoLeadECG_TRAIN.tsv'
config_seq2gmm['test_file'] = 'UCRArchive_2018/TwoLeadECG/TwoLeadECG_TEST.tsv'
config_seq2gmm['work_dir'] = 'checkpoint'
config_seq2gmm['log_dir'] = 'log'

