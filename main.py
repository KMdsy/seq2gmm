# main.py
from config import config_seq2gmm
from Seq2GMM import Seq2GMM
from utils import calculate_score
from utils import read_dataset
from new_utils import SplitData
import tensorflow as tf
import numpy as np
import copy
import os


log_dir = config_seq2gmm['log_dir']
if os.path.exists(log_dir) == False: os.makedirs(log_dir)
log_file = log_dir + '/' + config_seq2gmm['dataset_name'] + '.txt'
if not os.path.exists(log_file):
    f = open(log_file, 'w+')
    f.close()

if __name__ == "__main__":
    
    # print config
    for key, value in config_seq2gmm.items():
        print('{} = {}'.format(key, value))

    # load data
    data_handler = SplitData()
    if config_seq2gmm['is_load_segnum_from_config'] is True: 
        split_num = config_seq2gmm['split_frag_num']
    else:
        split_num = None # learn split_num using greedy algorithm

    data, label, fragments_num, split_num = data_handler.read_data(os.path.join(config_seq2gmm['dataset_dir'], config_seq2gmm['dataset_name']), 'train')
    t_data, t_label, t_fragments_num, _ = data_handler.read_data(os.path.join(config_seq2gmm['dataset_dir'], config_seq2gmm['dataset_name']), 'test', split_num)
    
    # train
    tf.reset_default_graph()
    print('Initializing Seq2gmm model')
    seq2gmm = Seq2GMM(config_seq2gmm, log_file)
    print('Start training')
    fullAUCs, _, _, _, fullAUPRs = seq2gmm.train(data, t_label, t_data, fragments_num, t_fragments_num)

    # log
    log = open(log_file, mode = 'a')
    # config
    for key, value in config_seq2gmm.items():
        print('{} = {}'.format(key, value), file = log)
    # performance
    print('full AUC\n',fullAUCs,file = log)
    print('full AUPR\n',fullAUPRs,file = log)
    print('best AUC:', max(fullAUCs),file = log)
    print('best AUPR:', max(fullAUPRs),file = log)
    print('\n\n\n',file = log)
    log.close()

