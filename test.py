'''
@author: Shaoyu Dou
'''
from config import config_seq2gmm
from utils import read_dataset
from Seq2GMM import Seq2GMM
from utils import calculate_score
from utils import ROC_AUC, roc_precision_recall
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import visualize as show
import os

dataset_dir = 'UCRArchive_2018'
dataset_name = 'TwoLeadECG'
log_dir = 'log'
if os.path.exists(log_dir) == False: os.makedirs(log_dir)
log_file = log_dir + '/' + dataset_name+'.txt'

if __name__ == "__main__":

    # set params of trained model
    trained_model_dir = 'trained_model/split_3_encoder_15_mix_1/'
    model_name = 'seq2gmm_model'
    config_seq2gmm['split_frag_num'] = 3
    config_seq2gmm['encoder_hidden_units'] = 15
    config_seq2gmm['num_mixture'] = 1
    # 
    config_seq2gmm['decoder_hidden_units'] = config_seq2gmm['encoder_hidden_units']
    config_seq2gmm['GMM_input_dim'] = config_seq2gmm['encoder_hidden_units'] + 2
    config_seq2gmm['num_dynamic_dim'] = config_seq2gmm['encoder_hidden_units']

    data, fragments_num, _, normal_cluster = read_dataset(config_seq2gmm, 'train', 'normal', config_seq2gmm['split_frag_num'])
    t_data, t_fragments_num, t_label, _ = read_dataset(config_seq2gmm, 'test', 'full', config_seq2gmm['split_frag_num'], normal_cluster)

    # load model
    seq2gmm = Seq2GMM(config_seq2gmm, log_file)
    
    saver = tf.train.Saver()
    saver.restore(seq2gmm.sess, os.path.join(trained_model_dir, model_name))
    print('model loaded.')


    pred_energy, _, _, _ = seq2gmm.test(t_data, t_fragments_num)
    energy, output_length, ground_truth, precision_list, recall_list, f1_list, accuracy_list, prediction_list = calculate_score(t_label, pred_energy, t_fragments_num)

    auc = ROC_AUC(ground_truth, energy)
    aupr = roc_precision_recall(ground_truth, energy)

    print('test dataset: {}, auc: {}, aupr: {}'.format(config_seq2gmm['train_file'], auc, aupr))
