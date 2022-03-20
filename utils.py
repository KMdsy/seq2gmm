# utils.py
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from config import config_seq2gmm
from numpy.linalg import lstsq
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn import metrics
import os
import pwlf
import pickle
import copy
import numpy as np
import scipy.signal as signal

def roc_precision_recall(ground_truth, energy):
    energy = np.array(energy)
    split_num = energy.shape[0]//ground_truth.shape[0]
    _ground_truth = []
    for item in ground_truth:
        for i in range(split_num):
            _ground_truth.append(item)
    _ground_truth = np.array(_ground_truth)
    _energy = np.reshape(energy, (energy.shape[0],))

    precision, recall, _ = precision_recall_curve(_ground_truth, _energy)
    aupr = metrics.auc(recall, precision)
    return aupr

def split_data(raw_data, split):
    data = []
    fragments_num = []
    if split == 1:
        return raw_data, [1]*len(raw_data)

    line_index = 0
    for line in raw_data:
        line_index += 1
        print('split: {}'.format(line_index))
        x = range(len(line))
        my_pwlf = pwlf.PiecewiseLinFit(x, line)
        breaks = my_pwlf.fit(split)
        fragments_num.append(len(breaks)-1)
        for i in range(len(breaks)-1):
            data.append(line[int(breaks[i]):int(breaks[i+1])+1])
    return data, fragments_num

def read_dataset(opts, dataset, type_, split, normal_cluster=None, aug=False):
    '''
    normal_cluster: normal label
    split: number of fragments
    '''
    if aug is True:
        if dataset == 'train':
            data = np.loadtxt(opts['aug_train_file'])
            file_name = opts['aug_train_file']
        elif dataset == 'test':
            data = np.loadtxt(opts['aug_test_file'])
            file_name = opts['aug_test_file']
        elif dataset == 'v':
            data = np.loadtxt(opts['aug_train_file'])
            file_name = opts['aug_train_file']
    else:
        if dataset == 'train':
            data = np.loadtxt(opts['train_file'])
            file_name = opts['train_file']
        elif dataset == 'test':
            data = np.loadtxt(opts['test_file'])
            file_name = opts['test_file']
        elif dataset == 'v':
            data = np.loadtxt(opts['train_file'])
            file_name = opts['train_file']
        
    if normal_cluster ==None:
        cluster = np.unique(data[:,0])
        max_num = 0
        normal_cluster = -1
        for c in cluster:
            t = data[np.where(data[:,0]==c)]
            if t.shape[0] > max_num:
                max_num = t.shape[0]
                normal_cluster = c
    print('Normal Cluster: ', normal_cluster)
    print('Time Series Length: ', data.shape[1])
            
    if type_ == 'normal':
        label = data[np.where(data[:,0]==normal_cluster)][:,0]
        data = data[np.where(data[:,0]==normal_cluster)][:,1:]
    elif type_ == 'abnormal':        
        label = data[np.where(data[:,0]!=normal_cluster)][:,0]
        data = data[np.where(data[:,0]!=normal_cluster)][:,1:]
    elif type_ == 'full':
        normal_data_idx= np.where(data[:,0]==normal_cluster)[0]
        abnormal_data_idx= np.where(data[:,0]!=normal_cluster)[0]
        np.random.seed(1)
        np.random.shuffle(abnormal_data_idx)
        abnormal_data_idx = abnormal_data_idx[0:int(len(normal_data_idx)*0.1)]
        data_idx = np.sort(np.concatenate((normal_data_idx, abnormal_data_idx),axis=0))
        
        label = data[data_idx,0].flatten()
        label[label==normal_cluster] = -2
        data = data[data_idx,1:]


    search_dir = os.path.dirname(file_name)
    target_file = file_name.split('/')[-1].split('.')[0]+'_split_%d'%split+'.pkl'

    if os.path.exists(search_dir+'/'+target_file) == True:
        print(search_dir+'/'+target_file)
        f = open(search_dir+'/'+target_file, 'rb')
        data = pickle.load(f)
        fragments_num = pickle.load(f)
        f.close()
    else:
        data, fragments_num = split_data(data, split)
        f = open(search_dir+'/'+target_file, 'wb')
        print(search_dir+'/'+target_file)
        pickle.dump(data, f)
        pickle.dump(fragments_num, f)
        f.close()
    if split == 1:
        list_data = []
        for row_idx in range(data.shape[0]):
            list_data.append(data[row_idx])
        data = list_data
    return data, fragments_num, label, normal_cluster



def calculate_score(anomaly, energy, fragments_num):
    ground_truth = copy.deepcopy(anomaly)
    ground_truth[ground_truth != -2] = 1
    ground_truth[ground_truth == -2] = 0
    
    
    output_length = energy.shape[0]
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    prediction_list = []
    best_accuracy = 0
    y_pred = []
    j = 0
    for fragment in fragments_num:
        pred = np.max(energy[j:j+fragment])

        j = j + fragment
        y_pred.append(pred)
    # energy of each time series
    energy = y_pred
    

    for i in [1,11,33,44,54,63,71,77,78,79,84,89,93,96,98,99]:
        energy_sorted = sorted(energy)

        boundary = energy_sorted[int(len(energy)*i/100)]
        prediction = []
        for e in energy:
            if e > boundary:
                prediction.append(1)
            else:
                prediction.append(0)
      
        accuracy = accuracy_score(ground_truth, prediction)

        if(accuracy>best_accuracy):
            best_accuracy = accuracy
            prediction_list = prediction
            
        precision = precision_score(ground_truth, prediction)
        recall = recall_score(ground_truth, prediction)
        f1 = f1_score(ground_truth, prediction)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
        print('  precision: ',"%.2f" % precision, '    recall: ',"%.2f" % recall, 'accuracy: ',"%.2f" % accuracy, '    f1_score: ',"%.2f" % f1)
    return energy, output_length, ground_truth, precision_list, recall_list, f1_list, accuracy_list, prediction_list

def make_batch(inputs, max_sequence_length = None, ifNormalize = False):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """       
    if ifNormalize:
        scaler = MinMaxScaler(feature_range=(0, 5))
        inputs = scaler.fit_transform(inputs)           
                    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
        
    print(f'\t\tseq_len={max_sequence_length}, batch_size={batch_size}')
    

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.float64) # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element


    inputs_time_major = inputs_batch_major
    
    return inputs_time_major, sequence_lengths
    
def reconstruction_distances(input_tensor, reconstruction):
    import tensorflow as tf
    with tf.variable_scope('reconstruction_distances'):
        squared_x = tf.reduce_sum(tf.square(input_tensor),
                                  name='squared_x',
                                  axis=1) + 1e-12
        input_tensor = input_tensor[:,:,0]
        reconstruction = reconstruction[:,:,0]
        dist = tf.norm(input_tensor - reconstruction, ord=2, axis=1, keepdims=True, name='dist')
        relative_dist = dist / tf.norm(input_tensor, ord=2, axis=1, keepdims=True, name='relative_dist')                                  
                                          
        # Cosine similarity
        n1 = tf.nn.l2_normalize(input_tensor,1)
        n2 = tf.nn.l2_normalize(reconstruction,1)
        cosine_similarity = tf.reduce_sum(tf.multiply(n1, n2), 1, keepdims=True, name='cosine_similarity')
        return squared_x, relative_dist, cosine_similarity, dist
        
from sklearn.metrics import roc_auc_score


def ROC_AUC(ground_truth,energy):
    '''
    roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None, max_fpr=None)
    y_score: Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions
    '''
    energy = np.array(energy)
    split_num = energy.shape[0]//ground_truth.shape[0]
    _ground_truth = []
    for item in ground_truth:
        for i in range(split_num):
            _ground_truth.append(item)
    _ground_truth = np.array(_ground_truth)
    _energy = np.reshape(energy, (energy.shape[0],))
    auc_score = roc_auc_score(_ground_truth,_energy)

    return auc_score



              
    
    