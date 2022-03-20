import re, os
import numpy as np
import tensorflow as tf

from seq2seq import seq2seq
from Estimator import Estimator
from GMM import GMM
from utils import reconstruction_distances, make_batch, calculate_score
from utils import ROC_AUC, roc_precision_recall



LOG_DIR = 'log_train'
CHECKPOINT_DIR = 'checkpoint'
class Seq2GMM(object):
    
    def __init__(self, opts, log_file):

        # model name
        self.model_name = 'seq2gmm'

        # log_file
        if os.path.exists(LOG_DIR) == False: os.makedirs(LOG_DIR)
        self.log_file = LOG_DIR + '/' + log_file.split('/')[-1].split('.')[0]+'_tmp.log'
        print('\n\nlog_file: {}\n\n'.format(self.log_file))
        if os.path.exists(self.log_file) == False:
            f = open(self.log_file, 'w')
            f.close()

        # config
        self.cfg = 'split: {}, encoder: {}, mix: {}'.format(opts['split_frag_num'], opts['encoder_hidden_units'], opts['num_mixture'])
        
        # checkpoint
        datasetname = opts['train_file'].split('/')[-1].split('_TRAIN')[0]
        print('\n\n{}\n\n'.format(datasetname))
        self.checkpoint_dir = '{}/{}/split_{}_encoder_{}_mix_{}'.format(CHECKPOINT_DIR, datasetname, opts['split_frag_num'], opts['encoder_hidden_units'], opts['num_mixture'])
        if os.path.exists(self.checkpoint_dir) == False: os.makedirs(self.checkpoint_dir)
        
        # others
        self.opts = opts 
        self.is_training = tf.placeholder_with_default(tf.constant(True), [], name='is_training')
        
        self._create_network()
        self._create_loss_optimizer()
        self.savers()
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
              
    def _create_network(self):
        opts = self.opts
        '''
        compression network
        '''
        self.encoder_inputs = tf.placeholder(shape=(None, None, opts['vocab_size']), dtype=tf.float64, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.frag_index = tf.placeholder(shape=(None, 1), dtype=tf.float64, name='fragment_index')
        self.decoder_targets = tf.placeholder(shape=(None, None, opts['vocab_size']), dtype=tf.float64, name='decoder_targets')
        self.ae = seq2seq(opts=self.opts, encoder_inputs=self.encoder_inputs, decoder_targets=self.decoder_targets, encoder_inputs_length=self.encoder_inputs_length, is_training=self.is_training)
        self.reconstruction = tf.transpose(self.ae.decoder_logits,[1,0,2])
        self.zc = self.ae.encoder_result[0]
        self.aeloss = self.ae.loss
        sq_x, self.relative_dist, self.cosine_similarity, dist = reconstruction_distances(tf.transpose(self.decoder_targets,[1,0,2]), self.reconstruction)
       
       
        '''
        estimator network
        '''      
        self.z = tf.concat([self.relative_dist, self.zc, self.cosine_similarity], axis=1)
        self.es = Estimator(opts=self.opts, z=self.z, is_training=self.is_training) 
        self.gamma = self.es.output_tensor
        self.label = tf.argmax(self.gamma,1)
        
        
        '''
        GMM
        '''
        self.gmm = GMM(opts)
        self.likelihood, self.energy = self.gmm.compute_likelihood(self.z, self.gamma)
        self.popt = tf.Variable(tf.zeros(shape=[opts['num_mixture'],2], dtype=tf.float64), name='popt')
        cal_energy = -tf.log(self.likelihood + 1e-12)
        param_list = []
        for i in range(opts['num_mixture']):
            extract = tf.gather(cal_energy,tf.where(tf.equal(self.label,i)))
            mean, var = tf.nn.moments(extract, axes=[0])
            param_list.append([mean, var])
        self.update_popt = self.popt.assign(tf.squeeze(tf.convert_to_tensor(param_list), axis=[2]))
        
        #output of whole net
        self.cons_error = tf.reduce_mean(dist, name='loss_reconstruction')
        
    def _create_loss_optimizer(self):
        lambda_1 = self.opts['lambda_1']
        '''
        Loss
        '''
        self.mlp_loss = self.energy
        self.loss = lambda_1 * self.mlp_loss + self.cons_error
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        pre_train_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.opts['learning_rate'])
        
        mlp_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        estimator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimator')
        seq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='seq2seq')
        joint_vars = estimator_vars + seq_vars
        
        with tf.control_dependencies(update_ops):
            self.mlp_op = mlp_optimizer.minimize(self.mlp_loss, var_list=estimator_vars)
            self.rec_op = pre_train_optimizer.minimize(self.cons_error)
            self.train_op = optimizer.minimize(self.loss, var_list=joint_vars)
    
    def savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=opts['max_to_keep'])       
        tf.add_to_collection('vars', self.gmm.u_k_list)
        tf.add_to_collection('vars', self.gmm.sigma_k_list)
        self.saver = saver
    
    def train(self, data, test_label, test_data, fragments_num, t_fragments_num, ifNormalize = False):
        opts =self.opts
        fullAUPRs = []
        fullAUCs = [] 
        rdAUCs = []
        csAUCs = []
        energys = []
        losses = []
        EOS = opts['EOS']
        PAD = opts['PAD']
        enum = opts['epoch_num']
        targets = [(sequence.tolist()) + [EOS] + [PAD] * 2 for sequence in data]
        data, inputs_length = make_batch(data, ifNormalize = ifNormalize)
        targets, _ = make_batch(targets, ifNormalize = ifNormalize)
        data = np.expand_dims(data, axis=2).swapaxes(0, 1)
        targets = np.expand_dims(targets, axis=2).swapaxes(0, 1)

        fragment_index = []
        for n in fragments_num:
            fragment_index = fragment_index + [i for i in range(n)]
            
        feed_d = {
            self.encoder_inputs: data,
            self.encoder_inputs_length: inputs_length,
            self.frag_index: np.array(fragment_index)[:, np.newaxis],
            self.decoder_targets: targets,
            self.is_training: True}
        
        self.best_AUC = [0,0]
        self.best_AUPR = [0,0]
        _, error, z = self.sess.run([self.rec_op, self.cons_error, self.z], feed_dict=feed_d)
        for epoch in range(1, enum):  
            '''
            Train
            '''                         
            means, covariance = self.gmm.pre_train(z)
            self.sess.run([self.gmm.pre_train_mu, self.gmm.pre_train_sigma], feed_dict=feed_d)
            
            '''
            joint Train
            '''
            _, error, z, loss, mlp_loss = self.sess.run([self.train_op, self.cons_error, self.z, self.loss, self.mlp_loss], feed_dict=feed_d)
            print('loss:',loss,'recon:',error,'energy:',mlp_loss, 'epoch:', epoch)
            
            losses.append(loss)
            
            
            if epoch % 10 == 0:
                self.saver.save(self.sess,os.path.join(self.checkpoint_dir,self.model_name),global_step=epoch)
                pred_energy, rd, cs, z = self.test(test_data, t_fragments_num, ifNormalize)
                energy, output_length, ground_truth, precision_list, recall_list, f1_list, accuracy_list, prediction_list = calculate_score(test_label, pred_energy, t_fragments_num)


                if(np.sum(ground_truth) != 0):
                    fullAUC = ROC_AUC(ground_truth, energy)    
                    rdAUC = ROC_AUC(ground_truth, rd)
                    csAUC = ROC_AUC(ground_truth, cs)
                    fullAUPR = roc_precision_recall(ground_truth, energy)

                
                    if fullAUC > self.best_AUC[0]:
                        self.best_AUC = [fullAUC, epoch]
                    if fullAUPR > self.best_AUPR[0]:
                        self.best_AUPR = [fullAUPR, epoch]
                        
                    # delete useless model
                    all_model_list = os.listdir(self.checkpoint_dir)
                    ep_set = list(set([self.best_AUPR[1], self.best_AUC[1]]))
                    all_model_list.remove('checkpoint')
                    for ep in ep_set:
                        all_model_list.remove('{}-{}.data-00000-of-00001'.format(self.model_name, ep))
                        all_model_list.remove('{}-{}.index'.format(self.model_name, ep))
                        all_model_list.remove('{}-{}.meta'.format(self.model_name, ep))
                    for _, item in enumerate(all_model_list):
                        os.remove(os.path.join(self.checkpoint_dir, item))
                    print('save to', self.checkpoint_dir)
                    print('model no.{} saved, best aupr: {}'.format(self.best_AUPR[1], self.best_AUPR[0]))
                    print('model no.{} saved, best auc: {}'.format(self.best_AUC[1], self.best_AUC[0]))
                    fullAUPRs.append(fullAUPR)
                    fullAUCs.append(fullAUC)
                    rdAUCs.append(rdAUC)
                    csAUCs.append(csAUC)
   
        return fullAUCs, rdAUCs, csAUCs, losses, fullAUPRs
        
    def test(self, data, fragments_num, ifNormalize = False):
        var_list = tf.get_collection('vars')

        opts = self.opts
        EOS = opts['EOS']
        PAD = opts['PAD']
        enum = opts['epoch_num']
        targets = [(sequence.tolist()) + [EOS] + [PAD] * 2 for sequence in data]
        data, inputs_length = make_batch(data, ifNormalize = ifNormalize)
        targets, _ = make_batch(targets, ifNormalize = ifNormalize)
        data = np.expand_dims(data, axis=2).swapaxes(0, 1)
        targets = np.expand_dims(targets, axis=2).swapaxes(0, 1)

        fragment_index = []
        for n in fragments_num:
            fragment_index = fragment_index + [i for i in range(n)]
            
        feed_d = {
            self.encoder_inputs: data,
            self.encoder_inputs_length: inputs_length,
            self.frag_index: np.array(fragment_index)[:, np.newaxis],
            self.decoder_targets: targets,
            self.is_training: False}
                
        likelihood, _ = self.gmm.compute_likelihood(self.z, self.gamma, var_list[0], var_list[1])
        vars_value, likelihood, z, label, popt, encoder_outputs, recon, rd, cs = self.sess.run([var_list, likelihood, self.z, self.label, self.popt, self.ae.encoder_outputs, self.reconstruction, self.relative_dist, self.cosine_similarity], feed_dict=feed_d)

        cal_energy = - np.log(likelihood + 1e-12)

        return np.array(cal_energy), rd, cs, z