import math
import tensorflow as tf
import numpy as np
import sklearn.mixture as mix

class GMM:
    def __init__(self, opts):
        self.config = opts
        self.num_dim = opts['GMM_input_dim']
        self.num_mixture = opts['num_mixture']
        #num_dynamic_dim is the number of dimensions in the low-dimensional 
        #representations provided by the compression network.
        self.num_dynamic_dim = opts['num_dynamic_dim']
        
        #K-means
        K = self.num_mixture
        z_dim = self.num_dim
        mu = 0.
        sigma = 5.
        u_k_init = np.random.normal(mu, sigma, [K, z_dim])
        sigma_init = tf.constant(np.ones((K, z_dim)))
        
        self.u_k_list = tf.Variable(u_k_init, name='u_k_list', dtype=tf.float64)
        self.sigma_k_list = tf.Variable(sigma_init, name='sigma_k_list', dtype=tf.float64)
     
        
    def pre_train(self, data):
        gmm = mix.GaussianMixture(n_components=self.num_mixture, covariance_type='diag', random_state=520)
        gmm.fit(data)
        new_cluster_mean = gmm.means_
        new_cluster_sigma = gmm.covariances_
        self.pre_train_mu = tf.assign(self.u_k_list, new_cluster_mean)
        self.pre_train_sigma = tf.assign(self.sigma_k_list, new_cluster_sigma)
        return gmm.means_, gmm.covariances_
        
    def compute_likelihood(self, sample_z, pi_input, mu=None, sigma=None):
        if mu == None:
            mu = self.u_k_list
        if sigma == None:
            sigma = self.sigma_k_list
        p = tf.cast(pi_input, dtype=tf.float64)
        x = sample_z
        x_t = tf.reshape(x, shape=[-1, 1, self.num_dim])
        x_t = tf.tile(x_t, [1, self.num_mixture, 1])

        mixture_mean = tf.cast(mu, dtype=tf.float64)

        det_diag_cov = tf.reduce_prod(sigma, axis=1)
        x_t64 = tf.cast(x_t, dtype=tf.float64)
        x_sub_mean = x_t64 - mixture_mean
        
        z_norm = x_sub_mean ** 2 / sigma
        t1 = p * tf.reduce_prod(tf.exp(-0.5 * z_norm), axis=2)
        t2 = ((2 * math.pi) ** (0.5 * self.num_dim)) * (det_diag_cov ** 0.5)
        tmp = (t1 / t2)

        likelihood = tf.reduce_sum(tmp, 1)
        energy = tf.reduce_mean(-tf.log(likelihood + 1e-12))
        return likelihood, energy
    