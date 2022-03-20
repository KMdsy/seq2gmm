# Estimator.py
import tensorflow as tf

class Estimator:
    def __init__(self, opts, z, is_training=True):
        with tf.variable_scope('estimator'):
            self.input_tensor = tf.cast(z, dtype=tf.float32, name=None)
            
            net = tf.keras.layers.Dense(128)(self.input_tensor)
            net = tf.keras.layers.Dense(128)(net)
            net = tf.keras.layers.Dense(opts['num_mixture'])(net)
            self.output_tensor = tf.nn.softmax(net, name='predicted_memebership')
            