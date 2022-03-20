import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

class seq2seq:
    def __init__(self, opts, encoder_inputs, decoder_targets, encoder_inputs_length, is_training=True):
        PAD = opts['PAD']
        EOS = opts['EOS']
        vocab_size = opts['vocab_size']
        encoder_hidden_units = opts['encoder_hidden_units']
        decoder_hidden_units = opts['decoder_hidden_units']
        split = opts['split_frag_num']
        
        with tf.variable_scope("seq2seq", initializer=tf.orthogonal_initializer(), reuse = tf.AUTO_REUSE):
            encoder_cell = tf.contrib.rnn.GRUCell(encoder_hidden_units)

            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs,
                dtype=tf.float64, time_major=True,
                sequence_length=encoder_inputs_length
            )
            self.encoder_result = encoder_final_state,
            #del encoder_outputs
            self.encoder_outputs = encoder_outputs
            del encoder_outputs
            # +2 additional steps, +1 leading <EOS> token for decoder inputs
            if split > 1:
                decoder_lengths = encoder_inputs_length + 3
                decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)
                encoder_max_time, batch_size, _ = tf.unstack(tf.shape(encoder_inputs))
                
                W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1, dtype=tf.float64))
                b = tf.Variable(tf.zeros([vocab_size], dtype=tf.float64))
                
                assert EOS == 1 and PAD == 0

                eos_step = tf.ones([batch_size, vocab_size], dtype=tf.float64, name='EOS')
                pad_step = tf.zeros([batch_size, vocab_size], dtype=tf.float64, name='PAD')

                def loop_fn_initial():
                    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
                    initial_input = eos_step
                    initial_cell_state = encoder_final_state
                    initial_cell_output = None
                    initial_loop_state = None  # we don't need to pass any additional information
                    return (initial_elements_finished,
                            initial_input,
                            initial_cell_state,
                            initial_cell_output,
                            initial_loop_state)
                
                def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

                    def get_next_input():
                        output_logits = tf.add(tf.matmul(previous_output, W), b)
                        return output_logits
                    
                    elements_finished = (time >= decoder_lengths) 
                    finished = tf.reduce_all(elements_finished) # -> boolean scalar
                    input = tf.cond(finished, lambda: pad_step, get_next_input)
                    state = previous_state
                    output = previous_output
                    loop_state = None

                    return (elements_finished, 
                            input,
                            state,
                            output,
                            loop_state)
                            
                def loop_fn(time, previous_output, previous_state, previous_loop_state):
                    if previous_state is None:    # time == 0
                        assert previous_output is None and previous_state is None
                        return loop_fn_initial()
                    else:
                        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
                        
                with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE ):
                    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
                decoder_outputs = decoder_outputs_ta.stack()
                
                decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
                decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
                decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
                self.decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
            else:
                decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)
                decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                    decoder_cell, decoder_targets,
                    dtype=tf.float64, time_major=True,
                    initial_state=encoder_final_state,
                    sequence_length=encoder_inputs_length + 3
                )
                self.decoder_logits = decoder_outputs

            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=decoder_targets,
                logits=self.decoder_logits,
            )
            self.loss = tf.reduce_mean(stepwise_cross_entropy)