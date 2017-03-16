import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version

from .layers import *

class LSTMPolicy(object):
  def __init__(self, ob_space, ac_space):
    self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

    kernel_dims = [16, 32]
    kernel_size = [8, 4]
    strides = [4, 2]

    # f_percept
    conv_x = build_conv(x, kernel_dims, kernel_size, strides, scope, reuse=False)
    enc_output = linear(conv_x, config.fc_dim, )

    enc_out = tf.expand_dims(flatten(conv_x), [0])

    size = 256
    lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

    self.state_size = lstm.state_size
    step_size = tf.shape(self.x)[:1]

    c_init = np.zeros((1, lstm.state_size.c), np.float32)
    h_init = np.zeros((1, lstm.state_size.h), np.float32)
    self.state_init = [c_init, h_init]
    c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
    h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
    self.state_in = [c_in, h_in]

    state_in = rnn.LSTMStateTuple(c_in, h_in)
    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
      lstm, enc_out, initial_state=state_in, sequence_length=step_size,
      time_major=False)

    lstm_c, lstm_h = lstm_state
    lstm_outputs = tf.reshape(lstm_outputs, [-1, size])
    self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

    self.logits = linear(lstm_outputs, ac_space, "action", normalized_columns_initializer(0.01))
    self.vf = tf.reshape(linear(lstm_outputs, 1, "value", normalized_columns_initializer(1.0)), [-1])

    self.sample = categorical_sample(self.logits, ac_space)[0, :]
    self.var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

  def get_initial_features(self):
    return self.state_init

  def act(self, ob, c, h):
    sess = tf.get_default_session()
    return sess.run([self.sample, self.vf] + self.state_out,
            {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

  def value(self, ob, c, h):
    sess = tf.get_default_session()
    return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

