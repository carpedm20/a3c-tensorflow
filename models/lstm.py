import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version

from .layers import *

class LSTMPolicy(object):
  def __init__(self, ob_space, ac_space):
    self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

    for i in range(4):
      x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
    # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
    x = tf.expand_dims(flatten(x), [0])

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
      lstm, x, initial_state=state_in, sequence_length=step_size,
      time_major=False)

    lstm_c, lstm_h = lstm_state
    x = tf.reshape(lstm_outputs, [-1, size])
    self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
    self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
    self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
    self.sample = categorical_sample(self.logits, ac_space)[0, :]
    self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

  def get_initial_features(self):
    return self.state_init

  def act(self, ob, c, h):
    sess = tf.get_default_session()
    return sess.run([self.sample, self.vf] + self.state_out,
            {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

  def value(self, ob, c, h):
    sess = tf.get_default_session()
    return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

