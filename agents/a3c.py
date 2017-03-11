import numpy as np
import scipy.signal
import tensorflow as tf
from collections import namedtuple

from .base import BaseAgent

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

def discount(x, gamma):
  return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class A3C(BaseAgent):
  def __init__(self, model_fn, env, task):
    super(A3C, self).__init__(model_fn, env, task)

    pi = self.local_network

    with tf.device(self.worker_device):
      self.adv = tf.placeholder(tf.float32, [None], name="adv")

      log_prob_tf = tf.nn.log_softmax(pi.logits)
      prob_tf = tf.nn.softmax(pi.logits)

      self.pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

      # loss of value function
      self.vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
      self.entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

      self.loss = self.pi_loss + 0.5 * self.vf_loss - self.entropy * 0.01

      self.build_shared_grad()

  def process_rollout(self, rollout, gamma=0.99, lambda_=1.0):
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]

    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)
