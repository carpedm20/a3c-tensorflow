from .base import BaseAgent

class A3C(BaseAgent):
  def __init__(self, model_fn, env, task):
    super(A3C, self).__init__(model_fn, env, task)

    worker_device = "/job:worker/task:{}/cpu:0".format(task)

    with tf.device(worker_device):
      self.adv = tf.placeholder(tf.float32, [None], name="adv")

      log_prob_tf = tf.nn.log_softmax(pi.logits)
      prob_tf = tf.nn.softmax(pi.logits)

      pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

      # loss of value function
      vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
      entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

      bs = tf.to_float(tf.shape(pi.x)[0])
      self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

      self.runner = RunnerThread(env, pi, 20, visualise)

      self.build_shared_grad()
