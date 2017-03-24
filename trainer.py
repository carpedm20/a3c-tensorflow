import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import FastSaver

class Trainer(object):
  def __init__(self, agent, env, server, task, base_dir):
    self.env = env
    self.agent = agent
    self.server = server

    self.task = task
    self.base_dir = base_dir

  def train(self):
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()

    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    tf.logging.info('Trainable vars:')
    slim.model_analyzer.analyze_vars(var_list, print_info=True)

    def init_fn(ses):
      tf.logging.info("Initializing all parameters.")
      ses.run(init_all_op)

    sess_config = tf.ConfigProto(
      device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(self.task)])

    summary_writer = tf.summary.FileWriter("{}_{}".format(self.base_dir, self.task))
    tf.logging.info("Events directory: %s_%s", self.base_dir, self.task)
    sv = tf.train.Supervisor(is_chief=(self.task == 0),
                             logdir=self.base_dir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=self.agent.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = 100000000

    with sv.managed_session(self.server.target, config=sess_config) as sess, sess.as_default():
      sess.run(self.agent.sync)
      self.agent.start(sess, summary_writer)

      global_step = sess.run(self.agent.global_step)
      tf.logging.info("Starting training at step=%d", global_step)

      while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
        self.agent.process(sess)
        global_step = sess.run(self.agent.global_step)

    # Ask for all the services to stop.
    sv.stop()
    tf.logging.info('reached %s steps. worker stopped.', global_step)

  def test(self):
    raise Exception("Not implemented yet")
