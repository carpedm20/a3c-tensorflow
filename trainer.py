class Trainer(object):
  def __init__(self, agent, env, server, task, log_dir):
    self.env = env
    self.agent = agent
    self.server = server

    self.task = task
    self.log_dir = log_dir

  def train(self):
    # Variable names that start with "local" are not saved in checkpoints.
    if use_tf12_api:
      variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
      init_op = tf.variables_initializer(variables_to_save)
      init_all_op = tf.global_variables_initializer()
    else:
      variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
      init_op = tf.initialize_variables(variables_to_save)
      init_all_op = tf.initialize_all_variables()
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
      logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
      logger.info("Initializing all parameters.")
      ses.run(init_all_op)

    sess_config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(self.task)])
    logdir = os.path.join(self.log_dir, 'train')

    summary_writer = tf.summary.FileWriter(logdir + "_%d" % self.task)

    logger.info("Events directory: %s_%s", logdir, self.task)
    sv = tf.train.Supervisor(is_chief=(self.task == 0),
                logdir=logdir,
                saver=saver,
                summary_op=None,
                init_op=init_op,
                init_fn=init_fn,
                summary_writer=summary_writer,
                ready_op=tf.report_uninitialized_variables(variables_to_save),
                global_step=trainer.global_step,
                save_model_secs=30,
                save_summaries_secs=30)

    num_global_steps = 100000000

    logger.info(
      "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
      "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")

    with sv.managed_session(server.target, config=sess_config) as sess, sess.as_default():
      sess.run(trainer.sync)
      trainer.start(sess, summary_writer)
      global_step = sess.run(trainer.global_step)
      logger.info("Starting training at step=%d", global_step)
      while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
        trainer.process(sess)
        global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

  def test(self):
    raise Exception("Not implemented yet")
