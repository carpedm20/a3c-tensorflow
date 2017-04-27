import tensorflow as tf

try:
  import queue
except:
  from six.moves import queue

class BaseAgent(object):
  def __init__(self, model_fn, env, config):
    self.env = env
    self.config = config
    self.task = config.task
    self.worker_device = "/job:worker/task:{}/cpu:0".format(task)

    with tf.device(tf.train.replica_device_setter(1, worker_device=self.worker_device)):
      with tf.variable_scope("global"):
        self.network = model_fn()
        self.global_step = tf.get_variable(
            "global_step", [], tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

    with tf.device(self.worker_device):
      with tf.variable_scope("local"):
        self.local_network = model_fn()
        self.local_network.global_step = self.global_step

      self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
      self.r = tf.placeholder(tf.float32, [None], name="r")

  def build_shared_grad(self):
    self.grads = tf.gradients(self.loss, self.local_network.var_list)

    clipped_grads, _ = tf.clip_by_global_norm(self.grads, self.config.max_grad_norm)

    # copy weights from the parameter server to the local model
    self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.local_network.var_list, self.network.var_list)])

    grads_and_vars = list(zip(clipped_grads, self.network.var_list))
    inc_step = self.global_step.assign_add(tf.shape(self.local_network.x)[0])

    # each worker has a different set of adam optimizer parameters
    self.lr = tf.train.exponential_decay(
            self.config.lr_start, self.global_step, self.config.lr_decay_step,
            self.config.lr_decay_rate, staircase=True, name='lr')

    opt = tf.train.AdamOptimizer(self.lr)
    self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
    self.summary_writer = None
    self.local_steps = 0

    self.build_summary()

  def start(self, sess, summary_writer):
    self.env.runner.start_runner(sess, self.local_network, summary_writer)
    self.summary_writer = summary_writer

  def pull_batch_from_queue(self):
    rollout = self.env.runner.queue.get(timeout=600.0)
    while not rollout.terminal:
      try:
        rollout.extend(self.env.runner.queue.get_nowait())
      except queue.Empty:
        break
    return rollout

  def process(self, sess):
    sess.run(self.sync)  # copy weights from shared to local
    rollout = self.pull_batch_from_queue()
    batch = self.process_rollout(rollout, gamma=0.99, lambda_=1.0)

    should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

    if should_compute_summary:
      fetches = [self.summary_op, self.train_op, self.global_step]
    else:
      fetches = [self.train_op, self.global_step]

    feed_dict = {
      self.local_network.x: batch.si,
      self.ac: batch.a,
      self.adv: batch.adv,
      self.r: batch.r,
      self.local_network.state_in[0]: batch.features[0],
      self.local_network.state_in[1]: batch.features[1],
    }

    fetched = sess.run(fetches, feed_dict=feed_dict)

    if should_compute_summary:
      self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
      self.summary_writer.flush()
    self.local_steps += 1

  def build_summary(self):
    bs = tf.to_float(tf.shape(self.local_network.x)[0])

    tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
    tf.summary.scalar("model/value_loss", self.vf_loss / bs)
    tf.summary.scalar("model/entropy", self.entropy / bs)
    tf.summary.image("model/state", self.local_network.x)
    tf.summary.scalar("model/grad_global_norm", tf.global_norm(self.grads))
    tf.summary.scalar("model/var_global_norm", tf.global_norm(self.local_network.var_list))
    tf.summary.scalar("model/lr", self.lr)

    self.summary_op = tf.summary.merge_all()

  def process_rollout(self, rollout):
    raise Exception("Not implemented yet")
