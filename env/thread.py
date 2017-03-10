class PartialRollout(object):
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.values = []
    self.r = 0.0
    self.terminal = False
    self.features = []

  def add(self, state, action, reward, value, terminal, features):
    self.states += [state]
    self.actions += [action]
    self.rewards += [reward]
    self.values += [value]
    self.terminal = terminal
    self.features += [features]

  def extend(self, other):
    assert not self.terminal
    self.states.extend(other.states)
    self.actions.extend(other.actions)
    self.rewards.extend(other.rewards)
    self.values.extend(other.values)
    self.r = other.r
    self.terminal = other.terminal
    self.features.extend(other.features)

class RunnerThread(threading.Thread):
  def __init__(self, env, policy, num_local_steps, visualise):
    threading.Thread.__init__(self)
    self.queue = queue.Queue(5)
    self.num_local_steps = num_local_steps
    self.env = env
    self.last_features = None
    self.policy = policy
    self.daemon = True
    self.sess = None
    self.summary_writer = None
    self.visualise = visualise

  def start_runner(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer
    self.start()

  def run(self):
    with self.sess.as_default():
      self._run()

  def _run(self):
    rollout_provider = env_runner(
        self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise)
    while True:
      self.queue.put(next(rollout_provider), timeout=600.0)

def env_runner(env, policy, num_local_steps, summary_writer, render):
  last_state = env.reset()
  last_features = policy.get_initial_features()
  length = 0
  rewards = 0

  while True:
    terminal_end = False
    rollout = PartialRollout()

    for _ in range(num_local_steps):
      fetched = policy.act(last_state, *last_features)
      action, value_, features = fetched[0], fetched[1], fetched[2:]
      # argmax to convert from one-hot
      state, reward, terminal, info = env.step(action.argmax())
      if render:
        env.render()

      # collect the experience
      rollout.add(last_state, action, reward, value_, terminal, last_features)
      length += 1
      rewards += reward

      last_state = state
      last_features = features

      if info:
        summary = tf.Summary()
        for k, v in info.items():
          summary.value.add(tag=k, simple_value=float(v))
        summary_writer.add_summary(summary, policy.global_step.eval())
        summary_writer.flush()

      timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
      if terminal or length >= timestep_limit:
        terminal_end = True
        if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
          last_state = env.reset()
        last_features = policy.get_initial_features()
        print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
        length = 0
        rewards = 0
        break

    if not terminal_end:
      rollout.r = policy.value(last_state, *last_features)

    # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
    yield rollout
