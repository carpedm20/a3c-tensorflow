from .base import BaseAgent

class OneStepQ(BaseAgent):
  def __init__(self, model_fn, env, config):
    super(OneStepQ, self).__init__(model_fn, env, config)
