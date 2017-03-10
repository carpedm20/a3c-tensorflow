from .base import BaseAgent

class NStepQ(BaseAgent):
  def __init__(self, model_fn, env, task):
    super(OneStepQ, self).__init__(model_fn, env, task)
