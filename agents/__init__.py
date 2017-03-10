from .a3c import A3C
from .n_step_q import NStepQ
from .one_step_q import OneStepQ

def get_agent(config):
  agent_type= config.agent_type.lower()
  if agent_type == 'a3c':
    agent = A3C
  elif agent_type == 'one_step_q':
    agent = OneStepQ
  elif agent_type == 'n_step_q':
    agent = NStepQ
  else:
    raise Exception("Unknown agent type: {}".format(agent_type))
  return agent
