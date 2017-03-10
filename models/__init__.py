from .fc import FCPolicy
from .lstm import LSTMPolicy

def get_model(config):
  model_type= config.model_type.lower()
  if model_type == 'fc':
    model = FCPolicy
  elif model_type == 'lstm':
    model = LSTMPolicy
  else:
    raise Exception("Unknown model type: {}".format(model_type))
  return model
