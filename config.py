#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
parser.add_argument('--task', default=0, type=int, help='Task index')
parser.add_argument('--job-name', default="worker", help='worker or ps')
parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
parser.add_argument('--is_train', default=True)
parser.add_argument('--random_seed', default=123)
parser.add_argument('--model_type', default='lstm')
parser.add_argument('--agent_type', default='a3c')

parser.add_argument('-r', '--remotes', default=None,
          help='References to environments to create (e.g. -r 20), '
              'or the address of pre-existing VNC servers and '
              'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')
parser.add_argument('--visualise', action='store_true',
          help="Visualise the gym environment by running env.render() between each timestep")

def get_config():
  config, unparsed = parser.parse_known_args()

  if not config.is_train:
    setattr(config, 'random_initialize', False)
  setattr(config, 'random_seed', config.random_seed + config.task)

  return config, unparsed
