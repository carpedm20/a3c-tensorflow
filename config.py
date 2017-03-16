#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--rnn_dim', type=int, default=256)
net_arg.add_argument('--hidden_dim', type=int, default=256)
net_arg.add_argument('--model_type', default='lstm')

# Environmnet
env_arg = add_argument_group('Environmnet')
env_arg.add_argument('--env', type=str, default='PongDeterministic-v0')
env_arg.add_argument('--terminal_if_dead', type=str2bool, default=False)

# Cluster
cluster_arg = add_argument_group('Cluster')
cluster_arg.add_argument('--job_name', type=str, default='worker', choices=['worker', 'ps'])
cluster_arg.add_argument('--task', type=int, default=0)
cluster_arg.add_argument('--local_ip', type=str, default='127.0.0.1')
cluster_arg.add_argument('--cluster_def_file', type=str, default="cluster.json")

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--agent_type', default='a3c')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--random_initialize', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='rmsprop')
train_arg.add_argument('--max_global_step', type=int, default=100000000)
train_arg.add_argument('--lr_start', type=float, default=0.001)
train_arg.add_argument('--lr_decay_step', type=int, default=5000)
train_arg.add_argument('--lr_decay_rate', type=float, default=1, help='1 means no lr decay')
train_arg.add_argument('--max_grad_norm', type=float, default=2.0)
train_arg.add_argument('--max_local_step', type=int, default=20)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--checkpoint_secs', type=int, default=300)
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--record_step', type=int, default=1000)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--output_dir', type=str, default='outputs')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--debug', type=str2bool, default=False)
misc_arg.add_argument('--random_seed', type=int, default=123)

def get_config():
  config, unparsed = parser.parse_known_args()

  if not config.is_train:
    setattr(config, 'random_initialize', False)
  setattr(config, 'random_seed', config.random_seed + config.task)

  return config, unparsed
