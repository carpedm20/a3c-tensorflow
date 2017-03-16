import os
import json
import logging
import numpy as np
from sys import platform
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cluster_spec(num_workers, num_ps):
  cluster = {}
  port = 12222

  all_ps = []
  host = '127.0.0.1'
  for _ in range(num_ps):
    all_ps.append('{}:{}'.format(host, port))
    port += 1
  cluster['ps'] = all_ps

  all_workers = []
  for _ in range(num_workers):
    all_workers.append('{}:{}'.format(host, port))
    port += 1
  cluster['worker'] = all_workers
  return cluster

class FastSaver(tf.train.Saver):
  def save(self, sess, save_path, global_step=None, latest_filename=None,
           meta_graph_suffix="meta", write_meta_graph=True):
    super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                meta_graph_suffix, False)

def build_cluster_def(num_workers, num_ps, port=2222):
  cluster = defaultdict(list)

  host = '127.0.0.1'
  for _ in range(num_ps):
    cluster['ps'].append('{}:{}'.format(host, port))
    port += 1

  for _ in range(num_workers):
    cluster['worker'].append('{}:{}'.format(host, port))
    port += 1

  return tf.train.ClusterSpec(cluster).as_cluster_def()

def build_cluster_def_from_file(path, local_ip):
  with open(path) as fp:
    network_config = json.load(fp)

  num_workers = 0
  start_task_ids = defaultdict(int)
  local_config = defaultdict(list)
  new_network_config = defaultdict(list)

  host_ip = network_config['ps'][0].split(':')[0]

  for task_name, task_ips in network_config.items():
    task_ips.sort()

    non_host_ips = [ip for ip in task_ips if host_ip not in ip]
    host_ips = [ip for ip in task_ips if host_ip in ip]

    for task_ip in host_ips + non_host_ips:
      cur_ip, cur_ports = task_ip.split(':')

      splitted_ports = [int(port) for port in cur_ports.split('~')]
      if len(splitted_ports) == 1:
        ports = splitted_ports
      else:
        ports = range(splitted_ports[0], splitted_ports[1]+1)

      cur_ips = ["{}:{}".format(cur_ip, port) for port in ports]
      new_network_config[task_name].extend(cur_ips)

      if task_ip.startswith(local_ip):
        local_config[task_name] = cur_ips

      if task_name == 'worker':
        if not start_task_ids.has_key(cur_ip):
          start_task_ids[cur_ip] = num_workers

        num_workers += len(ports)

  return new_network_config, local_config, start_task_ids[local_ip]

def str2bool(v):
  return v.lower() in ('true', '1')

def prepare_dirs_and_logger(config):
  logger = logging.getLogger('tensorflow')

  for hdlr in logger.handlers:
    logger.removeHandler(hdlr)

  logger.setLevel(tf.logging.INFO)

  if config.load_path:
    if config.load_path.startswith(config.env_name):
      config.model_name = config.load_path
    else:
      config.model_name = "{}_{}".format(config.env_name, config.load_path)
  else:
    config.model_name = "{}_{}".format(config.env_name, get_time())

  config.base_dir = os.path.join(config.log_dir, config.model_name)

  for path in [config.log_dir, config.base_dir]:
    if not os.path.exists(path):
      os.makedirs(path)

def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_config(model_dir, config):
  param_path = os.path.join(model_dir, "params.json")

  tf.logging.info("MODEL dir: %s" % model_dir)
  tf.logging.info("PARAM path: %s" % param_path)

  with open(param_path, 'w') as fp:
    json.dump(config.__dict__, fp,  indent=4, sort_keys=True)

def short_varname(name, max_length=60):
  if len(name) > max_length:
    return name[:max_length] + "..."
  else:
    return name

def analyze_vars(variables=None, grads_and_vars=None):
  print('---------')
  print('Variables: name (type shape) [size]')
  print('---------')

  total_size = 0
  if grads_and_vars is not None:
    var_grad_dict = {var: grad for grad, var in grads_and_vars}

  if variables is None:
    variables = var_grad_dict.keys()
    variable_names = [v.name for v in variables]
    variables = [v for (_, v) in sorted(zip(variable_names, variables))]

  for var in variables:
    var_size = var.get_shape().num_elements() or 0
    total_size += var_size

    sizes = var.get_shape()
    shape_str = "x".join([str(size) for size in sizes])

    if grads_and_vars is not None:
      grad_exists = var_grad_dict[var] is not None
      grad_str = "Grad: X" if not grad_exists else "Grad: O"

      tensor_description = "({}, {} {})".format(grad_str, var.dtype.name, shape_str)
    else:
      tensor_description = "({} {})".format(var.dtype.name, shape_str)

    print "{:<63} {} {}".format(
        short_varname(var.name), tensor_description, '[' + str(var_size) + ']')

  print('---------')
  print('Total size of variables: %d' % total_size)
  print('---------')

  return total_size

def show_all_variables(variables=None, grads_and_vars=None):
  if grads_and_vars is not None:
    analyze_vars(grads_and_vars=grads_and_vars)
  elif variables is None:
    analyze_vars(tf.trainable_variables(), None)
  else:
    variables_to_print = \
        [v for v in variables if v in tf.trainable_variables()]
    analyze_vars(variables_to_print, grads_and_vars)
