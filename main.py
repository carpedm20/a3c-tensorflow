#!/usr/bin/env python
import universe

import time
import sys, signal
import tensorflow as tf

from trainer import Trainer
from agents import get_agent
from models import get_model
from config import get_config
from utils import cluster_spec
from env.atari import create_env

config = None

def shutdown(signal, frame):
  tf.logging.warn('Received signal %s: exiting', signal)
  sys.exit(128 + signal) # because we already hooked

def main(_):
  spec = cluster_spec(config.num_workers, 1)
  cluster = tf.train.ClusterSpec(spec).as_cluster_def()

  signal.signal(signal.SIGHUP, shutdown)
  signal.signal(signal.SIGINT, shutdown)
  signal.signal(signal.SIGTERM, shutdown)

  if config.job_name == "worker":
    server = tf.train.Server(
        cluster, job_name="worker", task_index=config.task,
        config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))

    env = create_env(config.env_id, client_id=str(config.task))
    model_fn = lambda: get_model(config)(env.observation_space.shape, env.action_space.n)
    agent = get_agent(config)(model_fn, env, config)
    trainer = Trainer(agent, env, server, config.task, config.log_dir)

    if config.is_train:
      trainer.train()
    else:
      trainer.test()
  else:
    server = tf.train.Server(cluster, job_name="ps", task_index=config.task,
                 config=tf.ConfigProto(device_filters=["/job:ps"]))
    while True:
      time.sleep(1000)

if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
