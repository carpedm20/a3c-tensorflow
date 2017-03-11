import os
import sys
import math
import argparse
from six.moves import shlex_quote

from utils import get_time, str2bool, build_cluster_def_from_file

parser = argparse.ArgumentParser(description='Run commands')
parser.add_argument('-n', '--dry-run', action='store_true')
parser.add_argument('--monitor', type=str, default='tmux')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--board_port', type=int, default=6006)
parser.add_argument('--local_ip', type=str, default='127.0.0.1')
parser.add_argument('--cluster_def_file', type=str, default='cluster.json')

def new_cmd(session, name, cmd, monitor, log_dir, shell):
  if isinstance(cmd, (list, tuple)):
    cmd = ' '.join(shlex_quote(str(v)) for v in cmd)

  if monitor == 'tmux':
    return name, 'tmux send-keys -t {}:{} {} Enter'. \
        format(session, name, shlex_quote(cmd))
  elif monitor == 'child':
    return name, '{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh'. \
        format(cmd, log_dir, session, name, log_dir)
  elif monitor == 'nohup':
    return name, 'nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh'. \
        format(shell, shlex_quote(cmd), log_dir, session, name, log_dir)

def create_commands(
    config, session,
    shell='bash', monitor='tmux', mode=2, board_port=6006):
  log_dir = config.log_dir

  base_cmd = [
      'CUDA_VISIBLE_DEVICES=', sys.executable, 'main.py',
      '--log_dir', log_dir,
      '--is_train', config.is_train,
      '--local_ip', config.local_ip,
  ]

  time_str = get_time()
  def get_load_path(idx):
    if config.load_path == '':
      load_path = "{}-{}".format(time_str, idx)
    else:
      load_path = config.load_path
    return load_path

  cluster_def, local_def, start_task_id = build_cluster_def_from_file(
      config.cluster_def_file, config.local_ip)

  num_ps = len(local_def['ps'])
  num_worker = len(local_def['worker'])

  base_cmd.extend(['--num_workers', str(num_ps + num_worker)])

  cmds_map = []
  screen_idx = 0
  for task_name, task_ips in local_def.items():
    for idx, _ in enumerate(task_ips):
      task_arg = [
          '--job_name', task_name, '--task', start_task_id + idx,
          #'--load_path', get_load_path((start_task_id + idx) % int(math.sqrt(num_worker)))]
          '--load_path', get_load_path(start_task_id + idx)]
      cmds_map += [
          new_cmd(session, 'w-%d' % screen_idx, base_cmd + task_arg,  monitor, log_dir, shell)]
      screen_idx += 1

  board_config = ['tensorboard', '--logdir', log_dir, '--host', '0.0.0.0', '--port', config.board_port]
  cmds_map += [new_cmd(session, 'tb', board_config, monitor, log_dir, shell)]
  if monitor == 'tmux':
    cmds_map += [new_cmd(session, 'htop', ['htop'], monitor, log_dir, shell)]

  windows = [v[0] for v in cmds_map]

  notes = []
  cmds = [
    'mkdir -p {}'.format(log_dir),
    'echo {} {} > {}/cmd.sh'. \
        format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']), log_dir),
  ]

  if monitor == 'nohup' or monitor == 'child':
    cmds += ["echo '#!/bin/sh' >{}/kill.sh".format(log_dir)]
    notes += ['Run `source {}/kill.sh` to kill the job'.format(log_dir)]
  if monitor == 'tmux':
    notes += ['Use `tmux attach -t {}` to watch process output'.format(session)]
    notes += ['Use `tmux kill-session -t {}` to kill the job'.format(session)]
  else:
    notes += ['Use `tail -f {}/*.out` to watch process output'.format(log_dir)]
  notes += ['Point your browser to http://localhost:6006 to see Tensorboard']

  if monitor == 'tmux':
    cmds += [
    'tmux kill-session -t {}'.format(session),
    'tmux new-session -s {} -n {} -d {}'.format(session, windows[0], shell)
    ]
    for w in windows[1:]:
      cmds += ['tmux new-window -t {} -n {} {}'.format(session, w, shell)]
    cmds += ['sleep 1']
  for window, cmd in cmds_map:
    cmds += [cmd]

  return cmds, notes

def run():
  args = parser.parse_args()
  cmds, notes = create_commands(args, 'a3c')

  if args.dry_run:
    print('Dry-run mode due to -n flag, otherwise the following commands would be executed:')
  else:
    print('Executing the following commands:')

  print('\n'.join(cmds))
  print('')
  if not args.dry_run:
    if args.monitor == 'tmux':
      os.environ['TMUX'] = ''
    os.system('\n'.join(cmds))
  print('\n'.join(notes))

if __name__ == '__main__':
  run()
