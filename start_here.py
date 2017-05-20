__author__ = 'frankhe'

import time
import curses
import sys
import tensorflow as tf
import numpy as np
import multiprocessing as mp
import copy
import interaction
import neural_networks
import agents

FLAGS = tf.app.flags.FLAGS

# Experiment settings
tf.app.flags.DEFINE_integer('epochs', 50, 'Number of training epochs')
tf.app.flags.DEFINE_integer('steps_per_epoch', 250000, 'Number of steps per epoch')
tf.app.flags.DEFINE_integer('test_length', 125000, 'Number of steps per test')
tf.app.flags.DEFINE_integer('seed', 1, 'random seed')
tf.app.flags.DEFINE_integer('summary_fr', 500, 'summary every x training steps')
tf.app.flags.DEFINE_string('logs_path', './logs', 'tensor board path')
tf.app.flags.DEFINE_bool('curses', False, 'if use curses to show status')

# ALE Environment settings
tf.app.flags.DEFINE_string('rom', 'breakout', 'game ROM')
tf.app.flags.DEFINE_string('roms_path', './roms/', 'game ROMs path')
tf.app.flags.DEFINE_integer('frame_skip', 4, 'every frame_skip frames to act')
tf.app.flags.DEFINE_integer('buffer_length', 2, 'screen buffer size for one image')
tf.app.flags.DEFINE_float('repeat_action_probability', 0, 'Probability that action choice will be ignored')
tf.app.flags.DEFINE_float('input_scale', 255.0, 'image rescale')
tf.app.flags.DEFINE_integer('input_width', 84, 'environment to agent image width')
tf.app.flags.DEFINE_integer('input_height', 84, 'environment to agent image width')
tf.app.flags.DEFINE_integer('num_actions', 2, 'environment accepts x actions')
tf.app.flags.DEFINE_integer('max_start_no_op', 30, 'Maximum number of null_ops at the start')
tf.app.flags.DEFINE_bool('lol_end', True, 'lost of life ends training episode')

# Agent settings
tf.app.flags.DEFINE_float('lr', 0.00025, 'learning rate')
tf.app.flags.DEFINE_float('discount', 0.99, 'discount rate')
tf.app.flags.DEFINE_float('ep_st', 1.0, 'epsilon start value')
tf.app.flags.DEFINE_float('ep_min', 0.1, 'epsilon minimum value')
tf.app.flags.DEFINE_float('ep_decay', 1000000, 'steps for epsilon reaching minimum')
tf.app.flags.DEFINE_integer('phi_length', 4, 'frames for representing a state')
tf.app.flags.DEFINE_integer('memory', 1000000, 'replay memory size')
tf.app.flags.DEFINE_integer('batch', 32, 'training batch size')
tf.app.flags.DEFINE_string('network', 'nature', 'neural network type')
tf.app.flags.DEFINE_integer('freeze', 10000, 'freeze interval between updates, update network every x trainings')
tf.app.flags.DEFINE_string('loss_func', 'huber', 'loss function: huber; quadratic')
tf.app.flags.DEFINE_string('optimizer', 'rmsprop', 'optimizer type')
tf.app.flags.DEFINE_integer('train_fr', 4, 'training frequency: train a batch every x steps')
tf.app.flags.DEFINE_integer('train_st', 50000, 'training start: training starts after x steps')
tf.app.flags.DEFINE_bool('clip_reward', True, 'clip reward to -1, 1')

# Multi threads settings
tf.app.flags.DEFINE_integer('threads', 4, 'CPU threads for agents')
tf.app.flags.DEFINE_bool('use_gpu', True, 'use GPUs')
tf.app.flags.DEFINE_integer('gpus', 4, 'number of GPUs for agents')
tf.app.flags.DEFINE_string('gpu_config',
                           """{'gpu0': [0], 'gpu1': [1], 'gpu2': [2], 'gpu3': [3]}""",
                           'GPU configuration for agents, default gpu0')
tf.app.flags.DEFINE_string('threads_specific_config',
                           """{0: {'rom': 'breakout'}, 1: {'rom': 'pong'}, 2: {'rom': 'qbert'},
                            3: {'rom': 'space_invaders'}}""",
                           'configuration for each agent')


def initialize(pid, device, flags, message_queue):
    message = 'initialize process: {:d} with GPU: {} game: {}'.format(pid, device, flags.rom)
    message_queue.put([-1, 'print', message])
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device[-1]
    np.random.seed(flags.seed)
    tf.set_random_seed(flags.seed)
    try:
        import ale_python_interface
    except ImportError:
        import atari_py.ale_python_interface as ale_python_interface

    # initialize ALE environment
    if flags.rom.endswith('.bin'):
        rom = flags.rom
    else:
        rom = "%s.bin" % flags.rom
    full_rom_path = os.path.join(flags.roms_path, rom)
    ale = ale_python_interface.ALEInterface()
    ale.setInt('random_seed', flags.seed)
    ale.setBool('sound', False)
    ale.setBool('display_screen', False)
    ale.setFloat('repeat_action_probability', flags.repeat_action_probability)
    ale.loadROM(full_rom_path)
    num_actions = len(ale.getMinimalActionSet())
    flags.num_actions = num_actions
    flags.logs_path = os.path.join(flags.logs_path, '#' + str(pid) + '_' + flags.rom)

    # initialize agent
    network = neural_networks.DeepQNetwork(pid, flags, device)
    agent = agents.QLearning(pid, network, flags, message_queue)

    interaction.Interaction(pid, ale, agent, flags, message_queue).start()


def display_threads(message_dict):
    if not FLAGS.curses:
        one_line = '\r'
        for pid, element in message_dict.items():
            if pid == -1:
                print
                for message in element.get('print', []):
                    print message
            else:
                if 'step' in element:
                    total_steps = FLAGS.steps_per_epoch if element['step'][0] == 'TRAIN' else FLAGS.test_length
                    one_line += '#{:d}:{} E{:d} {:.1f}% '.format(
                        pid, element['step'][0], element['step'][1], (1.0 - float(element['step'][2])/total_steps) * 100)
                if 'speed' in element:
                    one_line += '  St/Sec: cur:{:d} avg:{:d}'.format(element['speed'][0], element['speed'][1])
        sys.stdout.write(one_line)
        sys.stdout.flush()
        return
    # for id, element in message_dict.items():
    #     if id == -1:
    #         for message in element.get('print', []):
    #             stdscr.addstr(print_line_number, 0, message)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.logs_path):
        tf.gfile.DeleteRecursively(FLAGS.logs_path)
    pid_device = {}
    d = eval(FLAGS.gpu_config)
    for device, pids in d.items():
        for pid in pids:
            pid_device[pid] = device

    """
    [pid, 'step', [testing, epoch, steps_left]]
    [pid, 'speed', [current, avg]]
    [-1, 'print', message]
    """
    message_queue = mp.Queue()

    threads_specific_config = eval(FLAGS.threads_specific_config)
    processes = []
    for pid in xrange(FLAGS.threads):
        flags = copy.deepcopy(FLAGS)
        for key, val in threads_specific_config.get(pid, {}).items():
            setattr(flags, key, val)
        process = mp.Process(target=initialize, args=(pid, pid_device.get(pid, "gpu0")[-1], flags, message_queue))
        process.daemon = True
        process.start()
        processes.append(process)

    while any(p.is_alive() for p in processes):
        time.sleep(5.0)
        message_dict = {}
        while not message_queue.empty():
            """
            {pid: {'step': [,,], 'speed': [,]}, -1: {'print': [m1, m2, ...]}}
            """
            pid, key, message = message_queue.get()
            element = message_dict.setdefault(pid, {})
            if key == 'step' or key == 'speed':
                element[key] = message
            if key == 'print':
                element.setdefault(key, []).append(message)
        if message_dict:  # not empty
            display_threads(message_dict)


if __name__ == '__main__':
    tf.app.run()
