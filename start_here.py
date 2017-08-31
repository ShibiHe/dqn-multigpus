__author__ = 'frankhe'
import sys
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import copy
import interaction
import neural_networks
import agents

FLAGS = tf.app.flags.FLAGS

# Experiment settings
tf.app.flags.DEFINE_integer('epochs', 50, 'Number of training epochs')
tf.app.flags.DEFINE_integer('steps_per_epoch', 250000, 'Number of steps per epoch')
tf.app.flags.DEFINE_integer('test_length', 125000, 'Number of steps per test')
tf.app.flags.DEFINE_integer('seed', 123456, 'random seed')
tf.app.flags.DEFINE_bool('diff_seed', True, 'enable different seed for each process')
tf.app.flags.DEFINE_integer('summary_fr', 3000, 'summary every x training steps')
tf.app.flags.DEFINE_string('logs_path', './logs', 'tensor board path')
tf.app.flags.DEFINE_bool('test', False, 'enable test mode')
tf.app.flags.DEFINE_bool('ckpt', False, 'enable save models')
tf.app.flags.DEFINE_integer('feeding_threads', 1, 'feeding data threads')
tf.app.flags.DEFINE_integer('feeding_queue_size', 50, 'feeding queue capacity')
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.3, 'gpu memory fraction')

# ALE Environment settings
tf.app.flags.DEFINE_string('rom', 'breakout', 'game ROM')
tf.app.flags.DEFINE_string('roms_path', './roms/', 'game ROMs path')
tf.app.flags.DEFINE_integer('frame_skip', 4, 'every frame_skip frames to act')
tf.app.flags.DEFINE_integer('buffer_length', 2, 'screen buffer size for one image')
tf.app.flags.DEFINE_float('repeat_action_probability', 0, 'Probability that action choice will be ignored')
tf.app.flags.DEFINE_float('input_scale', 255.0, 'image rescale')
tf.app.flags.DEFINE_integer('input_width', 84, 'environment to agent image width')  # 128 vgg
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
tf.app.flags.DEFINE_string('network', 'nature', 'neural network type, linear, nature, vgg')
tf.app.flags.DEFINE_integer('freeze', 2500, """freeze interval between updates, update network every x trainings. 
To be noticed, Nature paper is inconsistent with its code.""")
tf.app.flags.DEFINE_string('loss_func', 'huber', 'loss function: huber; quadratic')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer type')
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
tf.app.flags.DEFINE_string('threads_specific_config', "{}",
                           """{0: {'rom': 'breakout'}, 1: {'rom': 'pong'}, 2: {'rom': 'beam_rider'},
                            3: {'rom': 'space_invaders'}}      configuration for each agent""")

# optimality tightening
tf.app.flags.DEFINE_bool('ot', False, 'optimality tightening')
tf.app.flags.DEFINE_bool('close2', True, 'close bounds')
tf.app.flags.DEFINE_bool('one_bound', True, 'only use lower bounds')
tf.app.flags.DEFINE_integer('nob', 4, 'number of bounds')
tf.app.flags.DEFINE_float('pw', 0.8, 'penalty weight')


def initialize(pid, device, flags, comm, share_comm):
    message = 'initialize process: {:d} with GPU: {} game: {}'.format(pid, device, flags.rom)
    comm.send([-1, 'print', message], dest=flags.threads)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

    # adjust flags
    flags.num_actions = num_actions
    flags.logs_path = os.path.join(flags.logs_path, '#' + str(pid) + '_' + flags.rom)
    tf.gfile.MakeDirs(flags.logs_path)

    # print settings
    setting_file = open(os.path.join(flags.logs_path, 'flags.txt'), mode='w+')
    for key, item in flags.__flags.items():
        setting_file.write(key + ' : ' + str(item) + '\n')

    # initialize agent
    if flags.ot:
        network = neural_networks.OptimalityTighteningNetwork(pid, flags, device, share_comm)
    else:
        network = neural_networks.DeepQNetwork(pid, flags, device, share_comm)

    setting_file.write(network.nn_structure_file)
    setting_file.close()

    if flags.ot:
        agent = agents.OptimalityTigheningAgent(pid, network, flags, comm)
    else:
        agent = agents.QLearning(pid, network, flags, comm)
    interaction.Interaction(pid, ale, agent, flags, comm).start()


def display_threads(message_dict, flags=FLAGS):
    one_line = '\r\033[K'
    for pid, element in message_dict.items():
        if pid == -1:
            print
            for message in element.get('print', []):
                print message
        else:
            if 'step' in element:
                total_steps = flags.steps_per_epoch if element['step'][0] == 'TRAIN' else flags.test_length
                one_line += '  #{:d}:{} E{:d} {:.1f}% '.format(
                    pid, element['step'][0], element['step'][1],
                    (1.0 - float(element['step'][2]) / total_steps) * 100)
            if 'speed' in element:
                one_line += ' St/Sec: cur:{:d} avg:{:d} '.format(element['speed'][0], element['speed'][1])
    sys.stdout.write(one_line)
    sys.stdout.flush()
    return


def main(argv=None):
    # comm is used for message transmitting
    comm = MPI.COMM_WORLD
    pid = comm.Get_rank()

    pid_device = {}
    d = eval(FLAGS.gpu_config)
    for device, pids in d.items():
        for i in pids:
            pid_device[i] = device

    flags = copy.deepcopy(FLAGS)
    flags.seed += int(flags.diff_seed) * pid
    if flags.test:
        flags.threads = 2  # np=3
        flags.gpus = 2
        flags.epochs = 2
        flags.steps_per_epoch = 10000
        flags.test_length = 2000
        flags.summary_fr = 100
        flags.network = 'linear'
        flags.train_st = 2000
        flags.freeze = 100
        flags.ot = False
        flags.one_bound = True

    if pid == flags.threads:
        color = 0
    else:
        color = 1
    # share_comm is used for sharing parameters
    share_comm = MPI.COMM_WORLD.Split(color, pid)
    # print share_comm.Get_rank(), share_comm.Get_rank()

    if pid == flags.threads:
        # process=threads is the printer process and the main process
        if tf.gfile.Exists(FLAGS.logs_path):
            tf.gfile.DeleteRecursively(FLAGS.logs_path)
        comm.Barrier()
        if flags.logs_path == './logs':
            print 'WARNING: logs_path is not specified, default to ./logs'
        """
        [pid, 'step', [testing, epoch, steps_left]]
        [pid, 'speed', [current, avg]]
        [-1, 'print', message]
        """
        end_threads = np.zeros(flags.threads, dtype=np.bool_)
        while True:
            message_dict = {}
            for i in xrange(flags.threads * 12):
                if np.all(end_threads):
                    return
                pid, key, message = comm.recv(source=MPI.ANY_SOURCE)
                element = message_dict.setdefault(pid, {})
                if key == 'step' or key == 'speed':
                    element[key] = message
                if key == 'print':
                    element.setdefault(key, []).append(message)
                if key == 'END':
                    print '\n', pid, 'join',
                    end_threads[pid] = True
            if message_dict:  # not empty
                display_threads(message_dict)
    else:
        comm.Barrier()

    threads_specific_config = eval(flags.threads_specific_config)
    for key, val in threads_specific_config.get(pid, {}).items():
        setattr(flags, key, val)
    initialize(pid, pid_device.get(pid, "gpu0")[-1], flags, comm, share_comm)


if __name__ == '__main__':
    tf.app.run()
    # mpirun -np threads + 1 python start_here.py
    # mpirun -np 3 python start_here.py --test True


