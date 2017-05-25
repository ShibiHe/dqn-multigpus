__author__ = 'frankhe'
import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import os


class DeepQNetwork(object):
    def __init__(self, pid, flags, device):
        self.pid = pid
        self.nn_structure_file = ''
        self.flags = flags
        if not flags.use_gpu:
            device = '/cpu:0'
        else:
            device = '/gpu:' + device

        with tf.device(device):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            # inference graph-----------------------
            with tf.variable_scope('current'):
                self.nn_structure_file += 'CURRENT:\n'
                self.images = tf.placeholder(tf.float32, [None, flags.phi_length, flags.input_height, flags.input_width],
                                             name='images')
                self.action_values_given_state = self._inference(self.images)
            with tf.variable_scope('old'):
                self.nn_structure_file += 'OLD:\n'
                self.images_old = tf.placeholder(tf.float32, [None, flags.phi_length, flags.input_height, flags.input_width],
                                                 name='images')
                self.action_values_given_state_old = self._inference(self.images_old)

            # optimizer-----------------------------
            self._construct_optimizer()
            # loss and train graph------------------
            self._construct_training_graph()
            # update old network--------------------
            self._construct_copy_op()
            # summary ops---------------------------
            self._construct_summary_ops()

        config = tf.ConfigProto()
        config.log_device_placement = False
        if flags.use_gpu:
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current'))
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        self.update_network()
        self.summary_writer = tf.summary.FileWriter(flags.logs_path, self.sess.graph)

    def _construct_optimizer(self):
        self.opt = None
        if self.flags.optimizer == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(self.flags.lr, decay=0.95, epsilon=0.01)
        if self.flags.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(self.flags.lr, epsilon=0.01)
        assert self.opt is not None

    def _construct_training_graph(self):
        with tf.name_scope('training_input'):
            self.actions = tf.placeholder(tf.int32, (None,), 'actions')
            self.rewards = tf.placeholder(tf.float32, (None,), 'rewards')
            self.terminals = tf.placeholder(tf.bool, (None,), 'terminal')
        discount = tf.constant(self.flags.discount, tf.float32, [], 'discount', True)
        with tf.name_scope('diff'):
            targets = self.rewards + (1.0 - tf.cast(self.terminals, tf.float32)) * discount * \
                                     tf.reduce_max(self.action_values_given_state_old, axis=1)
            targets = tf.stop_gradient(targets)
            actions = tf.one_hot(self.actions, self.flags.num_actions, axis=-1, dtype=tf.float32)
            q_s_a = tf.reduce_sum(self.action_values_given_state * actions, axis=1)
            diff = q_s_a - targets
        with tf.name_scope('loss'):
            loss = None
            if self.flags.loss_func == 'quadratic':
                loss = 0.5 * tf.square(diff)
            if self.flags.loss_func == 'huber':
                quad = tf.minimum(tf.abs(diff), 1.0)
                linear = tf.abs(diff) - quad
                loss = 0.5 * tf.square(quad) + linear
            assert loss is not None
            self.loss = tf.reduce_sum(loss)
        self.grad_var_list = self.opt.compute_gradients(self.loss)
        self.apply_gradients = self.opt.apply_gradients(self.grad_var_list, self.global_step)

    def _construct_copy_op(self):
        with tf.name_scope('copy'):
            assign_ops = []
            for (cur, old) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old')):
                assert cur.name[7:] == old.name[3:]
                assign_ops.append(tf.assign(old, cur))
            self.copy_cur2old_op = tf.group(*assign_ops)

    def _construct_summary_ops(self):
        # summaries of testing and global values
        with tf.name_scope('summaries_for_agent'):
            self.epoch_time = tf.placeholder(tf.float32, [])
            tf.add_to_collection('summaries_per_epoch', tf.summary.scalar('TIME_epoch', self.epoch_time))
            self.state_action_avg_val = tf.placeholder(tf.float32, [])
            tf.add_to_collection('summaries_per_epoch', tf.summary.scalar('mean q value', self.state_action_avg_val))
            self.total_reward = tf.placeholder(tf.float32, [])
            tf.add_to_collection('summaries_per_epoch', tf.summary.scalar('TOTAL_REWARD__test', self.total_reward))
            self.reward_per_episode = tf.placeholder(tf.float32, [])
            tf.add_to_collection('summaries_per_epoch', tf.summary.scalar('REWARD_per_EPISODE__test', self.reward_per_episode))
            self.episode_avg_loss = tf.placeholder(tf.float32, [])
            tf.add_to_collection('summaries_per_episode', tf.summary.scalar('EPISODE_AVG_LOSS', self.episode_avg_loss))

        # image summaries for current and target images
        tf.add_to_collection('training_summaries',
                             tf.summary.image('current images', tf.expand_dims(self.images[0], -1), 4))
        tf.add_to_collection('training_summaries',
                             tf.summary.image('target images', tf.expand_dims(self.images_old[0], -1), 4))

        # Add histograms for trainable variables under current scope
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current'):
            tf.add_to_collection('training_summaries', tf.summary.histogram(var.name, var))

        # Add histograms for gradients of trainable variables
        for grad, var in self.grad_var_list:
            if grad is not None:
                tf.add_to_collection('training_summaries', tf.summary.histogram(var.name + '/gradients', grad))

        # add scalar for training loss
        tf.add_to_collection('training_summaries', tf.summary.scalar(self.loss.name + '/raw_loss', self.loss))

        # activation summaries already added during inference construction

        self.summary_per_epoch_op = tf.summary.merge(tf.get_collection('summaries_per_epoch'), name='summary_per_epoch_op')
        self.summary_per_episode_op = tf.summary.merge(tf.get_collection('summaries_per_episode'), name='summary_per_episode_op')
        self.training_summary_op = tf.summary.merge(tf.get_collection('training_summaries'), name='training_summaries')

    def _activation_summary(self, x):
        tf.add_to_collection('training_summaries', tf.summary.histogram(x.name + '/activations', x))
        tf.add_to_collection('training_summaries', tf.summary.scalar(x.name + '/sparsity', tf.nn.zero_fraction(x)))

    def _conv_layer(self, conv_in, size, channels, filters, stride, padding='SAME', data_format='NCHW'):
        kernel = tf.get_variable('weights', [size, size, channels, filters], dtype=tf.float32,
                                 initializer=tfc.layers.variance_scaling_initializer(uniform=True))
        conv = tf.nn.conv2d(conv_in, kernel, [1, 1, stride, stride], padding=padding, data_format=data_format)
        bias = tf.get_variable('bias', [filters], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        pre_activations = tf.nn.bias_add(conv, bias, data_format=data_format)
        conv_out = tf.nn.relu(pre_activations)
        self._activation_summary(conv_out)
        s = "CONV In: {!s:20} Out: {!s:20} W: {!s:16} N_dim:{:8d}\tW_dim:{:8d}\n".format(
            conv_in.get_shape(), conv_out.get_shape(), kernel.get_shape(),
            reduce(lambda x, y: x * y, conv_in.get_shape().as_list()[1:]),
            reduce(lambda x, y: x * y, kernel.get_shape().as_list()))
        self.nn_structure_file += s
        return conv_out

    def _pool_layer(self, pool_in, size=2, stride=2):
        pool_out = tf.nn.max_pool(pool_in, ksize=[1, 1, size, size], strides=[1, 1, stride, stride],
                                  padding='SAME', data_format='NCHW')
        self._activation_summary(pool_out)
        s = "POOL In: {!s:20} Out: {!s:40} N_dim:{:8d}\n".format(
            pool_in.get_shape(), pool_out.get_shape(),
            reduce(lambda x, y: x * y, pool_in.get_shape().as_list()[1:]))
        self.nn_structure_file += s
        return pool_out

    def _linear_layer(self, linear_in, dim, hiddens):
        weights = tf.get_variable('weights', [dim, hiddens], tf.float32,
                                  initializer=tfc.layers.variance_scaling_initializer(uniform=True))
        bias = tf.get_variable('bias', [hiddens], tf.float32,
                               initializer=tf.constant_initializer(0.1))
        pre_activations = tf.add(tf.matmul(linear_in, weights), bias)
        linear_out = tf.nn.relu(pre_activations)
        self._activation_summary(linear_out)
        s = "LINR In: {!s:20} Out: {!s:20} W: {!s:16} N_dim:{:8d}\tW_dim:{:8d}\n".format(
            linear_in.get_shape(), linear_out.get_shape(), weights.get_shape(),
            reduce(lambda x, y: x * y, linear_in.get_shape().as_list()[1:]),
            reduce(lambda x, y: x * y, weights.get_shape().as_list()))
        self.nn_structure_file += s
        return linear_out

    def _inference(self, images):
        network_type = self.flags.network
        input_height = self.flags.input_height
        input_width = self.flags.input_width
        output_dim = self.flags.num_actions
        channels = self.flags.phi_length
        """
        images batch * channels * height * width
        :param input_width: 84 or 160
        :param input_height: 84 or 160
        :param output_dim: num_actions
        :param channels: phi_length
        :return: inference layer
        """
        images = images / self.flags.input_scale
        action_values = None
        if network_type == 'linear':
            with tf.variable_scope('linear'):
                images = tf.reshape(images, (-1, channels * input_height * input_width))
                dim = images.get_shape()[1].value
                weights = tf.get_variable('weights', shape=(dim, output_dim), dtype=tf.float32,
                                          initializer=tfc.layers.variance_scaling_initializer(1.0))
                bias = tf.get_variable('bias', shape=(output_dim,), dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                action_values = tf.add(tf.matmul(images, weights), bias)
                self._activation_summary(action_values)
        if network_type == 'nature':
            with tf.variable_scope('conv1'):
                size = 8 ; channels = channels ; filters = 32 ; stride = 4
                conv1 = self._conv_layer(images, size, channels, filters, stride, 'VALID')
            with tf.variable_scope('conv2'):
                size = 4 ; channels = 32 ; filters = 64 ; stride = 2
                conv2 = self._conv_layer(conv1, size, channels, filters, stride, 'VALID')
            with tf.variable_scope('conv3'):
                size = 3 ; channels = 64 ; filters = 64 ; stride = 1
                conv3 = self._conv_layer(conv2, size, channels, filters, stride, 'VALID')
            conv3_shape = conv3.get_shape().as_list()
            with tf.variable_scope('linear1'):
                hiddens = 512 ; dim = conv3_shape[1] * conv3_shape[2] * conv3_shape[3]
                reshape = tf.reshape(conv3, [-1, dim])
                linear1 = self._linear_layer(reshape, dim, hiddens)
            with tf.variable_scope('linear2'):
                hiddens = output_dim ; dim = 512
                action_values = self._linear_layer(linear1, dim, hiddens)
                self._activation_summary(action_values)
        if network_type == 'vgg':
            with tf.variable_scope('conv1'):
                size = 7; channels = channels; filters = 16; stride = 3
                conv1 = self._conv_layer(images, size, channels, filters, stride)
            with tf.variable_scope('conv1_2'):
                size = 3; channels = 16; filters = 32; stride = 1
                conv1_2 = self._conv_layer(conv1, size, channels, filters, stride)
            with tf.variable_scope('pool1'):
                pool1 = self._pool_layer(conv1_2)
            with tf.variable_scope('conv2'):
                size = 3; channels = 32; filters = 64;  stride = 1
                conv2 = self._conv_layer(pool1, size, channels, filters, stride)
            with tf.variable_scope('conv2_2'):
                size = 3; channels = 64; filters = 64;  stride = 1
                conv2_2 = self._conv_layer(conv2, size, channels,filters, stride)
            with tf.variable_scope('pool2'):
                pool2 = self._pool_layer(conv2_2)
            with tf.variable_scope('conv3'):
                size = 3; channels = 64; filters = 64;  stride = 1
                conv3 = self._conv_layer(pool2, size, channels, filters, stride)
            with tf.variable_scope('pool3'):
                pool3 = self._pool_layer(conv3)
            pool_shape = pool3.get_shape().as_list()
            with tf.variable_scope('linear1'):
                hiddens = 512 ; dim = pool_shape[1] * pool_shape[2] * pool_shape[3]
                reshape = tf.reshape(pool3, [-1, dim])
                linear1 = self._linear_layer(reshape, dim, hiddens)
            with tf.variable_scope('linear2'):
                hiddens = output_dim ; dim = 512
                action_values = self._linear_layer(linear1, dim, hiddens)
                self._activation_summary(action_values)
        assert action_values is not None
        return action_values

    def train(self, current_states, target_states, rewards, actions, terminals):
        _, loss, global_step = self.sess.run([self.apply_gradients, self.loss, self.global_step],
                                             feed_dict={self.images: current_states,
                                                        self.images_old: target_states,
                                                        self.actions: actions,
                                                        self.rewards: rewards,
                                                        self.terminals: terminals})
        if global_step % self.flags.summary_fr == 0:
            summary = self.sess.run(self.training_summary_op, feed_dict={self.images: current_states,
                                                                         self.images_old: target_states,
                                                                         self.actions: actions,
                                                                         self.rewards: rewards,
                                                                         self.terminals: terminals})
            self.summary_writer.add_summary(summary, global_step)
        if global_step % self.flags.freeze == 0:
            self.update_network()
        return loss

    def update_network(self):
        self.sess.run(self.copy_cur2old_op)

    def get_action_values(self, states):
        return self.sess.run(self.action_values_given_state, feed_dict={self.images: states})

    def get_action_values_old(self, states):
        return self.sess.run(self.action_values_given_state_old, feed_dict={self.images_old: states})

    def choose_action(self, phi):
        phi = np.expand_dims(phi, axis=0)
        action_values = self.get_action_values(phi)
        action = np.argmax(action_values, axis=1)[0]
        return action

    def epoch_summary(self, epoch, epoch_time, mean_q, total_reward, reward_per_ep):
        """
            self.epoch_time
            self.state_action_avg_val = tf.placeholder(tf.float32, [])
            self.total_reward = tf.placeholder(tf.float32, [])
            self.reward_per_episode = tf.placeholder(tf.float32, [])
        """
        summary, step = self.sess.run([self.summary_per_epoch_op, self.global_step],
                                      feed_dict={self.epoch_time: epoch_time,
                                                 self.state_action_avg_val: mean_q,
                                                 self.total_reward: total_reward,
                                                 self.reward_per_episode: reward_per_ep})
        self.summary_writer.add_summary(summary, epoch)

    def epoch_model_save(self, epoch):
        ckpt_path = os.path.join(self.flags.logs_path, 'model.ckpt')
        self.saver.save(self.sess, ckpt_path, epoch)

    def episode_summary(self, episode_avg_loss):
        summary, step = self.sess.run([self.summary_per_episode_op, self.global_step],
                                      feed_dict={self.episode_avg_loss: episode_avg_loss})
        self.summary_writer.add_summary(summary, step)


class OptimalityTighteningNetwork(DeepQNetwork):
    def __init__(self, pid, flags, device):
        super(OptimalityTighteningNetwork, self).__init__(pid, flags, device)

    def _construct_training_graph(self):
        with tf.variable_scope('q_s_a'):
            self.actions = tf.placeholder(tf.int32, (None,), 'actions')
            actions = tf.one_hot(self.actions, self.flags.num_actions, axis=-1, dtype=tf.float32)
            self.q_s_a = tf.reduce_sum(self.action_values_given_state * actions, axis=1)
        with tf.variable_scope('diff'):
            self.targets = tf.placeholder(tf.float32, (None, ), 'targets')
            diff = self.q_s_a - self.targets
        with tf.name_scope('loss'):
            loss = None
            if self.flags.loss_func == 'quadratic':
                loss = 0.5 * tf.square(diff)
            if self.flags.loss_func == 'huber':
                quad = tf.minimum(tf.abs(diff), 1.0)
                linear = tf.abs(diff) - quad
                loss = 0.5 * tf.square(quad) + linear
            assert loss is not None
            self.loss = tf.reduce_sum(loss)
        self.grad_var_list = self.opt.compute_gradients(self.loss)
        self.apply_gradients = self.opt.apply_gradients(self.grad_var_list, self.global_step)

    def train(self, current_states, actions, targets, **kwargs):
        _, loss, global_step = self.sess.run([self.apply_gradients, self.loss, self.global_step],
                                             feed_dict={self.images: current_states,
                                                        self.actions: actions,
                                                        self.targets: targets})
        if global_step % self.flags.summary_fr == 0:
            summary = self.sess.run(self.training_summary_op, feed_dict={self.images: current_states,
                                                                         self.actions: actions,
                                                                         self.targets: targets})
            self.summary_writer.add_summary(summary, global_step)
        if global_step % self.flags.freeze == 0:
            self.update_network()
        return loss

    def get_action_values_given_actions(self, states, actions):
        return self.sess.run(self.q_s_a, feed_dict={self.images: states, self.actions: actions})

if __name__ == '__main__':
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    my_flags = AttrDict(input_height=84, input_width=84, num_actions=6, phi_length=4, use_gpu=False, network='linear',
                        logs_path='./logs', discount=0.99, loss_func='quadratic', optimizer='rmsprop', lr=0.1,
                        input_scale=2, freeze=10, summary_fr=1)
    dqn = DeepQNetwork(1, my_flags, '0')
    # phi = np.random.randint(0, 10, (my_flags.phi_length, my_flags.input_height, my_flags.input_width), dtype='uint8')
    # print dqn.choose_action(phi)

    bs = 2
    states1 = np.random.randint(0, 10, (bs, my_flags.phi_length, my_flags.input_height, my_flags.input_width), dtype='uint8')
    states2 = np.random.randint(0, 10, (bs, my_flags.phi_length, my_flags.input_height, my_flags.input_width), dtype='uint8')
    phi = np.random.randint(0, 10, (my_flags.phi_length, my_flags.input_height, my_flags.input_width), dtype='uint8')
    actions = np.random.randint(0, my_flags.num_actions, (bs,))
    rewards = np.random.randint(0, 10, (bs,))
    terminals = np.random.randint(0, 1, (bs,), dtype=np.bool_)

    print dqn.get_action_values(states1)
    print dqn.get_action_values_old(states2)
    print actions
    print rewards
    print terminals

    # print dqn.train(states1, states2, rewards, actions, terminals)
    print dqn.get_action_values(np.expand_dims(phi, 0))[0]
    print dqn.choose_action(phi)
