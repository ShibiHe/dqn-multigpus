__author__ = 'frankhe'
import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import threading
import os


class DeepQNetwork(object):
    def __init__(self, pid, flags, device):
        self.pid = pid
        self.nn_structure_file = ''
        self.flags = flags
        self.feeding_threads_num = flags.feeding_threads
        self.feeding_threads = []
        self.train_data_set = None
        self.training_started = False
        self.epm = None

        if not flags.use_gpu:
            device = '/cpu:0'
        else:
            device = '/gpu:' + device

        with tf.device(device):
            # global step
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            # feeding ops
            self._construct_feeding_op()

            # inference graph-----------------------
            self._construct_inference()

            # optimizer-----------------------------
            self._construct_optimizer()
            # get features
            self._construct_feature_extraction()
            # loss and train graph------------------
            self._construct_training_graph()
            # update old network--------------------
            self._construct_copy_op()
            # summary ops---------------------------
            self._construct_summary_ops()

    def init(self):
        config = tf.ConfigProto()
        config.log_device_placement = False
        if self.flags.use_gpu:
            config.gpu_options.allow_growth = False
            if self.flags.gpu_memory_fraction != 0.0:
                config.gpu_options.per_process_gpu_memory_fraction = self.flags.gpu_memory_fraction
            config.allow_soft_placement = True

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current'))
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        self.sess.run(self.copy_cur2old_op)
        self.summary_writer = tf.summary.FileWriter(self.flags.logs_path, self.sess.graph)
        self.coord = tf.train.Coordinator()
        return self.sess

    def _start_feeding_data(self):
        for i in xrange(self.feeding_threads_num):
            t = threading.Thread(target=self._feeding_thread_process, args=())
            t.setDaemon(True)
            self.feeding_threads.append(t)
            t.start()

    def stop_feeding(self):
        self.coord.request_stop()
        self.sess.run(self.q_close_op)
        self.coord.join(self.feeding_threads, stop_grace_period_secs=1.0)
        self.epm.stop_updating()
        self.sess.close()

    def _feeding_thread_process(self):
        while not self.coord.should_stop():
            imgs, actions, rewards, terminals, return_value = self.train_data_set.random_batch(
                self.flags.batch, get_return_value=True)
            if self.coord.should_stop():
                return
            try:
                self.sess.run(self.enqueue_op, feed_dict={self.images: imgs[:, :-1, ...],
                                                          self.images_old: imgs[:, 1:, ...],
                                                          self.actions: actions,
                                                          self.rewards: rewards,
                                                          self.return_value: return_value,
                                                          self.terminals: terminals})
            except tf.errors.CancelledError:
                return

    def _construct_feeding_op(self):
        """self.images: current_states, self.images_old: target_states, 
                    self.actions: actions, self.rewards: rewards, self.terminals: terminals}"""
        with tf.name_scope('training_input'):
            self.images = tf.placeholder(tf.float32,
                                         [None, self.flags.phi_length, self.flags.input_height, self.flags.input_width],
                                         name='images')
            self.images_old = tf.placeholder(tf.float32,
                                             [None, self.flags.phi_length, self.flags.input_height, self.flags.input_width],
                                             name='images_old')
            self.actions = tf.placeholder(tf.int32, [None])
            self.rewards = tf.placeholder(tf.float32, [None])
            self.return_value = tf.placeholder(tf.float32, [None])
            self.terminals = tf.placeholder(tf.bool, [None])
            self.queue = tf.FIFOQueue(capacity=self.flags.feeding_queue_size,
                                      dtypes=[tf.float32, tf.float32, tf.int32, tf.float32, tf.float32, tf.bool])
            self.enqueue_op = self.queue.enqueue(
                [self.images, self.images_old, self.actions, self.rewards, self.return_value, self.terminals])
            self.q_size_op = self.queue.size()
            self.q_close_op = self.queue.close(cancel_pending_enqueues=True)
            input_tensors = self.queue.dequeue()
            self.feed_images, self.feed_images_old, self.feed_actions, self.feed_rewards, self.feed_return_value, self.feed_terminals = input_tensors
            self.feed_images.set_shape([self.flags.batch, self.flags.phi_length, self.flags.input_height, self.flags.input_width])
            self.feed_images_old.set_shape([self.flags.batch, self.flags.phi_length, self.flags.input_height, self.flags.input_width])
            self.feed_actions.set_shape([self.flags.batch])
            self.feed_rewards.set_shape([self.flags.batch])
            self.feed_return_value.set_shape([self.flags.batch])
            self.feed_terminals.set_shape([self.flags.batch])

    def _construct_inference(self):
        with tf.variable_scope('current') as current_scope:
            self.nn_structure_file += 'CURRENT:\n'
            self.action_values_given_state = self._inference(self.images)
            current_scope.reuse_variables()
            self.feed_action_values_given_state = self._inference(self.feed_images)
            self._activation_summary(self.feed_action_values_given_state)
            # current_scope.reuse_variables()
            # self.feed_double_dqn_given_state_old = self._inference(self.feed_images_old)
        with tf.variable_scope('old') as old_scope:
            self.nn_structure_file += 'OLD:\n'
            self.action_values_given_state_old = self._inference(self.images_old)
            old_scope.reuse_variables()
            self.feed_action_values_given_state_old = self._inference(self.feed_images_old)
            self._activation_summary(self.feed_action_values_given_state_old)

    def _construct_optimizer(self):
        with tf.name_scope('learning_rate_decay'):
            decay = tf.constant(
                (self.flags.lr_min - self.flags.lr) / (self.flags.lr_decay_b - self.flags.lr_decay_a), dtype=tf.float32)
            self.learning_rate = tf.case(
                [(tf.less(self.global_step, self.flags.lr_decay_a), lambda: self.flags.lr),
                 (tf.less(self.global_step, self.flags.lr_decay_b),
                    lambda: self.flags.lr + (self.global_step - self.flags.lr_decay_a) * decay),
                 (tf.greater_equal(self.global_step, self.flags.lr_decay_b), lambda: self.flags.lr_min)],
                default=lambda: self.flags.lr)
        self.opt = None
        if self.flags.optimizer == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.95, epsilon=0.01)
        if self.flags.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate, beta2=0.99, epsilon=0.0001)
        assert self.opt is not None

    def _construct_feature_extraction(self):
        features_list = tf.get_collection('features', scope='old/')
        self.features = features_list[0]
        self.feed_features = features_list[1]  # this is from data pipeline during training
        assert self.feed_features.shape == [self.flags.batch, 512]

    def _construct_training_graph(self):
        discount = tf.constant(self.flags.discount, tf.float32, [], 'discount', True)
        with tf.name_scope('diff'):
            # double_actions = tf.one_hot(tf.argmax(self.feed_double_dqn_given_state_old, axis=1),
            #                             self.flags.num_actions, axis=-1, dtype=tf.float32)
            targets = self.feed_rewards + (1.0 - tf.cast(self.feed_terminals, tf.float32)) * discount * \
                                          tf.reduce_max(self.feed_action_values_given_state_old, axis=1)
            targets = tf.stop_gradient(targets)
            actions = tf.one_hot(self.feed_actions, self.flags.num_actions, axis=-1, dtype=tf.float32)
            q_s_a = tf.reduce_sum(self.feed_action_values_given_state * actions, axis=1)
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
                assert cur.name[cur.name.rfind('/') + 1:] == old.name[old.name.rfind('/') + 1:]
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
        # tf.add_to_collection('training_summaries',
        #                      tf.summary.image('current images', tf.expand_dims(self.feed_images[0], -1), 4))
        # tf.add_to_collection('training_summaries',
        #                      tf.summary.image('target images', tf.expand_dims(self.feed_images_old[0], -1), 4))

        # Add histograms for trainable variables under current scope
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current'):
            tf.add_to_collection('training_summaries', tf.summary.histogram(var.name, var))

        # Add histograms for gradients of trainable variables
        for grad, var in self.grad_var_list:
            if grad is not None:
                tf.add_to_collection('training_summaries', tf.summary.histogram(var.name + '/gradients', grad))

        # add histograms for feature vector
        self._activation_summary(self.feed_features)

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
        # self._activation_summary(conv_out)
        s = "CONV In: {!s:20} Out: {!s:20} W: {!s:16} N_dim:{:8d}\tW_dim:{:8d}\n".format(
            conv_in.get_shape(), conv_out.get_shape(), kernel.get_shape(),
            reduce(lambda x, y: x * y, conv_in.get_shape().as_list()[1:]),
            reduce(lambda x, y: x * y, kernel.get_shape().as_list()))
        self.nn_structure_file += s
        return conv_out

    def _pool_layer(self, pool_in, size=2, stride=2):
        pool_out = tf.nn.max_pool(pool_in, ksize=[1, 1, size, size], strides=[1, 1, stride, stride],
                                  padding='SAME', data_format='NCHW')
        # self._activation_summary(pool_out)
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
        # self._activation_summary(linear_out)
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
        feature_dim = self.flags.feature_dim
        channels = self.flags.phi_length
        """
        images batch * channels * height * width
        :param input_width: 84 or 128 160
        :param input_height: 84 or 128 160
        :param output_dim: num_actions
        :param channels: phi_length
        :return: inference layer
        """
        images = images / self.flags.input_scale
        action_values = None
        if network_type == 'linear':
            images = tf.reshape(images, (-1, channels * input_height * input_width))
            dim = images.get_shape()[1].value
            with tf.variable_scope('linear1'):
                linear1 = self._linear_layer(images, dim, feature_dim)
                tf.add_to_collection('features', linear1)
            with tf.variable_scope('linear2'):
                action_values = self._linear_layer(linear1, feature_dim, output_dim)
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
                hiddens = feature_dim ; dim = conv3_shape[1] * conv3_shape[2] * conv3_shape[3]
                reshape = tf.reshape(conv3, [-1, dim])
                linear1 = self._linear_layer(reshape, dim, hiddens)
                tf.add_to_collection('features', linear1)
            with tf.variable_scope('linear2'):
                hiddens = output_dim ; dim = feature_dim
                action_values = self._linear_layer(linear1, dim, hiddens)
        if network_type == 'vgg':
            with tf.variable_scope('conv1'):
                size = 7; channels = channels; filters = 16; stride = 4
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
                size = 3; channels = 64; filters = 128;  stride = 1
                conv3 = self._conv_layer(pool2, size, channels, filters, stride)
            with tf.variable_scope('pool3'):
                pool3 = self._pool_layer(conv3)
            pool_shape = pool3.get_shape().as_list()
            with tf.variable_scope('linear1'):
                hiddens = feature_dim ; dim = pool_shape[1] * pool_shape[2] * pool_shape[3]
                reshape = tf.reshape(pool3, [-1, dim])
                linear1 = self._linear_layer(reshape, dim, hiddens)
                tf.add_to_collection('features', linear1)
            with tf.variable_scope('linear2'):
                hiddens = output_dim ; dim = feature_dim
                action_values = self._linear_layer(linear1, dim, hiddens)
        assert action_values is not None
        return action_values

    def get_features(self, states):
        return self.sess.run(self.features, feed_dict={self.images_old: states})

    def add_train_data_set(self, data_set):
        self.train_data_set = data_set

    def train(self):
        if not self.training_started:  # first time training
            self._start_feeding_data()
            self.epm.start_updating_memory()
            self.training_started = True
        _, loss, global_step = self.sess.run([self.apply_gradients, self.loss, self.global_step])
        if global_step % self.flags.summary_fr == 0:
            summary = self.sess.run(self.training_summary_op)
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

    def _construct_feeding_op(self):
        """self.images: current_states, self.images_old: target_states, 
                    self.actions: actions, self.rewards: rewards, self.terminals: terminals}"""
        self.images = tf.placeholder(tf.float32,
                                     [None, self.flags.phi_length, self.flags.input_height, self.flags.input_width],
                                     name='images')
        self.images_old = tf.placeholder(tf.float32,
                                         [None, self.flags.phi_length, self.flags.input_height, self.flags.input_width],
                                         name='images_old')
        with tf.name_scope('training_input'):
            self.center_positions = tf.placeholder(tf.int32, (None,))
            self.center_terminals = tf.placeholder(tf.bool, (None,))
            self.forward_positions = tf.placeholder(tf.int32, (None, self.flags.nob))
            self.backward_positions = tf.placeholder(tf.int32, (None, self.flags.nob))
            self.center_images = tf.placeholder(
                tf.float32,
                (None, self.flags.phi_length, self.flags.input_height, self.flags.input_width),
                name='center_images')
            self.forward_images = tf.placeholder(
                tf.float32,
                (None, self.flags.nob, self.flags.phi_length, self.flags.input_height, self.flags.input_width),
                name='forward_images')
            self.backward_images = tf.placeholder(
                tf.float32,
                (None, self.flags.nob, self.flags.phi_length, self.flags.input_height, self.flags.input_width),
                name='backward_images')
            self.center_actions = tf.placeholder(tf.int32, (None,))
            self.backward_actions = tf.placeholder(tf.int32, (None, self.flags.nob))
            self.center_return_values = tf.placeholder(tf.float32, (None,))
            self.forward_return_values = tf.placeholder(tf.float32, (None, self.flags.nob))
            self.backward_return_values = tf.placeholder(tf.float32, (None, self.flags.nob))
            self.forward_discounts = tf.placeholder(tf.float32, (None, self.flags.nob))
            self.backward_discounts = tf.placeholder(tf.float32, (None, self.flags.nob))

            self.queue = tf.FIFOQueue(
                capacity=self.flags.feeding_queue_size,
                dtypes=[tf.int32, tf.bool, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32,
                        tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            self.enqueue_op = self.queue.enqueue(
                [self.center_positions, self.center_terminals, self.forward_positions, self.backward_positions,
                 self.center_images, self.forward_images, self.backward_images, self.center_actions,
                 self.backward_actions, self.center_return_values, self.forward_return_values,
                 self.backward_return_values, self.forward_discounts, self.backward_discounts])
            self.q_size_op = self.queue.size()
            self.q_close_op = self.queue.close(cancel_pending_enqueues=True)
            input_tensors = self.queue.dequeue()
            self.feed_center_positions, self.feed_center_terminals, self.feed_forward_positions,\
                self.feed_backward_positions, self.feed_center_images, self.feed_forward_images,\
                self.feed_backward_images, self.feed_center_actions, self.feed_backward_actions,\
                self.feed_center_return_values, self.feed_forward_return_values, self.feed_backward_return_values,\
                self.feed_forward_discounts, self.feed_backward_discounts = input_tensors

            self.feed_center_positions.set_shape((self.flags.batch,))
            self.feed_center_terminals.set_shape((self.flags.batch,))
            self.feed_forward_positions.set_shape((self.flags.batch, self.flags.nob))
            self.feed_backward_positions.set_shape((self.flags.batch, self.flags.nob))
            self.feed_center_images.set_shape(
                (self.flags.batch, self.flags.phi_length, self.flags.input_height, self.flags.input_width))
            self.feed_forward_images.set_shape(
                (self.flags.batch, self.flags.nob, self.flags.phi_length, self.flags.input_height,
                 self.flags.input_width))
            self.feed_backward_images.set_shape(
                (self.flags.batch, self.flags.nob, self.flags.phi_length, self.flags.input_height,
                 self.flags.input_width))
            self.feed_center_actions.set_shape((self.flags.batch,))
            self.feed_backward_actions.set_shape((self.flags.batch, self.flags.nob))
            self.feed_center_return_values.set_shape((self.flags.batch,))
            self.feed_forward_return_values.set_shape((self.flags.batch, self.flags.nob))
            self.feed_backward_return_values.set_shape((self.flags.batch, self.flags.nob))
            self.feed_forward_discounts.set_shape((self.flags.batch, self.flags.nob))
            self.feed_backward_discounts.set_shape((self.flags.batch, self.flags.nob))

    def _feeding_thread_process(self):
        while not self.coord.should_stop():
            if self.flags.close2:
                self.train_data_set.random_batch_with_close_bounds(self.flags.batch)
            else:
                pass  # not close bounds
            if self.coord.should_stop():
                return
            try:
                self.sess.run(self.enqueue_op,
                              feed_dict={self.center_positions: self.train_data_set.center_positions,
                                         self.center_terminals: self.train_data_set.center_terminals,
                                         self.forward_positions: self.train_data_set.forward_positions,
                                         self.backward_positions: self.train_data_set.backward_positions,
                                         self.center_images: self.train_data_set.center_imgs,
                                         self.forward_images: self.train_data_set.forward_imgs,
                                         self.backward_images: self.train_data_set.backward_imgs,
                                         self.center_actions: self.train_data_set.center_actions,
                                         self.backward_actions: self.train_data_set.backward_actions,
                                         self.center_return_values: self.train_data_set.center_return_values,
                                         self.forward_return_values: self.train_data_set.forward_return_values,
                                         self.backward_return_values: self.train_data_set.backward_return_values,
                                         self.forward_discounts: self.train_data_set.forward_discounts,
                                         self.backward_discounts: self.train_data_set.backward_discounts})
            except tf.errors.CancelledError:
                return

    def _construct_inference(self):
        with tf.variable_scope('current') as current_scope:
            self.nn_structure_file += 'CURRENT:\n'
            self.action_values_given_state = self._inference(self.images)
            current_scope.reuse_variables()
            feed_q_s_a_values = self._inference(self.feed_center_images)
            # feed_q_s_a_values: (N * A)
            actions = tf.one_hot(self.feed_center_actions, self.flags.num_actions, axis=-1, dtype=tf.float32)
            self.feed_q_values = tf.reduce_sum(feed_q_s_a_values * actions, axis=1)
            # self.feed_q_values: (N,)

        with tf.variable_scope('old') as old_scope:
            self.nn_structure_file += 'OLD:\n'
            self.action_values_given_state_old = self._inference(self.images_old)
            old_scope.reuse_variables()
            if self.flags.one_bound:
                self.feed_target_q_table = self._inference(
                    tf.reshape(self.feed_forward_images, [-1] + self.feed_forward_images.get_shape().as_list()[2:]))
                # feed_target_q_table: (N * nob, A)
            else:
                target_q_images = tf.concat([self.feed_forward_images, self.feed_backward_images], axis=1)  # N,2nob,4HW
                target_q_images = tf.reshape(target_q_images, [-1] + target_q_images.get_shape().as_list()[2:])
                self.feed_target_q_table = self._inference(target_q_images)  # N*2nob, A

    def _construct_training_graph(self):
        with tf.variable_scope('map_fn_input'):
            input_tensor = tf.concat([
                tf.reshape(self.feed_q_values, [self.flags.batch, -1]),  # N, 1
                tf.cast(tf.reshape(self.feed_center_positions, (self.flags.batch, -1)), tf.float32),  # N, 1
                tf.reshape(self.feed_center_return_values, (self.flags.batch, -1)),  # N, 1
                tf.cast(tf.reshape(self.feed_center_terminals, (self.flags.batch, -1)), tf.float32),  # N, 1
                tf.cast(tf.reshape(self.feed_backward_actions, [self.flags.batch, -1]), tf.float32),  # N, 1
                tf.cast(tf.reshape(self.feed_backward_positions, (self.flags.batch, -1)), tf.float32),  # N, nob
                tf.reshape(self.feed_forward_return_values, (self.flags.batch, -1)),  # N, nob
                tf.reshape(self.feed_forward_discounts, (self.flags.batch, -1)),  # N, nob
                tf.reshape(self.feed_backward_return_values, (self.flags.batch, -1)),  # N, nob
                tf.reshape(self.feed_backward_discounts, (self.flags.batch, -1)),  # N, nob
                tf.reshape(self.feed_target_q_table, [self.flags.batch, -1])  # N, nob * A or N, 2*nob*A
            ], axis=1)

        def train_body(x):
            """
            q_values[i]: x[0]; center_position: x[1]; center_return_values[i]: x[2]; center_terminals: x[3];
            backward_actions[i, j]: x[offset: offset + nob]
            backward_positions[i, j]: x[offset + nob: offset + 2*nob]
            forward_return_values[i, j]: x[offset + 2 * nob: offset+ 3 * nob]
            forward_discounts[i, j]: x[offset + 3 * nob: offset + 4 * nob]
            backward_return_values[i, j]: x[offset + 4 * nob: offset + 5 * nob]
            backward_discounts[i, j]: x[offset + 5 * nob: offset + 6 * nob]
            target_q_table x[offset + 6 * nob:]
            """
            offset = 4

            def f1_terminal():
                return x[2]

            def f2_not_terminal_two_bounds():
                forward_target_q_table = tf.reshape(
                    x[offset + 6 * self.flags.nob: offset + 6 * self.flags.nob + self.flags.num_actions * self.flags.nob],
                    [self.flags.nob, -1])  # nob, A
                backward_target_q_table = tf.reshape(
                    x[offset + 6 * self.flags.nob + self.flags.num_actions * self.flags.nob:],
                    [self.flags.nob, -1])  # nob, A

                forward_target_q_table = tf.reduce_max(forward_target_q_table, axis=1)  # nob
                backward_actions = tf.one_hot(tf.cast(x[offset: offset + self.flags.nob], tf.int32),
                                              depth=self.flags.num_actions, axis=-1, dtype=tf.float32)
                backward_target_q_table = tf.reduce_sum(backward_target_q_table * backward_actions, axis=1)  # nob

                forward_targets = x[2] - tf.multiply(x[offset + 2 * self.flags.nob: offset + 3 * self.flags.nob],
                                                     x[offset + 3 * self.flags.nob: offset + 4 * self.flags.nob]) + \
                                         tf.multiply(x[offset + 3 * self.flags.nob: offset + 4 * self.flags.nob],
                                                     forward_target_q_table)
                backward_targets = tf.divide(-x[offset + 4 * self.flags.nob: offset + 5 * self.flags.nob] +
                                             tf.multiply(x[offset + 5 * self.flags.nob: offset + 6 * self.flags.nob],
                                                         x[2]) + backward_target_q_table,
                                             x[offset + 5 * self.flags.nob: offset + 6 * self.flags.nob])
                backward_mask = tf.cast(tf.equal(x[offset + self.flags.nob: offset + 2 * self.flags.nob], x[1] + 1),
                                        tf.float32)
                backward_targets = backward_mask * x[0] + (1 - backward_mask) * backward_targets

                v0 = forward_targets[0]
                v_max = tf.maximum(tf.reduce_max(forward_targets[1:]), x[2])
                v_min = tf.reduce_min(backward_targets)

                v1 = tf.case({
                    tf.logical_and(tf.greater(x[0], v_min), tf.less(x[0], v_max)): lambda: (v_max + v_min) * 0.5,
                    tf.greater(v_max, x[0]): lambda: v_max,
                    tf.less(v_min, x[0]): lambda: v_min},
                    default=lambda: v0)
                return v0 * self.flags.pw + (1 - self.flags.pw) * v1

            def f2_not_terminal_one_bound():
                forward_target_q_table = tf.reshape(
                    x[offset + 6 * self.flags.nob: offset + 6 * self.flags.nob + self.flags.num_actions * self.flags.nob],
                    [self.flags.nob, -1])  # nob, A
                forward_target_q_table = tf.reduce_max(forward_target_q_table, axis=1)  # nob
                forward_targets = x[2] - tf.multiply(x[offset + 2 * self.flags.nob: offset + 3 * self.flags.nob],
                                                     x[offset + 3 * self.flags.nob: offset + 4 * self.flags.nob]) + \
                                         tf.multiply(x[offset + 3 * self.flags.nob: offset + 4 * self.flags.nob],
                                                     forward_target_q_table)
                v0 = forward_targets[0]
                v_max = tf.maximum(tf.reduce_max(forward_targets[1:]), x[2])
                return tf.cond(tf.greater(v_max, x[0]),
                               lambda: v0 * self.flags.pw + (1 - self.flags.pw) * v_max, lambda: v0)
            return tf.cond(tf.cast(x[3], tf.bool), f1_terminal,
                           f2_not_terminal_one_bound if self.flags.one_bound else f2_not_terminal_two_bounds)

        with tf.variable_scope('targets'):
            targets = tf.map_fn(train_body, input_tensor)
            self.targets = tf.stop_gradient(targets)
        with tf.variable_scope('diff'):
            diff = self.feed_q_values - self.targets
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
