__author__ = 'frankhe'
import numpy as np
import tensorflow as tf
import threading
import time


class EpisodicMemory(object):
    def __init__(self, flags, device):
        self.flags = flags
        self.original_height = flags.input_height
        self.original_width = flags.input_width
        self.width = flags.input_width / self.flags.shrink
        self.height = flags.input_height / self.flags.shrink
        self.train_data_set = None
        self.sess = None
        self.coord = tf.train.Coordinator()
        self.update_threads = []
        self.buffer_step = self.flags.buffer_step
        self.counter = 0

        # for test
        self.memory_update_file = None

        if not flags.epm_use_gpu:
            device = '/cpu:0'
        else:
            device = '/gpu:' + device

        with tf.device(device):
            with tf.variable_scope('Episodic'):
                # initialize hash tables
                self.table = tf.Variable(tf.zeros((self.flags.episodic_memory,
                                                   self.flags.buckets,
                                                   self.flags.num_actions,
                                                   self.flags.buckets + 1), dtype=tf.int32),
                                         trainable=False,
                                         name='hash_table',
                                         dtype=tf.int32)
                self.hash_table_non_zero = tf.reduce_sum(tf.cast(tf.not_equal(self.table[..., -1], 0), tf.int32))

                # initialize projection matrix
                self.projection_matrices = []
                for i in range(self.flags.buckets):
                    p_matrix = tf.Variable(tf.random_normal((self.width * self.height,
                                                             self.flags.hash_dim)),
                                           trainable=False,
                                           dtype=tf.float32,
                                           name='projection' + str(i))
                    self.projection_matrices.append(p_matrix)

                # initialize bit weights
                weights = []
                weight = 1
                for _ in range(self.flags.hash_dim):
                    weights.append(weight)
                    weight = (weight << 1) % self.flags.episodic_memory
                self.bit_weights = tf.constant(weights,
                                               dtype=tf.int32,
                                               shape=[self.flags.hash_dim],
                                               name='bit_weights',
                                               verify_shape=True)

                # compute keys
                self.states_input_batch = tf.placeholder(tf.uint8, [None, self.original_height, self.original_width])
                resized = tf.image.resize_images(
                    tf.cast(tf.expand_dims(self.states_input_batch, -1), tf.float32), [self.height, self.width])
                states_batch = tf.reshape(resized, [-1, self.height * self.width])
                keys_batch = []
                for i in range(self.flags.buckets):
                    # out: N * hash_dim   complexity: O(N*7056*hash_dim)
                    bit_keys = tf.cast(tf.sign(tf.matmul(states_batch, self.projection_matrices[i])), tf.int32)  # N,dim

                    # here we use _dot_product_mod to avoid overflow when prime number is large
                    # keys = tf.map_fn(self._dot_product_mod, bit_keys, back_prop=False)  # N
                    # normal matmul method
                    keys = tf.matmul(bit_keys, tf.reshape(self.bit_weights, [-1, 1])) % self.flags.episodic_memory
                    keys = tf.reshape(keys, [-1])

                    keys_batch.append(keys)
                self.batch_keys = tf.stack(keys_batch, axis=1, name='keys')  # N * bucket_size

                # update memory
                self.rewards = tf.placeholder(tf.float32, [None])
                self.actions = tf.placeholder(tf.int32, [None])
                rewards = tf.cast(self.rewards, tf.int32)
                actions = self.actions
                batch_key_values = tf.stack(keys_batch + [rewards, actions], axis=1, name='batch_update')  # N * (bucket_size + 2)
                indices = tf.map_fn(self._get_hash_table_indices, batch_key_values,
                                    back_prop=False,
                                    parallel_iterations=1024)  # N * bucket * 3
                values = batch_key_values[..., :-1]  # value in hash containing bucket keys and a reward
                values = tf.reshape(tf.tile(values, [1, self.flags.buckets]),
                                    [-1, self.flags.buckets, self.flags.buckets + 1])  # N * bucket * 3
                current_reward = values[..., -1]  # N, bucket
                old_reward = tf.reshape(tf.gather_nd(self.table, indices=tf.reshape(indices, [-1, 3]))[..., -1],
                                        [-1, self.flags.buckets])  # N * bucket
                mask = tf.greater_equal(current_reward, old_reward)
                final_indices = tf.boolean_mask(indices, mask)
                final_values = tf.boolean_mask(values, mask)
                self.update = tf.scatter_nd_update(self.table, indices=final_indices, updates=final_values,
                                                   use_locking=False)

                # lookup a key in memory
                single_keys = self.batch_keys[0]  # (5,)
                indices = []
                for i in range(self.flags.buckets):
                    index = tf.stack([single_keys[i], i])
                    indices.append(index)
                res = tf.gather_nd(self.table, indices=indices)
                query_result = tf.transpose(res, [1, 0, 2])  # A * bucket * (bucket+1)
                query_rewards = query_result[..., -1]  # A * bucket
                query_all_keys = query_result[..., :-1]  # A * bucket * bucket

                similarity = self._compute_batch_vector_similarity(tf.reshape(query_all_keys, [-1, self.flags.buckets]),
                                                                   single_keys)
                self.similarity = tf.reshape(similarity, [self.flags.num_actions, self.flags.buckets])
                action_norm_sim = tf.reduce_sum(self.similarity, axis=1, keep_dims=True)  # A * 1
                sim_weights = tf.cast(self.similarity, tf.float32) / tf.cast(action_norm_sim, tf.float32)  # A*B
                final_estimated_rewards = tf.cast(query_rewards, tf.float32) * sim_weights

                self.estimated_reward = tf.reduce_sum(final_estimated_rewards, axis=1)  # A

    def _dot_product_mod(self, a):
        mod = self.flags.episodic_memory
        b = self.bit_weights
        c = tf.mod(tf.multiply(a, b), mod)
        key = tf.scan(lambda cum, x: (cum + x) % mod, c, back_prop=False)[-1]
        return key

    def _get_hash_table_indices(self, x):
        """return index x[bucket], bucket, action"""
        action = x[-1]
        # reward = x[-2]
        indices = []
        for i in range(self.flags.buckets):
            index = tf.stack([x[i], i, action])  # (3,)
            indices.append(index)
        return tf.stack(indices)

    def _compute_batch_vector_similarity(self, m, y):
        """input: m is (N, d)  y is (d,)  output: (N,)"""
        def f1(m_x):
            return self._compute_vector_vector_similarity(m_x, y)
        return tf.map_fn(f1, m, back_prop=False, parallel_iterations=128)

    @staticmethod
    def _compute_vector_vector_similarity(a, b):
        """a, b are vectors"""
        eq_matrix = tf.equal(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
        dim0 = tf.reduce_sum(tf.cast(tf.reduce_any(eq_matrix, axis=0), tf.int32))
        dim1 = tf.reduce_sum(tf.cast(tf.reduce_any(eq_matrix, axis=1), tf.int32))
        return tf.minimum(dim0, dim1)

    def start_updating_memory(self):
        self.train_data_set.batch_top = self.train_data_set.bottom
        for i in xrange(self.flags.feeding_threads):
            t = threading.Thread(target=self._update_thread_process, args=())
            t.setDaemon(True)
            self.update_threads.append(t)
            t.start()

    def stop_updating(self):
        self.coord.request_stop()
        self.coord.join(self.update_threads, stop_grace_period_secs=1.0)

    def _update_thread_process(self):
        while not self.coord.should_stop():
            if (self.train_data_set.batch_top + self.buffer_step) % self.train_data_set.max_steps > self.train_data_set.top:
                time.sleep(30.0)
                continue
            imgs = np.take(self.train_data_set.imgs,
                           np.arange(self.train_data_set.batch_top, self.train_data_set.batch_top + self.buffer_step),
                           axis=0,
                           mode='wrap')
            cum_rewards = np.take(self.train_data_set.cum_unclipped_rewards,
                                  np.arange(self.train_data_set.batch_top, self.train_data_set.batch_top + self.buffer_step),
                                  mode='wrap')
            actions = np.take(self.train_data_set.actions,
                              np.arange(self.train_data_set.batch_top, self.train_data_set.batch_top + self.buffer_step),
                              mode='wrap')
            self.train_data_set.batch_top = (self.train_data_set.batch_top + self.buffer_step) % self.train_data_set.max_steps
            self.sess.run(self.update,
                          feed_dict={self.states_input_batch: imgs,
                                     self.rewards: cum_rewards,
                                     self.actions: actions})
            self.counter += 1
            # for test
            self.memory_update_file.write('batchtop= ' + str(self.train_data_set.batch_top) + ' top=' +
                                          str(self.train_data_set.top) + '\n')
            if self.counter % 10 == 0:
                non_zero = self.sess.run(self.hash_table_non_zero)
                self.memory_update_file.write('hash table nonzero: ' + str(non_zero) + '/' +
                                              str(self.flags.episodic_memory * self.flags.buckets *
                                                  self.flags.num_actions) + '\n')
            self.memory_update_file.flush()

    def add_train_data_set(self, data_set):
        self.train_data_set = data_set

    def add_sess(self, sess):
        self.sess = sess

    def compute_keys(self, states):
        """states: N * 84 * 84"""
        return self.sess.run(self.batch_keys, feed_dict={self.states_input_batch: states})

    def lookup_single_state(self, state):
        states = np.reshape(state, [1, self.original_height, self.original_width])
        rewards = self.sess.run(self.estimated_reward, feed_dict={self.states_input_batch: states})
        return rewards

