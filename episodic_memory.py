__author__ = 'frankhe'
import numpy as np
import tensorflow as tf


class EpisodicMemory(object):
    def __init__(self, flags, device):
        self.flags = flags
        self.width = flags.input_width
        self.height = flags.input_height
        self.train_data_set =None
        self.add_indices = []
        self.add_rewards = []
        self.sess = None

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
                self.states_input_batch = tf.placeholder(tf.uint8, [None, self.height, self.width])
                states_batch = tf.reshape(tf.cast(self.states_input_batch, tf.float32), [-1, self.height * self.width])
                keys_batch = []
                for i in range(self.flags.buckets):
                    # out: N * hash_dim   complexity: O(N*7056*hash_dim)
                    bit_keys = tf.cast(tf.sign(tf.matmul(states_batch, self.projection_matrices[i])), tf.int32)
                    keys = tf.map_fn(self._dot_product_mod, bit_keys, back_prop=False)  # N
                    keys_batch.append(keys)
                self.batch_keys = tf.stack(keys_batch, axis=1, name='keys')  # N * bucket_size

                # update memory
                self.rewards = tf.placeholder(tf.float32, [None])
                self.actions = tf.placeholder(tf.int32, [None])
                rewards = tf.cast(self.rewards, tf.int32)
                actions = self.actions
                batch_key_values = tf.stack(keys_batch + [rewards, actions], axis=1, name='batch_update')  # N * (bucket_size + 1)
                self.update = tf.map_fn(self._update_hash_table, batch_key_values, back_prop=False)

                # lookup a key in memory
                single_keys = self.batch_keys[0]  # (5,)
                results = []
                for i in range(self.flags.buckets):
                    index = tf.stack([single_keys[i], i])
                    res = tf.gather_nd(self.table, indices=index)
                    results.append(res)
                query_result = tf.stack(results, axis=1)  # A * bucket * (bucket+1)
                query_rewards = query_result[..., -1]  # A * bucket
                query_all_keys = query_result[..., :-1]  # A * bucket * bucket

                similarity = self._compute_batch_vector_similarity(tf.reshape(query_all_keys, [-1, self.flags.buckets]),
                                                                   single_keys)
                self.similarity = tf.reshape(similarity, [self.flags.num_actions, self.flags.buckets])

                self.estimated_reward = tf.multiply(tf.cast(query_rewards, tf.float32) / self.flags.buckets,
                                                    tf.cast(self.similarity, tf.float32))

    def _dot_product_mod(self, a):
        mod = self.flags.episodic_memory
        b = self.bit_weights
        c = tf.mod(tf.multiply(a, b), mod)
        key = tf.scan(lambda cum, x: (cum + x) % mod, c, back_prop=False)[-1]
        return key

    def _update_hash_table(self, x):
        action = x[-1]
        reward = x[-2]
        for i in range(self.flags.buckets):
            index = tf.stack([x[i], i, action])  # (3,)
            old_reward = tf.gather_nd(self.table, indices=index)[-1]  # (bucket + 1,)

            def f1():
                tf.scatter_nd_update(self.table, indices=index, updates=x[:-1])
                return 1
            return tf.cond(tf.greater_equal(reward, old_reward),
                           true_fn=f1,
                           false_fn=lambda: 0)

    def _compute_batch_vector_similarity(self, m, y):
        """input: m is (N, d)  y is (d,)  output: (N,)"""
        def f1(m_x):
            return self._compute_vector_vector_similarity(m_x, y)
        return tf.map_fn(f1, m, back_prop=False)

    @staticmethod
    def _compute_vector_vector_similarity(a, b):
        """a, b are vectors"""
        eq_matrix = tf.equal(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
        dim0 = tf.reduce_sum(tf.cast(tf.reduce_any(eq_matrix, axis=0), tf.int32))
        dim1 = tf.reduce_sum(tf.cast(tf.reduce_any(eq_matrix, axis=1), tf.int32))
        return tf.minimum(dim0, dim1)

    def add_train_data_set(self, data_set):
        self.train_data_set = data_set

    def add_sess(self, sess):
        self.sess = sess

    def compute_keys(self, states):
        """states: N * 84 * 84"""
        return self.sess.run(self.batch_keys, feed_dict={self.states_input_batch: states})

    def add_item(self, index, cumulative_unclipped_reward):
        self.add_indices.append(index)
        self.add_rewards.append(cumulative_unclipped_reward)

    def add_flush(self):
        res = self.sess.run(self.update,
                            feed_dict={self.states_input_batch: self.train_data_set.imgs[self.add_indices],
                                       self.rewards: self.add_rewards,
                                       self.actions: self.train_data_set.actions[self.add_indices]})
        return res

    def lookup_single_state(self, state):
        states = np.reshape(state, [1, self.height, self.width])
        sim, rewards = self.sess.run([self.similarity, self.estimated_reward], feed_dict={self.states_input_batch: states})
        return sim, rewards

