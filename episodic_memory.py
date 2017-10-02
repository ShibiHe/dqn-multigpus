__author__ = 'frankhe'
import numpy as np
import tensorflow as tf


class EpisodicMemory(object):
    def __init__(self, flags, device):
        self.flags = flags
        self.width = flags.input_width
        self.height = flags.input_height
        self.sess = None

        if not flags.epm_use_gpu:
            device = '/cpu:0'
        else:
            device = '/gpu:' + device

        with tf.device(device):
            with tf.variable_scope('Episodic'):
                # initialize hash tables
                for action in range(self.flags.num_actions):
                    with tf.variable_scope('action' + str(action)):
                        for bucket in range(self.flags.buckets):
                            tf.Variable(tf.zeros((self.flags.episodic_memory, self.flags.buckets + 1)),
                                        trainable=False,
                                        name='bucket' + str(bucket),
                                        dtype=tf.float32)

                self.buckets = [self.flags.episodic_memory] * self.flags.buckets
                self.projection_matrices = []
                for i in range(self.flags.buckets):
                    p_matrix = tf.Variable(tf.random_normal((self.width * self.height,
                                                             self.flags.hash_dim)),
                                           trainable=False,
                                           dtype=tf.float32,
                                           name='projection' + str(i))
                    self.projection_matrices.append(p_matrix)
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

                self.states_input_batch = tf.placeholder(tf.uint8, [None, self.height, self.width])
                states_batch = tf.reshape(tf.cast(self.states_input_batch, tf.float32), [-1, self.height * self.width])
                keys_batch = []
                for i in range(self.flags.buckets):
                    # out: N * hash_dim   complexity: O(N*7056*hash_dim)
                    bit_keys = tf.cast(tf.sign(tf.matmul(states_batch, self.projection_matrices[i])), tf.int32)
                    keys = tf.map_fn(self._dot_product_mod, bit_keys)  # N
                    keys_batch.append(keys)  # bucket_size * N
                self.batch_keys = tf.stack(keys_batch)  # N * bucket_size

                self.p_place = tf.placeholder(tf.float32, (self.width * self.height, self.flags.hash_dim))
                self.p_assign = []
                for i in range(self.flags.buckets):
                    self.p_assign.append(tf.assign(self.projection_matrices[i], self.p_place))

    def _dot_product_mod(self, a):
        mod = self.flags.episodic_memory
        b = self.bit_weights
        c = tf.mod(tf.multiply(a, b), mod)
        key = tf.scan(lambda cum, x: (cum + x) % mod, c, back_prop=False)[-1]
        return key

    def compute_keys(self, states):
        """states: N * 84 * 84"""
        return self.sess.run(self.batch_keys, feed_dict={self.states_input_batch: states})







    #
    #
    #
    # def add_item(self, phi_state, action, unclipped_reward):
    #     self.action_memories[action].add_item(phi_state, unclipped_reward)
    #     pass
    #
    # def lookup(self, feature):
    #     for action in xrange(self.flags.num_actions):
    #         self.action_values[action] = self.action_memories[action].lookup(feature)
    #     return self.action_values
    #
    # def refresh_features(self):
    #     for action in xrange(self.flags.num_actions):
    #         self.action_memories[action].refresh_keys()
