__author__ = 'frankhe'
import numpy as np


def feature_distance(a, b, metric='l2'):
    """ a: N*d;  b: d  return N"""
    if metric == 'l2':
        return np.sum(np.square(a - b), axis=1)
    if metric == 'cos':
        return np.matmul(a, b)


class ActionMemory(object):
    def __init__(self, flags, network):
        self.flags = flags
        self.network = network
        self.width = flags.input_width
        self.height = flags.input_height
        self.max_length = self.flags.episodic_memory
        self.keys = np.zeros((self.max_length, self.flags.feature_dim), dtype='float32')
        self.values = np.zeros(self.max_length, dtype='float32')
        self.phi_states = np.zeros((self.max_length, self.flags.phi_length, self.height, self.width), dtype='uint8')
        self.frequency = np.zeros(self.max_length, 'float32')
        self.top = 0

        self.buffer_length = self.flags.episodic_memory_buffer
        self.imgs_buffer = np.zeros((self.buffer_length, self.flags.phi_length, self.height, self.width), dtype='uint8')
        self.unclipped_cumulative_reward_buffer = np.zeros(self.buffer_length, dtype='float32')
        self.buffer_top = 0
        assert self.max_length % self.buffer_length == 0 and self.max_length > self.buffer_length

    def add_item(self, phi_state, unclipped_reward):
        self.imgs_buffer[self.buffer_top] = phi_state
        self.unclipped_cumulative_reward_buffer[self.buffer_top] = unclipped_reward
        self.buffer_top += 1
        if self.buffer_top >= self.buffer_length:
            self.flush_buffer()
            self.buffer_top = 0

    def flush_buffer(self):
        features = self.network.get_features(self.imgs_buffer)
        if self.top < self.max_length:
            self.keys[self.top: self.top + self.buffer_length] = features
            self.values[self.top: self.top + self.buffer_length] = self.unclipped_cumulative_reward_buffer
            self.phi_states[self.top: self.top + self.buffer_length] = self.imgs_buffer
            self.top += self.buffer_length
            return

        idx = np.argpartition(self.frequency, self.buffer_length)[:self.buffer_length]
        self.keys[idx] = features
        self.values[idx] = self.unclipped_cumulative_reward_buffer
        self.phi_states[idx] = self.imgs_buffer

        self.frequency.fill(0.0)

    def lookup(self, feature):
        if self.top < self.max_length:
            return 10.0
        distance = feature_distance(self.keys, feature)
        idx = np.argpartition(distance, self.flags.knn)[:self.flags.knn]
        weight = 1.0 / (distance[idx] + 0.001)
        weight = weight / np.sum(weight)
        self.frequency[idx] = self.frequency[idx] + weight
        cumulative_reward = np.matmul(weight, self.values[idx])
        return cumulative_reward

    def refresh_keys(self):
        if self.top < self.max_length:
            return
        self.keys = self.network.get_features(self.phi_states)


class EpisodicMemory(object):
    def __init__(self, flags, network):
        self.flags = flags
        self.network = network
        self.action_memories = [ActionMemory(self.flags, self.network)] * self.flags.num_actions
        self.action_values = np.zeros(self.flags.num_actions, 'float32')

    def add_item(self, phi_state, action, unclipped_reward):
        self.action_memories[action].add_item(phi_state, unclipped_reward)
        pass

    def lookup(self, feature):
        for action in xrange(self.flags.num_actions):
            self.action_values[action] = self.action_memories[action].lookup(feature)
        return self.action_values

    def refresh_features(self):
        for action in xrange(self.flags.num_actions):
            self.action_memories[action].refresh_keys()
