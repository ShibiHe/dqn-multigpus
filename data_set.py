__author__ = 'frankhe'
import numpy as np


class DataSet(object):
    def __init__(self, flags):
        self.flags = flags
        self.width = flags.input_width
        self.height = flags.input_height
        self.max_steps = flags.memory
        self.phi_length = flags.phi_length

        self.imgs = np.zeros((self.max_steps, self.height, self.width), dtype='uint8')
        self.actions = np.zeros(self.max_steps, dtype='int32')
        self.rewards = np.zeros(self.max_steps, dtype='float32')
        self.return_value = np.zeros(self.max_steps, dtype='float32')
        self.terminal = np.zeros(self.max_steps, dtype='bool')
        self.start_index = np.zeros(self.max_steps, dtype='int32')
        self.terminal_index = np.zeros(self.max_steps, dtype='int32')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, action, reward, terminal, return_value=0.0, start_index=-1):
        if self.flags.clip_reward:
            reward = np.clip(reward, -1, 1)
        self.imgs[self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal
        self.return_value[self.top] = return_value
        self.start_index[self.top] = start_index
        self.terminal_index[self.top] = -1

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        return self.size

    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.imgs.take(indexes, axis=0, mode='wrap')

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype='uint8')
        phi[0:self.phi_length - 1] = self.imgs.take(indexes,
                                                    axis=0,
                                                    mode='wrap')
        phi[-1] = img
        return phi

    def random_batch(self, batch_size, get_return_value=False, get_index=False):
        # Allocate the response.
        imgs = np.zeros((batch_size,
                         self.phi_length + 1,
                         self.height,
                         self.width),
                        dtype='uint8')
        actions = np.zeros((batch_size,), dtype='int32')
        rewards = np.zeros((batch_size,), dtype='float32')
        terminal = np.zeros((batch_size,), dtype='bool')
        return_value = np.zeros((batch_size,), dtype='float32')
        positions = np.zeros((batch_size,), dtype='int32')

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = np.random.randint(self.bottom, self.bottom + self.size - self.phi_length)

            # Both the before and after states contain phi_length
            # frames, overlapping except for the first and last.
            all_indices = np.arange(index, index + self.phi_length + 1)
            end_index = index + self.phi_length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but its last frame (the
            # second-to-last frame in imgs) may be terminal. If the last
            # frame of the initial state is terminal, then the last
            # frame of the transitioned state will actually be the first
            # frame of a new episode, which the Q learner recognizes and
            # handles correctly during training by zeroing the
            # discounted future reward estimate.
            if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')):
                continue

            # Add the state transition to the response.
            imgs[count] = self.imgs.take(all_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            return_value[count] = self.return_value.take(end_index, mode='wrap')
            positions[count] = end_index % self.size
            count += 1

        return_tuple = [imgs, actions, rewards, terminal]
        if get_return_value:
            return_tuple.append(return_value)
        if get_index:
            return_tuple.append(positions)
        return return_tuple

