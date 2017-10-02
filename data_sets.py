__author__ = 'frankhe'
import numpy as np


class DataSet(object):
    def __init__(self, flags, max_steps=None):
        self.flags = flags
        self.width = flags.input_width
        self.height = flags.input_height
        self.max_steps = flags.memory
        if max_steps is not None:
            self.max_steps = max_steps
        self.phi_length = flags.phi_length

        self.imgs = np.zeros((self.max_steps, self.height, self.width), dtype='uint8')
        self.actions = np.zeros(self.max_steps, dtype='int32')
        self.rewards = np.zeros(self.max_steps, dtype='float32')
        self.unclipped_rewards = np.zeros(self.max_steps, dtype='float32')
        self.return_value = np.zeros(self.max_steps, dtype='float32')
        self.terminal = np.zeros(self.max_steps, dtype='bool')
        self.start_index = np.zeros(self.max_steps, dtype='int32')
        self.terminal_index = np.zeros(self.max_steps, dtype='int32')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, action, reward, terminal, return_value=0.0, start_index=-1):
        self.unclipped_rewards[self.top] = reward
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

    def last_phi(self, index=None):
        """Return the most recent phi (sequence of image frames)."""
        if index is None:
            index = self.top
        indexes = np.arange(index - self.phi_length, index)
        phi = self.imgs.take(indexes, axis=0, mode='wrap')
        terminals = self.terminal.take(indexes[:-1], mode='wrap')
        for i in xrange(self.phi_length-2,-1,-1):
            if terminals[i]:
                phi[0:i+1] = np.stack([phi[i+1]] * (i+1))
                break
        return phi

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        phi = self.last_phi()
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


class OptimalityTighteningDataset(DataSet):
    def __init__(self, flags, max_step=None):
        super(OptimalityTighteningDataset, self).__init__(flags, max_step)
        self.discount_table = np.power(self.flags.discount, np.arange(30))
        batch_size = self.flags.batch
        transitions_len = self.flags.nob
        self.center_imgs = np.zeros((batch_size,
                                     self.phi_length,
                                     self.height,
                                     self.width),
                                    dtype='uint8')
        self.forward_imgs = np.zeros((batch_size,
                                      transitions_len,
                                      self.phi_length,
                                      self.height,
                                      self.width),
                                     dtype='uint8')
        self.backward_imgs = np.zeros((batch_size,
                                       transitions_len,
                                       self.phi_length,
                                       self.height,
                                       self.width),
                                      dtype='uint8')
        self.center_positions = np.zeros((batch_size, ), dtype='int32')
        self.forward_positions = np.zeros((batch_size, transitions_len), dtype='int32')
        self.backward_positions = np.zeros((batch_size, transitions_len), dtype='int32')

        self.center_actions = np.zeros((batch_size, ), dtype='int32')
        self.backward_actions = np.zeros((batch_size, transitions_len), dtype='int32')

        self.center_terminals = np.zeros((batch_size, ), dtype='bool')
        self.center_rewards = np.zeros((batch_size, ), dtype='float32')

        self.center_return_values = np.zeros((batch_size, ), dtype='float32')
        self.forward_return_values = np.zeros((batch_size, transitions_len), dtype='float32')
        self.backward_return_values = np.zeros((batch_size, transitions_len), dtype='float32')

        self.forward_discounts = np.zeros((batch_size, transitions_len), dtype='float32')
        self.backward_discounts = np.zeros((batch_size, transitions_len), dtype='float32')

    def random_batch_with_close_bounds(self, batch_size):
        assert batch_size == self.flags.batch
        transition_range = transitions_len = self.flags.nob
        count = 0
        while count < batch_size:
            index = np.random.randint(self.bottom, self.bottom + self.size - self.phi_length)
            all_indices = np.arange(index, index + self.phi_length)
            center_index = index + self.phi_length - 1
            """
            frame0 frame1 frame2 frame3
            index                center_index = index+phi-1
            """
            if np.any(self.terminal.take(all_indices[0:-1], mode='wrap')):
                continue
            if np.any(self.terminal_index.take(all_indices, mode='wrap') == -1):
                continue
            terminal_index = self.terminal_index.take(center_index, mode='wrap')
            start_index = self.start_index.take(center_index, mode='wrap')
            self.center_positions[count] = center_index
            self.center_terminals[count] = self.terminal.take(center_index, mode='wrap')
            self.center_rewards[count] = self.rewards.take(center_index, mode='wrap')

            """ get forward transitions """
            if terminal_index < center_index:
                terminal_index += self.size
            max_forward_index = max(min(center_index + transition_range, terminal_index), center_index + 1) + 1
            self.forward_positions[count] = center_index + 1
            for i, j in zip(range(transitions_len), range(center_index + 1, max_forward_index)):
                self.forward_positions[count, i] = j
            """ get backward transitions """
            if start_index + self.size < center_index:
                start_index += self.size
            min_backward_index = max(center_index - transition_range, start_index + self.phi_length - 1)
            self.backward_positions[count] = center_index + 1
            for i, j in zip(range(transitions_len), range(center_index - 1, min_backward_index - 1, -1)):
                self.backward_positions[count, i] = j
                if self.terminal_index.take(j, mode='wrap') == -1:
                    self.backward_positions[count, i] = center_index + 1

            self.center_imgs[count] = self.imgs.take(all_indices, axis=0, mode='wrap')
            for j in xrange(transitions_len):
                forward_index = self.forward_positions[count, j]
                backward_index = self.backward_positions[count, j]
                self.forward_imgs[count, j] = self.imgs.take(
                    np.arange(forward_index - self.phi_length + 1, forward_index + 1), axis=0, mode='wrap')
                self.backward_imgs[count, j] = self.imgs.take(
                    np.arange(backward_index - self.phi_length + 1, backward_index + 1), axis=0, mode='wrap')
            self.center_actions[count] = self.actions.take(center_index, mode='wrap')
            self.backward_actions[count] = self.actions.take(self.backward_positions[count], mode='wrap')
            self.center_return_values[count] = self.return_value.take(center_index, mode='wrap')
            self.forward_return_values[count] = self.return_value.take(self.forward_positions[count], mode='wrap')
            self.backward_return_values[count] = self.return_value.take(self.backward_positions[count], mode='wrap')
            distance = np.absolute(self.forward_positions[count] - center_index)
            self.forward_discounts[count] = self.discount_table[distance]
            distance = np.absolute(self.backward_positions[count] - center_index)
            self.backward_discounts[count] = self.discount_table[distance]
            # print self.backward_positions[count][::-1], self.center_positions[count], self.forward_positions[count]
            # print 'start=', start_index, 'center=', self.center_positions[count], 'end=', terminal_index
            # raw_input()
            count += 1


if __name__ == '__main__':
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    my_flags = AttrDict(input_height=84, input_width=84, memory=1000, phi_length=4, clip_reward=True, discount=0.99,
                        batch=32, nob=4)
    d = OptimalityTighteningDataset(my_flags, 100)
    print d.discount_table
