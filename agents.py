__author__ = 'frankhe'
import time
import numpy as np
import data_sets


class QLearning(object):
    def __init__(self, pid, network, flags, message_queue):
        self.pid = pid
        self.network = network
        self.flags = flags
        self.message_queue = message_queue
        self.train_data_set = data_sets.DataSet(flags)
        self.test_data_set = data_sets.DataSet(flags, max_steps=flags.phi_length * 2)
        self.epsilon = flags.ep_st
        if flags.ep_decay != 0:
            self.epsilon_rate = (flags.ep_st - flags.ep_min) / flags.ep_decay
        else:
            self.epsilon_rate = 0

        # global attributes:
        self.epoch_start_time = time.time()
        self.start_index = 0
        self.terminal_index = None
        self.steps_sec_ema = 0  # add to tensorboard per epoch
        self.epoch_time = 0  # add to tensorboard per epoch
        self.state_action_avg_val = 0  # add to tensorboard per epoch
        self.testing_data = None

        # episode attributes:
        self.step_counter = 0
        self.trained_batch_counter = 0
        self.episode_reward = 0
        self.loss_averages = None
        self.start_time = None
        self.last_action = None
        self.last_img = None

        self.testing = False
        self.episode_counter = 0
        self.total_reward = 0  # add to tensorboard per epoch
        self.reward_per_episode = 0  # add to tensorboard per epoch

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.trained_batch_counter = 0
        self.episode_reward = 0  # only useful when testing
        # We report the mean loss for every episode.
        self.loss_averages = []
        self.start_time = time.time()
        action = np.random.randint(0, self.flags.num_actions)  # could be changed to NN.choose_action
        self.last_action = action
        self.last_img = observation
        return action

    def end_episode(self, reward, terminal):
        """
        :param terminal:
         training:
            terminal = True:  game over or lost a life
            terminal = False: game is not over but not enough steps
         testing:
            terminal = True: game over
            terminal = False: game is not over but not enough steps
        :param reward: received reward
        """
        self.step_counter += 1
        episode_time = time.time() - self.start_time

        if self.testing:
            self.episode_reward += reward
            # we do not count this episode if agent running out of steps
            # we do count if episode is finished or this episode is the only episode
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:
            self.train_data_set.add_sample(self.last_img, self.last_action, reward, True,
                                           start_index=self.start_index)
        """
        update index not finished
        """

        rho = 0.98
        self.steps_sec_ema = rho * self.steps_sec_ema + (1.0 - rho) * (self.step_counter / episode_time)
        # print 'PID:', self.pid, 'steps/second current:{:.2f}, avg:{:.2f}'.format(self.step_counter/episode_time,
        #                                                                          self.steps_sec_ema)
        message = [self.pid, 'speed', [int(self.step_counter / episode_time), int(self.steps_sec_ema)]]
        self.message_queue.put(message)
        if self.loss_averages:  # if not empty
            self.network.episode_summary(np.mean(self.loss_averages))

    def choose_action(self, data_set, img, epsilon, reward_received):
        data_set.add_sample(self.last_img, self.last_action, reward_received, False, start_index=self.start_index)
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.flags.num_actions)
        phi = data_set.phi(img)
        if self.step_counter < self.flags.phi_length:
            phi[0:self.flags.phi_length - 1] = np.stack([img for _ in xrange(self.flags.phi_length - 1)], axis=0)
        action = self.network.choose_action(phi)
        return action

    def _train(self):
        imgs, actions, rewards, terminals = self.train_data_set.random_batch(self.flags.batch)
        cur_images = imgs[:, :-1, ...]
        target_images = imgs[:, 1:, ...]
        return self.network.train(cur_images, target_images, rewards, actions, terminals)

    def step(self, reward, observation):
        """
        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array uint8
        Returns:
           An integer action.
        """
        self.step_counter += 1
        # TESTING---------------------------
        if self.testing:
            self.episode_reward += reward
            action = self.choose_action(self.test_data_set, observation, .01, reward)
        else:
            # Training--------------------------
            if len(self.train_data_set) > self.flags.train_st:
                self.epsilon = max(self.flags.ep_min, self.epsilon - self.epsilon_rate)
                action = self.choose_action(self.train_data_set, observation, self.epsilon, reward)
                if self.step_counter % self.flags.train_fr == 0:
                    loss = self._train()
                    self.trained_batch_counter += 1
                    self.loss_averages.append(loss)
            else:
                action = self.choose_action(self.train_data_set, observation, self.epsilon, reward)
        self.last_action = action
        self.last_img = observation
        return action

    def finish_epoch(self, epoch):
        # save model epoch parameters
        self.network.epoch_model_save(epoch)
        current_time = time.time()
        self.epoch_time = current_time - self.epoch_start_time

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        test_data_size = 3200
        if self.testing_data is None and len(self.train_data_set) > test_data_size:
            imgs, _, _, _ = self.train_data_set.random_batch(test_data_size)
            self.testing_data = imgs[:, :-1, ...]
        if self.testing_data is not None:
            self.state_action_avg_val = np.mean(np.max(self.network.get_action_values(self.testing_data), axis=1))
        self.reward_per_episode = self.total_reward / float(self.episode_counter)
        # print 'PID', self.pid, 'reward per episode:', self.reward_per_episode, 'total reward', self.total_reward, \
        #     'mean q:', self.state_action_avg_val
        message = 'PID:{:d}  epoch:{:d}  total_reward={:.1f}  reward_per_episode={:.1f}     mean q={:.1f}'.format(
            self.pid, epoch, self.total_reward, self.reward_per_episode, self.state_action_avg_val)
        self.message_queue.put([-1, 'print', message])
        self.network.epoch_summary(epoch, self.epoch_time, self.state_action_avg_val, self.total_reward,
                                   self.reward_per_episode)
        self.epoch_start_time = time.time()
