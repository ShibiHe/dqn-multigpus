__author__ = 'frankhe'
import time
import numpy as np
import data_sets


class QLearning(object):
    def __init__(self, pid, network, flags, epm, comm):
        self.pid = pid
        self.network = network
        self.flags = flags
        self.epm = epm
        self.comm = comm
        self.train_data_set = data_sets.DataSet(flags)
        self.test_data_set = data_sets.DataSet(flags, max_steps=flags.phi_length * 2)
        self.network.add_train_data_set(self.train_data_set)
        self.epm.add_train_data_set(self.train_data_set)
        self.network.epm = self.epm
        sess = self.network.init()
        self.epm.add_sess(sess)

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
        self.trained_batch_counter = 0
        self.global_step_counter = 0

        # episode attributes:
        self.step_counter = 0
        self.episode_reward = 0
        self.loss_averages = None
        self.start_time = None
        self.last_action = None
        self.last_img = None

        self.testing = False
        self.episode_counter = 0
        self.total_reward = 0  # add to tensorboard per epoch
        self.reward_per_episode = 0  # add to tensorboard per epoch

        # for test
        self.action_slection_file = None

    def add_record_files(self, a, b):
        self.epm.memory_update_file = a
        self.action_slection_file = b

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
        self.global_step_counter += 1
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
        post end episode functions
        """
        self._post_end_episode(terminal)

        rho = 0.98
        self.steps_sec_ema = rho * self.steps_sec_ema + (1.0 - rho) * (self.step_counter / episode_time)
        # print 'PID:', self.pid, 'steps/second current:{:.2f}, avg:{:.2f}'.format(self.step_counter/episode_time,
        #                                                                          self.steps_sec_ema)
        message = [self.pid, 'speed', [int(self.step_counter / episode_time), int(self.steps_sec_ema)]]
        self.comm.send(message, dest=self.flags.threads)
        if self.loss_averages:  # if not empty
            self.network.episode_summary(np.mean(self.loss_averages))

    def _post_end_episode(self, terminal):
        if self.testing:
            return
        unclipped_cumulative_reward = 0.0
        self.start_index = self.train_data_set.top
        self.terminal_index = index = (self.start_index - 1) % self.train_data_set.max_steps
        while True:
            unclipped_cumulative_reward = unclipped_cumulative_reward * self.flags.discount + \
                                          self.train_data_set.unclipped_rewards[index]
            # self.train_data_set.terminal_index[index] = self.terminal_index
            self.train_data_set.cum_unclipped_rewards[index] = unclipped_cumulative_reward
            index = (index - 1) % self.train_data_set.max_steps
            if self.train_data_set.terminal[index] or index == self.train_data_set.bottom:
                break

    def choose_action(self, data_set, img, epsilon, reward_received):
        data_set.add_sample(self.last_img, self.last_action, reward_received, False,
                            start_index=self.start_index)
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.flags.num_actions)
        phi = data_set.phi(img)
        action = self.network.choose_action(phi)
        sim, unclipped_rewards = self.epm.lookup_single_state(phi[-1])
        index = np.unravel_index(np.argmax(unclipped_rewards), sim.shape)
        final_sim = sim[index]

        if np.random.randint(0, self.flags.buckets) < final_sim:
            # for test
            self.action_slection_file.write(str(action) + '  act2: ' + str(index[0]) + ' sim=' + str(final_sim) +
                                            ' reward=' + str(unclipped_rewards[index]) + '\n')
            self.action_slection_file.flush()
            return index[0]
        return action

    def _train(self):
        return self.network.train()

    def step(self, reward, observation):
        """
        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array uint8
        Returns:
           An integer action.
        """
        self.step_counter += 1
        self.global_step_counter += 1
        # TESTING---------------------------
        if self.testing:
            self.episode_reward += reward
            action = self.choose_action(self.test_data_set, observation, .01, reward)
        else:
            # Training--------------------------
            if len(self.train_data_set) > self.flags.train_st:
                self.epsilon = max(self.flags.ep_min, self.epsilon - self.epsilon_rate)
                if self.trained_batch_counter > self.flags.ep_decay_b:
                    self.epsilon = 0.01
                action = self.choose_action(self.train_data_set, observation, self.epsilon, reward)
                if self.global_step_counter % self.flags.train_fr == 0:
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
        if self.flags.ckpt:
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
        self.comm.send([-1, 'print', message], dest=self.flags.threads)
        self.network.epoch_summary(epoch, self.epoch_time, self.state_action_avg_val, self.total_reward,
                                   self.reward_per_episode)
        self.epoch_start_time = time.time()

    def finish_everything(self):
        self.network.stop_feeding()
        self.comm.send([self.pid, 'END', ''], dest=self.flags.threads)
        # self.comm.Barrier()


class OptimalityTigheningAgent(QLearning):
    def __init__(self, pid, network, flags, epm, comm):
        super(OptimalityTigheningAgent, self).__init__(pid, network, flags, epm, comm)
        self.train_data_set = data_sets.OptimalityTighteningDataset(flags)
        self.network.add_train_data_set(self.train_data_set)

    def _post_end_episode(self, terminal):
        if self.testing:
            return
        if terminal:
            q_return = 0.0
        else:
            phi = self.train_data_set.phi(self.last_img)
            phi = np.expand_dims(phi, 0)
            q_return = np.mean(self.network.get_action_values(phi))
        self.start_index = self.train_data_set.top
        self.terminal_index = index = (self.start_index - 1) % self.train_data_set.max_steps
        while True:
            q_return = q_return * self.flags.discount + self.train_data_set.rewards[index]
            self.train_data_set.return_value[index] = q_return
            self.train_data_set.terminal_index[index] = self.terminal_index
            index = (index - 1) % self.train_data_set.max_steps
            if self.train_data_set.terminal[index] or index == self.train_data_set.bottom:
                break

    def deprecated_train(self):
        if self.flags.close2:
            self.train_data_set.random_batch_with_close_bounds(self.flags.batch)
        else:
            pass
        target_q_imgs = np.concatenate((self.train_data_set.forward_imgs, self.train_data_set.backward_imgs), axis=1)
        target_q_imgs = np.reshape(target_q_imgs, (-1,) + target_q_imgs.shape[2:])
        """here consider center image's target too as a lower bound"""
        target_q_table = self.network.get_action_values_old(target_q_imgs)
        target_q_table = np.reshape(target_q_table, (self.flags.batch, -1) + (target_q_table.shape[-1], ))

        q_values = self.network.get_action_values_given_actions(self.train_data_set.center_imgs, self.train_data_set.center_actions)

        states1 = np.zeros((self.flags.batch, self.flags.phi_length, self.flags.input_height, self.flags.input_width), dtype='uint8')
        actions1 = np.zeros((self.flags.batch, ), dtype='int32')
        targets1 = np.zeros((self.flags.batch, ), dtype='float32')
        """
            0 1 2 3* 4 5 6 7 8 V_R
            0 1 2 4  5 6 7 8 V_R
            V0 = r3 + y*Q4; V1 = r3 +y*r4 + y^2*Q5
            Q2 -r2 = Q3*y; Q1 - r1 - y*r2  = y^2*Q3
            V-1 = (Q2 - r2) / y; V-2 = (Q1 - r1 - y*r2)/y^2; V-3 = (Q0 -r0 -y*r1 - y^2*r2)/y^3
            r1 + y*r2 = R1 - y^2*R3
            Q1 = r1+y*r2 + y^2*Q3
        """
        for i in xrange(self.flags.batch):
            q_value = q_values[i]
            center_position = int(self.train_data_set.center_positions[i])
            if self.train_data_set.terminal.take(center_position, mode='wrap'):
                states1[i] = self.train_data_set.center_imgs[i]
                actions1[i] = self.train_data_set.center_actions[i]
                targets1[i] = self.train_data_set.center_return_values[i]
                continue
            forward_targets = np.zeros(self.flags.nob, dtype=np.float32)
            backward_targets = np.zeros(self.flags.nob, dtype=np.float32)
            for j in xrange(self.flags.nob):
                if j > 0 and self.train_data_set.forward_positions[i, j] == center_position + 1:
                    forward_targets[j] = q_value
                else:
                    forward_targets[j] = self.train_data_set.center_return_values[i] - \
                                         self.train_data_set.forward_return_values[i, j] * \
                                         self.train_data_set.forward_discounts[i, j] + \
                                         self.train_data_set.forward_discounts[i, j] * \
                                         np.max(target_q_table[i, j])

                if self.train_data_set.backward_positions[i, j] == center_position + 1:
                    backward_targets[j] = q_value
                else:
                    backward_targets[j] = (-self.train_data_set.backward_return_values[i, j] +
                                           self.train_data_set.backward_discounts[i, j] *
                                           self.train_data_set.center_return_values[i] +
                                           target_q_table[i, self.flags.nob + j,
                                                          self.train_data_set.backward_actions[i, j]]) / \
                                          self.train_data_set.backward_discounts[i, j]

            forward_targets = np.append(forward_targets, self.train_data_set.center_return_values[i])
            v0 = v1 = forward_targets[0]
            v_max = np.max(forward_targets[1:])
            v_min = np.min(backward_targets)
            if v_max - 0.1 > q_value > v_min + 0.1:
                v1 = v_max * 0.5 + v_min * 0.5
            elif v_max - 0.1 > q_value:
                v1 = v_max
            elif v_min + 0.1 < q_value:
                v1 = v_min

            states1[i] = self.train_data_set.center_imgs[i]
            actions1[i] = self.train_data_set.center_actions[i]
            targets1[i] = v0 * self.flags.pw + (1 - self.flags.pw) * v1

        return self.network.train(states1, actions1, targets1)


