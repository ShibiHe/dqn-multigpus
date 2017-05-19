__author__ = 'frankhe'
import numpy as np
import image_preprocessing


class Interaction(object):
    def __init__(self, pid, environment, agent, flags, message_queue):
        """
        :param environment:
         bool game_over()
         void reset_game()
         int act(action)      action 0: null action
         getScreenGrayscale() screen_data = np.empty((height,width,1), dtype=np.uint8)
         getScreenDims() (width, height)   (160 210) in Atari
         int lives()
         getMinimalActionSet()  [int]
        """
        self.pid = pid
        self.env = environment
        self.width, self.height = self.env.getScreenDims()
        self.agent = agent
        self.flags = flags
        self.message_queue = message_queue

        # terminal when lost of lives == True: do not need to reset env ; == False reset
        self.terminal_lol = False
        self.min_action_set = self.env.getMinimalActionSet()
        self.buffer_length = self.flags.buffer_length
        self.buffer_count = 0
        self.screen_buffer = np.empty(
            shape=(self.buffer_length, self.height, self.width), dtype=np.uint8)

    def start(self):
        for epoch in xrange(1, self.flags.epochs + 1):
            self.run_epoch(epoch, self.flags.steps_per_epoch)
            self.agent.finish_epoch(epoch)
            if self.flags.test_length > 0:
                self.agent.start_testing()
                self.run_epoch(epoch, self.flags.test_length, True)
                self.agent.finish_testing(epoch)

    def run_epoch(self, epoch, num_steps, testing=False):
        self.terminal_lol = False  # make sure reset env at the start of an epoch
        steps_left = num_steps
        while steps_left > 0:
            prefix = 'TEST' if testing else 'TRAIN'
            message = [self.pid, 'step', [prefix, epoch, steps_left]]
            self.message_queue.put(message)
            # print 'PID:', self.pid, prefix, 'epoch:', str(epoch), 'steps_left:', str(steps_left)
            num_steps = self.run_episode(steps_left, testing)
            steps_left -= num_steps

    def run_episode(self, max_steps, testing):
        if not self.terminal_lol or self.env.game_over():
            self.env.reset_game()
            # no-ops
            no_ops = np.random.randint(self.buffer_length - 2, self.flags.max_start_no_op+1)
            for _ in xrange(no_ops):
                self._act(0)  # null action
            # Make sure the screen buffer is filled at the beginning of each episode.
            self._act(0)
            self._act(0)
        start_lives = self.env.lives()
        action = self.agent.start_episode(self.get_observation())
        num_steps = 0
        while True:
            reward = self._step(self.min_action_set[action])
            self.terminal_lol = self.flags.lol_end and not testing and (self.env.lives() < start_lives)
            terminal = self.env.game_over() or self.terminal_lol
            num_steps += 1
            if terminal or num_steps >= max_steps:
                self.agent.end_episode(reward, terminal)
                break
            action = self.agent.step(reward, self.get_observation())
        return num_steps

    def _act(self, action):
        reward = self.env.act(action)
        index = self.buffer_count % self.buffer_length
        self.env.getScreenGrayscale(self.screen_buffer[index, ...])
        self.buffer_count += 1
        return reward

    def _step(self, action):
        reward = 0
        for _ in xrange(self.flags.frame_skip):
            reward += self._act(action)
        return reward

    def get_observation(self):
        assert self.buffer_count >= self.buffer_length
        index = self.buffer_count % self.buffer_length - 1
        max_image = self.screen_buffer[index]
        for i in xrange(self.buffer_length):
            max_image = np.maximum(max_image, self.screen_buffer[index-i, ...])
        return image_preprocessing.resize(max_image, size=(self.flags.input_height, self.flags.input_width))





