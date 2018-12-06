from rl.policy import Policy
import numpy as np
from pysc2.lib import actions

FUNCTIONS = actions.FUNCTIONS


class Sc2Policy(Policy):

    def __init__(self, env, eps=0.1, testing=False):
        super(Sc2Policy, self).__init__()
        self.eps = eps
        self.sc2_env = env
        self.testing = testing

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert self.sc2_env
        assert q_values.ndim == 1
        nb_actions = 3
        nb_pixels = self.sc2_env._SCREEN

        action = 0
        x_coord = 0
        y_coord = 0

        if np.random.uniform() < self.eps and not self.testing:
            action = np.random.random_integers(0, nb_actions-1)
            if action == 2:
                x_coord = np.random.random_integers(0, nb_pixels-1)
                y_coord = np.random.random_integers(0, nb_pixels-1)

            # print("Rand sel Action ", action, "args ", arguments)
        else:
            action = np.argmax(q_values[0:2])
            if action == 2:
                x_coord = np.argmax(q_values[4:4+nb_pixels])
                y_coord = np.argmax(q_values[4+nb_pixels:])
                print(x_coord, y_coord, q_values[4:4 + nb_pixels], q_values[4 + nb_pixels:])

        # print(np.argmax(q_values[4:4+nb_pixels]), np.random.random_integers(0, nb_pixels-1))

        real_action = FUNCTIONS.no_op()

        if action == 0:
            if 0 in self.sc2_env.last_obs.observation.available_actions:
                real_action = FUNCTIONS.no_op()
        elif action == 1:
            if 7 in self.sc2_env.last_obs.observation.available_actions:
                real_action = FUNCTIONS.select_army("select")
        elif action == 2:
            if 331 in self.sc2_env.last_obs.observation.available_actions:
                real_action = FUNCTIONS.Move_screen("now", (x_coord, y_coord))

        return real_action


    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(Sc2Policy, self).get_config()
        config['eps'] = self.eps
        config['testing'] = self.testing
        return config