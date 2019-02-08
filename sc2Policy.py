from rl.policy import Policy
import numpy as np
from sc2DqnAgent import Sc2Action


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
        # assert q_values.ndim == 1
        nb_actions = 3
        nb_pixels = self.sc2_env._SCREEN

        action = Sc2Action()

        if np.random.uniform() < self.eps and not self.testing:

            action.action = np.random.random_integers(0, nb_actions-1)
            action.coords = (np.random.random_integers(0, nb_pixels-1),  np.random.random_integers(0, nb_pixels-1))

            # print("Rand sel Action ", action, "args ", arguments)
        else:
            action.action = np.argmax(q_values[0])

            action.coords = np.unravel_index(q_values[1].argmax(), q_values[1].shape)[1:3]

            # action.coords = np.unravel_index(np.reshape(q_values[1][0][:][:], (16, 16)).argmax(), np.reshape(
            # q_values[1][0][:][:], (16, 16)).shape)

        assert len(action.coords) == 2

        return action

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(Sc2Policy, self).get_config()
        config['eps'] = self.eps
        config['testing'] = self.testing
        return config
