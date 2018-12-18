from rl.core import Env

from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

FUNCTIONS = actions.FUNCTIONS


class Sc2Env(Env):
    env: sc2_env.SC2Env = None
    last_obs = None

    _ENV_NAME = "MoveToBeacon"
    _SCREEN = 16
    _MINIMAP = 16
    _VISUALIZE = False

    def __init__(self):
        print("init SC2")
        self.env = sc2_env.SC2Env(
            map_name=self._ENV_NAME,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=self._SCREEN,
                    minimap=self._MINIMAP
                ),
                use_feature_units=True
            ),
            step_mul=20,
            game_steps_per_episode=0,
            visualize=self._VISUALIZE
        )

    def action_to_sc2(self, act):

        real_action = FUNCTIONS.no_op()

        if act >= 1:
            if 331 in self.last_obs.observation.available_actions:

                trans_act = act - 1

                x_coord = int(trans_act / self._SCREEN)
                y_coord = int(trans_act % self._SCREEN)

                real_action = FUNCTIONS.Move_screen("now", (x_coord, y_coord))

        return real_action

    def step(self, action):
        # print(action, " ACTION")

        real_action = self.action_to_sc2(action)

        observation = self.env.step(actions=(real_action,))
        # print("stepped", observation[0].observation["feature_screen"][5])
        self.last_obs = observation[0]
        small_observation = observation[0].observation["feature_screen"][5]
        # small_observation = small_observation.reshape(1, small_observation.shape[0], small_observation.shape[0], 1)

        return small_observation, observation[0].reward, observation[0].last(), {}

    def reset(self):
        self.env.reset()

        observation = self.env.step(actions=(FUNCTIONS.select_army("select"),))
        self.last_obs = observation[0]
        small_observation = observation[0].observation["feature_screen"][5]
        # small_observation = small_observation.reshape(1, small_observation.shape[0], small_observation.shape[0], 1)

        return small_observation

    def render(self, mode: str = 'human', close: bool = False):
        pass

    def close(self):
        if self.env:
            self.env.close()

    def seed(self, seed=None):
        if seed:
            self.env._random_seed = seed

    def configure(self, *args, **kwargs):

        switcher = {
            '_ENV_NAME': self.set_env_name,
            '_SCREEN': self.set_screen,
            '_MINIMAP': self.set_minimap,
            '_VISUALIZE': self.set_visualize,
        }

        if kwargs is not None:
            for key, value in kwargs:
                func = switcher.get(key, lambda: print)
                func(value)

    def set_env_name(self, name: str):
        self._ENV_NAME = name

    def set_screen(self, screen: int):
        self._SCREEN = screen

    def set_visualize(self, visualize: bool):
        self._VISUALIZE = visualize

    def set_minimap(self, minimap: int):
        self._MINIMAP = minimap
