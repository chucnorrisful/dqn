from pysc2.agents import base_agent
from pysc2.lib import actions, features

import numpy

FUNCTIONS = actions.FUNCTIONS
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class DqnAgent(base_agent.BaseAgent):

    def __init__(self):
        super(DqnAgent, self).__init__()

    def step(self, obs):
        super(DqnAgent, self).step(obs)
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            if not beacon:
                return FUNCTIONS.no_op()
            beacon_center = numpy.mean(beacon, axis=0).round()
            return FUNCTIONS.Move_screen("now", beacon_center)
        else:
            return FUNCTIONS.select_army("select")

    def setup(self, obs_spec, action_spec):
        super(DqnAgent, self).setup(obs_spec, action_spec)

    def reset(self):
        super(DqnAgent, self).reset()


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))
