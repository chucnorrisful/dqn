from absl import app
from env import Sc2Env
from SC2DqnAgent import SC2DQNAgent
from sc2Processor import Sc2Processor
from sc2Policy import Sc2Policy
import numpy

from pysc2.env import sc2_env
from pysc2.lib import features

from pysc2.agents.scripted_agent import MoveToBeacon
from pysc2.agents.random_agent import RandomAgent

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Conv2D

from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

_ENV_NAME = "MoveToBeacon"
_SCREEN = 24
_MINIMAP = 24
_VISUALIZE = False
_EPISODES = 1000

_TEST = False


def __main__(unused_argv):

    try:
        env = Sc2Env()
        env.seed(1234)
        numpy.random.seed(123)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = 2

        # print(nb_actions)

        main_input = Input(shape=(1, env._SCREEN, env._SCREEN), name='main_input')
        permuted_input = Permute((2, 3, 1))(main_input)
        x = Conv2D(16, (5, 5), padding='same', activation='relu')(permuted_input)
        branch = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

        coord_out = Conv2D(1, (1, 1), padding='same', activation='relu')(branch)

        # act_out = Flatten(branch)
        act_out = Dense(256, activation='relu')(branch)
        act_out = Flatten()(act_out)
        act_out = Dense(nb_actions, activation='linear')(act_out)

        full_conv_sc2 = Model(main_input, [act_out, coord_out])

        print(act_out.shape)
        print(coord_out.shape)
        print(full_conv_sc2.summary())

        memory = SequentialMemory(limit=1000000, window_length=1)
        # policy = BoltzmannQPolicy()
        policy = LinearAnnealedPolicy(Sc2Policy(env=env), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=1000000)
        # policy = Sc2Policy(env)
        # processor = Sc2Processor()

        dqn = SC2DQNAgent(model=full_conv_sc2, nb_actions=nb_actions, screen_size=env._SCREEN, enable_dueling_network=False, memory=memory,
                       nb_steps_warmup=1000, enable_double_dqn=True,
                       policy=policy, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)

        dqn.compile(Adam(lr=0.00025), metrics=['mae'])

        weights_filename = 'dqn_{}_weights.h5f'.format(_ENV_NAME)
        checkpoint_weights_filename = 'dqn_' + _ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(_ENV_NAME)

        if _TEST:
            dqn.load_weights('dqn_MoveToBeacon_weights_3000000.h5f')
            dqn.test(env, nb_episodes=10, visualize=False)
        else:
            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=30000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, nb_steps=3000000, nb_max_start_steps=0, callbacks=callbacks, log_interval=10000)

            dqn.save_weights(weights_filename, overwrite=True)


    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        pass


def naive_sequential_q_agent():
    try:
        env = Sc2Env()
        env.seed(1234)
        numpy.random.seed(123)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = env._SCREEN * env._SCREEN + 1

        print(nb_actions)
        # TODO: Permute and remove channels_first
        model = Sequential()
        model.add(Convolution2D(32, 4, input_shape=(1, env._SCREEN, env._SCREEN), data_format='channels_first'))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 4, data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 4, data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        print(model.summary())

        memory = SequentialMemory(limit=1000000, window_length=1)
        # policy = BoltzmannQPolicy()
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=300000)
        # policy = Sc2Policy(env)
        # processor = Sc2Processor()

        dqn = DQNAgent(model=model, nb_actions=nb_actions, enable_dueling_network=True, memory=memory,
                       nb_steps_warmup=1000, enable_double_dqn=True,
                       policy=policy, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)

        dqn.compile(Adam(lr=0.00025), metrics=['mae'])

        weights_filename = 'dqn_{}_weights.h5f'.format(_ENV_NAME)
        checkpoint_weights_filename = 'dqn_' + _ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(_ENV_NAME)

        if _TEST:
            dqn.load_weights('finalWeights/dqn_MoveToBeacon_weights_2310000_16dim_20step.h5f')
            dqn.test(env, nb_episodes=10, visualize=False)
        else:
            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=30000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, nb_steps=3000000, nb_max_start_steps=0, callbacks=callbacks, log_interval=10000)

            dqn.save_weights(weights_filename, overwrite=True)


    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        pass

def simple_scripted_agent():
    episodes = 0
    agent = MoveToBeacon()
    # agent = RandomAgent()

    try:
        env = Sc2Env()

        env.seed(1234)
        numpy.random.seed(123)

        print("setup")

        # obs = env.env.observation_spec()
        # act = env.env.action_spec()

        agent.setup(env.env.observation_spec(), env.env.action_spec())

        timesteps = env.reset()
        agent.reset()

        print(timesteps)

        while True:
            step_actions = [agent.step(timesteps[0])]
            if timesteps[0].last():
                break
            timesteps = env.step(step_actions)

        print("end")
        print(step_actions)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        pass


if __name__ == '__main__':
    app.run(__main__)

# # Get the environment and extract the number of actions.
# env = sc2_env.SC2Env(
#     map_name=_ENV_NAME,
#     players=[sc2_env.Agent(sc2_env.Race.terran)],
#     agent_interface_format=features.AgentInterfaceFormat(
#         feature_dimensions=features.Dimensions(
#             screen=_SCREEN,
#             minimap=_MINIMAP
#         ),
#         use_feature_units=True
#     ),
#     step_mul=8,
#     visualize=_VISUALIZE
# )
#
# np.random.seed(123)
# env.seed(123)
# # nb_actions = env.action_space.n
#
# print(env.observation_spec())
# print(env.action_spec())

# Next, we build a very simple model.
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
# print(model.summary())
