from absl import app
from env import Sc2Env1Output, Sc2Env2Outputs
from SC2DqnAgent import SC2DQNAgent
from sc2Processor import Sc2Processor
from sc2Policy import Sc2Policy
import numpy
import traceback
import os
import json
import baselines.deepq.replay_buffer as rpb

from pysc2.env import sc2_env
from pysc2.lib import features

from pysc2.agents.scripted_agent import MoveToBeacon
from pysc2.agents.random_agent import RandomAgent

import keras.layers

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, Conv2D, MaxPooling2D

from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


_ENV_NAME = "CollectMineralShards"
_SCREEN = 32
_MINIMAP = 16

_VISUALIZE = False
_TEST = False


# IDEAS:
# start 50% of episodes with selected army


def __main__(unused_argv):

    fully_conf_q_agent_4()


def fully_conf_q_agent_4():
    try:
        env = Sc2Env2Outputs(screen=_SCREEN, visualize=_VISUALIZE, env_name=_ENV_NAME, training=not _TEST)
        env.seed(666)
        numpy.random.seed(666)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = 3
        agent_name = "fullyConv_v4"
        run_name = "03"

        # print(nb_actions)

        main_input = Input(shape=(2, env.screen, env.screen), name='main_input')
        permuted_input = Permute((2, 3, 1))(main_input)
        x = Conv2D(16, (5, 5), padding='same', activation='relu')(permuted_input)
        branch = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

        coord_out = Conv2D(1, (1, 1), padding='same', activation='relu')(branch)

        act_out = Flatten()(branch)
        act_out = Dense(256, activation='relu')(act_out)
        # act_out = Flatten()(act_out)
        act_out = Dense(nb_actions, activation='linear')(act_out)

        full_conv_sc2 = Model(main_input, [act_out, coord_out])

        print(act_out.shape)
        print(coord_out.shape)
        # print(full_conv_sc2.summary())

        memory = SequentialMemory(limit=1000000, window_length=1)
        # policy = BoltzmannQPolicy()
        policy = LinearAnnealedPolicy(Sc2Policy(env=env), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=300000)

        test_policy = Sc2Policy(env=env, eps=0.005)
        # policy = Sc2Policy(env)
        processor = Sc2Processor(screen=env._SCREEN)

        dqn = SC2DQNAgent(model=full_conv_sc2, nb_actions=nb_actions, screen_size=env._SCREEN,
                          enable_dueling_network=False, memory=memory, processor=processor, nb_steps_warmup=100,
                          enable_double_dqn=True,
                          policy=policy, test_policy=test_policy, gamma=.995, target_model_update=10000,
                          train_interval=4, delta_clip=1.)

        dqn.compile(Adam(lr=0.00025), metrics=['mae'])

        directory = "weights/{}_{}_{}".format(agent_name, _ENV_NAME, run_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_filename = directory + '/dqn_weights.h5f'
        checkpoint_weights_filename = directory + '/dqn_weights_{step}.h5f'
        log_filename = directory + '/dqn_log.json'

        save_hyper_parameters(full_conv_sc2, env, directory)

        if _TEST:
            dqn.load_weights('/home/benjamin/PycharmProjects/dqn/weights/'
                             'fullyConv_v4_CollectMineralShards_01/dqn_weights_2550000.h5f')
            dqn.test(env, nb_episodes=20, visualize=True)
        else:

            # dqn.load_weights('finalWeights/dqn_MoveToBeacon_weights_6300000_fullyConv_v1.h5f')
            # dqn.step = 6300000

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=50000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, nb_steps=10000000, nb_max_start_steps=0, callbacks=callbacks, log_interval=10000,
                    action_repetition=3)

            dqn.save_weights(weights_filename, overwrite=True)


    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        traceback.print_exc()
        pass


def fully_conf_q_agent():
    try:
        env = Sc2Env2Outputs()
        env.seed(666)
        numpy.random.seed(666)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = 2
        agent_name = "fullyConv_v3"
        run_name = "01"

        # print(nb_actions)

        main_input = Input(shape=(1, env._SCREEN, env._SCREEN), name='main_input')
        permuted_input = Permute((2, 3, 1))(main_input)
        x = Conv2D(16, (5, 5), padding='same', activation='relu')(permuted_input)
        branch = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

        coord_out = Conv2D(1, (1, 1), padding='same', activation='relu')(branch)

        act_out = Flatten()(branch)
        act_out = Dense(256, activation='relu')(act_out)
        # act_out = Flatten()(act_out)
        act_out = Dense(nb_actions, activation='linear')(act_out)

        full_conv_sc2 = Model(main_input, [act_out, coord_out])

        print(act_out.shape)
        print(coord_out.shape)
        # print(full_conv_sc2.summary())

        memory = SequentialMemory(limit=1000000, window_length=1)
        # policy = BoltzmannQPolicy()
        policy = LinearAnnealedPolicy(Sc2Policy(env=env), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=300000)

        test_policy = Sc2Policy(env=env, eps=0.005)
        # policy = Sc2Policy(env)
        # processor = Sc2Processor()

        dqn = SC2DQNAgent(model=full_conv_sc2, nb_actions=nb_actions, screen_size=env._SCREEN,
                          enable_dueling_network=False, memory=memory, nb_steps_warmup=10000, enable_double_dqn=False,
                          policy=policy, test_policy=test_policy, gamma=.99, target_model_update=10000,
                          train_interval=4, delta_clip=1.)

        dqn.compile(Adam(lr=0.00025), metrics=['mae'])

        directory = "weights/{}_{}_{}".format(agent_name, _ENV_NAME, run_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_filename = directory + '/dqn_weights.h5f'
        checkpoint_weights_filename = directory + '/dqn_weights_{step}.h5f'
        log_filename = directory + '/dqn_log.json'

        save_hyper_parameters(full_conv_sc2, env, directory)

        if _TEST:
            dqn.load_weights('finalWeights/dqn_MoveToBeacon_weights_6300000_fullyConv_v1.h5f')
            dqn.test(env, nb_episodes=20, visualize=True)
        else:

            # dqn.load_weights('finalWeights/dqn_MoveToBeacon_weights_6300000_fullyConv_v1.h5f')
            # dqn.step = 6300000

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=50000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, nb_steps=10000000, nb_max_start_steps=0, callbacks=callbacks, log_interval=10000,
                    action_repetition=3)

            dqn.save_weights(weights_filename, overwrite=True)


    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        traceback.print_exc()
        pass


def seq_q_agent_5():
    try:
        env = Sc2Env1Output(screen=_SCREEN, visualize=_VISUALIZE, env_name=_ENV_NAME, training=not _TEST)
        env.seed(42)
        numpy.random.seed(42)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = 1 + env._SCREEN * env._SCREEN * 2

        print(nb_actions)

        agent_name = "seq_v5"
        run_name = "04"

        main_input = Input(shape=(2, env._SCREEN, env._SCREEN), name='main_input')
        permuted_input = Permute((2, 3, 1))(main_input)

        # tower_1 = Conv2D(1, (1, 1), padding='same', activation='tanh')(permuted_input)

        dense1 = Flatten()(permuted_input)
        dense1 = Dense(env._SCREEN, activation='relu')(dense1)
        dense1 = Dense(env._SCREEN, activation='relu')(dense1)
        dense1 = Dense(env._SCREEN, activation='relu')(dense1)
        dense1 = Dense(nb_actions, activation='relu')(dense1)

        model = Model(main_input, dense1)
        print(model.summary())

        memory = SequentialMemory(limit=5000000, window_length=1)
        # memory = rpb.PrioritizedReplayBuffer(1000000, 0.7)
        # policy = BoltzmannQPolicy()
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=1000000)
        # policy = EpsGreedyQPolicy()
        # policy = Sc2Policy(env)

        processor = Sc2Processor(screen=env._SCREEN)

        dqn = DQNAgent(model=model, nb_actions=nb_actions, enable_dueling_network=True, memory=memory,
                       nb_steps_warmup=10000, enable_double_dqn=True, processor=processor,
                       policy=policy, gamma=.999, target_model_update=10000, train_interval=4, delta_clip=1.)

        dqn.compile(Adam(lr=0.001), metrics=['mae'])

        directory = "weights/{}_{}_{}".format(agent_name, _ENV_NAME, run_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_filename = directory + '/dqn_weights.h5f'
        checkpoint_weights_filename = directory + '/dqn_weights_{step}.h5f'
        log_filename = directory + '/dqn_log.json'

        save_hyper_parameters(model, env, directory)

        if _TEST:
            dqn.load_weights('dqn_MoveToBeacon_weights_4800000_ol.h5f')
            dqn.test(env, nb_episodes=20, visualize=True)

        else:

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=30000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, nb_steps=10000000, nb_max_start_steps=0, callbacks=callbacks, log_interval=10000)

            dqn.save_weights(weights_filename, overwrite=True)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        traceback.print_exc()
        pass


def seq_q_agent_4():
    try:
        env = Sc2Env1Output(screen=_SCREEN, visualize=_VISUALIZE, env_name=_ENV_NAME)
        env.seed(42)
        numpy.random.seed(42)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = 1 + env._SCREEN * env._SCREEN * 2

        print(nb_actions)

        agent_name = "seq_v4"
        run_name = "02"

        main_input = Input(shape=(2, env._SCREEN, env._SCREEN), name='main_input')
        permuted_input = Permute((2, 3, 1))(main_input)

        tower_1 = Conv2D(16, (5, 5), padding='same', activation='relu')(permuted_input)
        tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)
        tower_1 = Conv2D(1, (1, 1), padding='same', activation='relu')(tower_1)

        dense1 = Flatten()(tower_1)
        dense1 = Dense(nb_actions, activation='relu')(dense1)

        model = Model(main_input, dense1)
        print(model.summary())

        memory = SequentialMemory(limit=1000000, window_length=1)
        # policy = BoltzmannQPolicy()
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=2000000)
        # policy = EpsGreedyQPolicy()
        # policy = Sc2Policy(env)

        processor = Sc2Processor(screen=env._SCREEN)

        dqn = DQNAgent(model=model, nb_actions=nb_actions, enable_dueling_network=True, memory=memory,
                       nb_steps_warmup=10000, enable_double_dqn=True, processor=processor,
                       policy=policy, gamma=.99, target_model_update=10000, train_interval=2, delta_clip=1.)

        dqn.compile(Adam(lr=0.00025), metrics=['mae'])

        directory = "weights/{}_{}_{}".format(agent_name, _ENV_NAME, run_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_filename = directory + '/dqn_weights.h5f'
        checkpoint_weights_filename = directory + '/dqn_weights_{step}.h5f'
        log_filename = directory + '/dqn_log.json'

        save_hyper_parameters(model, env, directory)

        if _TEST:
            dqn.load_weights('dqn_MoveToBeacon_weights_4800000_ol.h5f')
            dqn.test(env, nb_episodes=20, visualize=True)
        else:

            dqn.load_weights('/home/benjamin/PycharmProjects/dqn/weights/'
                             'seq_v4_CollectMineralShards_02/dqn_weights_3780000_ol.h5f')

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=30000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, nb_steps=10000000, nb_max_start_steps=0, callbacks=callbacks, log_interval=10000)

            dqn.save_weights(weights_filename, overwrite=True)


    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        traceback.print_exc()
        pass


# should be capable of MTB and CMS but is not really
def seq_q_agent_3():
    try:
        env = Sc2Env1Output(screen=_SCREEN, visualize=_VISUALIZE)
        env.seed(2)
        numpy.random.seed(2)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = 1 + env._SCREEN * env._SCREEN

        print(nb_actions)

        main_input = Input(shape=(2, env._SCREEN, env._SCREEN), name='main_input')
        permuted_input = Permute((2, 3, 1))(main_input)

        tower_1 = Conv2D(16, (5, 5), padding='same', activation='relu')(permuted_input)
        tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)

        dense1 = Flatten()(tower_1)
        dense1 = Dense(env._SCREEN * env._SCREEN, activation='relu')(dense1)
        dense1 = Dense(nb_actions, activation='relu')(dense1)

        model = Model(main_input, dense1)
        print(model.summary())

        memory = SequentialMemory(limit=1000000, window_length=1)
        # policy = BoltzmannQPolicy()
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=1000000)
        # policy = Sc2Policy(env)
        processor = Sc2Processor(screen=env._SCREEN)

        dqn = DQNAgent(model=model, nb_actions=nb_actions, enable_dueling_network=True, memory=memory,
                       nb_steps_warmup=10000, enable_double_dqn=True, processor=processor,
                       policy=policy, gamma=.999, target_model_update=10000, train_interval=2, delta_clip=1.)

        dqn.compile(Adam(lr=0.00025), metrics=['mae'])

        weights_filename = 'dqn_{}_weights.h5f'.format(_ENV_NAME)
        checkpoint_weights_filename = 'dqn_' + _ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(_ENV_NAME)

        if _TEST:
            dqn.load_weights('dqn_MoveToBeacon_weights_1890000.h5f')
            dqn.test(env, nb_episodes=20, visualize=True)
        else:

            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=30000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, nb_steps=10000000, nb_max_start_steps=0, callbacks=callbacks, log_interval=10000)

            dqn.save_weights(weights_filename, overwrite=True)


    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        traceback.print_exc()
        pass


# only works for MoveToBeacon
def naive_sequential_q_agent_2():
    try:
        env = Sc2Env1Output(screen=_SCREEN, visualize=_VISUALIZE)
        env.seed(2)
        numpy.random.seed(2)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = env._SCREEN * env._SCREEN + 1

        print(nb_actions)

        main_input = Input(shape=(1, env._SCREEN, env._SCREEN), name='main_input')
        permuted_input = Permute((2, 3, 1))(main_input)

        tower_1 = Conv2D(16, (5, 5), padding='same', activation='relu')(permuted_input)
        tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)

        dense1 = Flatten()(tower_1)
        dense1 = Dense(nb_actions, activation='relu')(dense1)
        dense1 = Dense(nb_actions, activation='relu')(dense1)

        model = Model(main_input, dense1)
        print(model.summary())

        memory = SequentialMemory(limit=1000000, window_length=1)
        # policy = BoltzmannQPolicy()
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=300000)
        # policy = Sc2Policy(env)
        # processor = Sc2Processor()

        dqn = DQNAgent(model=model, nb_actions=nb_actions, enable_dueling_network=True, memory=memory,
                       nb_steps_warmup=10000, enable_double_dqn=True,
                       policy=policy, gamma=.999, target_model_update=10000, train_interval=2, delta_clip=1., )

        dqn.compile(Adam(lr=0.00025), metrics=['mae'])

        weights_filename = 'dqn_{}_weights.h5f'.format(_ENV_NAME)
        checkpoint_weights_filename = 'dqn_' + _ENV_NAME + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(_ENV_NAME)

        if _TEST:
            dqn.load_weights('weights/seq_v2_20step_16/dqn_MoveToBeacon_weights_630000.h5f')
            dqn.test(env, nb_episodes=20, visualize=True)
        else:
            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=30000)]
            callbacks += [FileLogger(log_filename, interval=100)]
            dqn.fit(env, nb_steps=3000000, nb_max_start_steps=0, callbacks=callbacks, log_interval=10000,
                    action_repetition=3)

            dqn.save_weights(weights_filename, overwrite=True)


    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(e)
        traceback.print_exc()
        pass


def naive_sequential_q_agent():
    try:
        env = Sc2Env1Output()
        env.seed(1234)
        numpy.random.seed(123)

        #    0/no_op                                              ()
        #    7/select_army                                        (7/select_add [2])
        #  331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])

        nb_actions = env._SCREEN * env._SCREEN + 1

        print(nb_actions)

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
        traceback.print_exc()
        pass


def simple_scripted_agent():
    episodes = 0
    agent = MoveToBeacon()
    # agent = RandomAgent()

    try:
        env = sc2_env.SC2Env()

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


def save_hyper_parameters(model, env, path):
    net = []
    for layer in model.layers:
        net.append(layer.get_output_at(0).get_shape().as_list())

    hyper = {
        "ENV_NAME": _ENV_NAME,
        "_SCREEN": _SCREEN,
        "_MINIMAP": _MINIMAP,
        "_TEST": _TEST,
        "_NETWORK": net,
        "_ENV_STEP": env.env._step_mul
    }

    with open(path + '/hyper.json', 'w') as outfile:
        json.dump(hyper, outfile)

if __name__ == '__main__':
    app.run(__main__)

