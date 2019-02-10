from __future__ import division
import warnings

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense, Conv2D, Flatten
from rl.agents.dqn import AbstractDQNAgent, Agent
from agent2 import Agent2
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from baselines.common.schedules import LinearSchedule

# modded, check
def mean_q(y_true, y_pred):
    mean_a = K.mean(K.max(y_pred[0], axis=-1))
    mean_b = K.mean(K.max(y_pred[1], axis=(1, 2)))
    return K.mean(mean_a, mean_b)


class Sc2Action:

    # default: noop
    def __init__(self, act=0, x=0, y=0):
        self.coords = (x, y)
        self.action = act


# V2 -> patched to use prioritized replay buffer as memory


class AbstractSc2DQNAgent2(Agent2):
    """Write me
    """
    def __init__(self, nb_actions, screen_size, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=10000,
                 delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):
        super(AbstractSc2DQNAgent2, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.screen_size = screen_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        # assert q_values.shape == (len(state_batch), self.nb_actions) (len(state_batch), 2)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state])
        # q_values = self.compute_batch_q_values([state]).flatten()
        # assert q_values.shape == (2, 1) ?
        return q_values

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'screen_size': self.screen_size,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }


# A modified version of the Keras-rl DQN Agent to handle a much larger actionspace (multiple outputs required)
class Sc2DqnAgent_v2(AbstractSc2DQNAgent2):
    """
    # Arguments
        model__: A Keras model.
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enable target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn__: A boolean which enable dueling architecture proposed by Mnih et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581).
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)

    """

    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(Sc2DqnAgent_v2, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) != 2:
            raise ValueError(
                'Model "{}" has more or less than two outputs. DQN expects a model that has exactly 2 outputs.'.format(
                    model))

        # no shape checks yet
        # if model.output[0]._keras_shape != (None, self.nb_actions):
        #     raise ValueError(
        #         'Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action'
        #         ', in this case {}.'.format(model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type

        # dueling not working // dueling not learning
        if self.enable_dueling_network:

            # linearOutput
            # get the second last layer of the model, abandon the last layer
            # layer = model.layers[-2]
            layer = model.layers[5]
            # nb_action = model.output._keras_shape[-1]
            nb_action = model.output[0]._keras_shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                lin_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                     output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                lin_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                     output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                lin_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            # conv layer > include 1,1x1 conv (?) [yes didnt work now trying no]
            conv_layer = model.layers[3].output
            # conv_size = model.output[1]._keras_shape[1]

            conv_flat = Flatten()(conv_layer)
            conv_value = Dense(1, activation="linear")(conv_flat)
            conv_action = Conv2D(1, (1, 1), padding="same", activation="linear")(conv_layer)

            conv_lambda_in = [conv_value, conv_action]

            if self.dueling_type == 'avg':
                conv_outputlayer = Lambda(
                    lambda a: K.expand_dims(K.expand_dims(a[0], -1), -1) + a[1] - K.mean(a[1], keepdims=True)
                                          )(conv_lambda_in)
            elif self.dueling_type == 'max':
                conv_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                     output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                conv_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(inputs=model.input, outputs=[lin_outputlayer, conv_outputlayer])

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.beta_schedule = LinearSchedule(100000,
                                       initial_p=0.5,
                                       final_p=1.0)

        # State.
        self.reset_states()

    def get_config(self):
        config = super(Sc2DqnAgent_v2, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true_a, y_true_b, y_pred_a, y_pred_b, mask_a, mask_b = args
            loss = [huber_loss(y_true_a, y_pred_a, self.delta_clip),
                    huber_loss(y_true_b, y_pred_b, self.delta_clip)]
            loss[0] *= mask_a  # apply element-wise mask
            loss[1] *= mask_b  # apply element-wise mask
            sum_loss_a = K.sum(loss[0])
            sum_loss_b = K.sum(loss[1])
            return K.sum([sum_loss_a, sum_loss_b], axis=-1)

        def clipped_masked_error_v2(args):
            y_true_a, y_true_b, y_pred_a, y_pred_b, mask_a, mask_b = args
            loss = [huber_loss(y_true_a, y_pred_a, self.delta_clip),
                    huber_loss(y_true_b, y_pred_b, self.delta_clip)]
            loss[0] *= mask_a  # apply element-wise mask
            loss[1] *= mask_b  # apply element-wise mask
            sum_loss_a = K.sum(loss[0])
            sum_loss_a = sum_loss_a * 0
            sum_loss_b = K.sum(loss[1])
            return K.sum([sum_loss_a, sum_loss_b], axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.

        y_pred = self.model.output
        print("lol")
        y_true_a = Input(name='y_true_a', shape=(self.nb_actions,))
        y_true_b = Input(name='y_true_b', shape=(self.screen_size, self.screen_size, 1))
        mask_a = Input(name='mask_a', shape=(self.nb_actions,))
        mask_b = Input(name='mask_b', shape=(self.screen_size, self.screen_size, 1))

        # Layer loss was called with an input that isn't a symbolic tensor. Received type: <class 'list'>.
        # Full input:
        # [<tf.Tensor 'y_true_a:0' shape=(?, 2, 1) dtype=float32>,
        # <tf.Tensor 'y_true_b:0' shape=(?, 16, 16, 1) dtype=float32>,
        # [<tf.Tensor 'dense_2/BiasAdd:0' shape=(?, 2) dtype=float32>,
        # <tf.Tensor 'conv2d_3/Relu:0' shape=(?, 16, 16, 1) dtype=float32>],
        # <tf.Tensor 'mask:0' shape=(?, 2, 1) dtype=float32>,
        # <tf.Tensor 'mask_1:0' shape=(?, 16, 16, 1) dtype=float32>].
        # All inputs to the layer should be tensors.
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true_a, y_true_b, y_pred[0], y_pred[1], mask_a, mask_b])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input

        trainable_model = Model(inputs=ins + [y_true_a, y_true_b, mask_a, mask_b], outputs=[loss_out, y_pred[0], y_pred[1]])
        print(trainable_model.summary())
        # assert len(trainable_model.output_names) == 2 what is this??
        # combined_metrics = {trainable_model.output_names[1]: metrics} i dunno ??
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses)  # metrics=combined_metrics
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    # modded
    def forward(self, observation):
        # Select an action.
        # state = self.memory.get_recent_state(observation
        state = [observation]
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal, observation_1):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.add(self.recent_observation, self.recent_action, reward,
                            observation_1, terminal)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size, self.beta_schedule.value(self.step))
            assert len(experiences[0]) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            action_batch = []
            reward_batch = []
            state1_batch = []
            terminal1_batch = []
            prio_batch = []
            id_batch = []

            experiences = zip(experiences[0], experiences[1], experiences[2], experiences[3], experiences[4],
                              experiences[5], experiences[6])

            for e in experiences:
                state0_batch.append(e[0])
                action_batch.append(e[1])
                reward_batch.append(e[2])
                state1_batch.append(e[3])
                terminal1_batch.append(0. if e[4] else 1.)
                prio_batch.append(e[5])
                id_batch.append(e[6])

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q1_values = self.model.predict_on_batch(state1_batch)

                actions_a = np.argmax(q1_values[0], -1)
                actions_b = []
                for ac_b in q1_values[1]:
                    actions_b.append(np.unravel_index(ac_b.argmax(), ac_b.shape)[0:2])
                actions_b = np.array(actions_b)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q1_values = self.target_model.predict_on_batch(state1_batch)

                q_batch_a = target_q1_values[0][range(self.batch_size), actions_a]
                q_batch_b = []
                for (i, square_q) in enumerate(target_q1_values[1]):
                    q_batch_b.append(square_q[:, :, 0][actions_b[i][0], actions_b[i][1]])

                q_batch_b = np.array(q_batch_b)
            else:

                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                # target_q_values = self.target_model.predict_on_batch(state1_batch)

                target_q1_values = self.target_model.predict_on_batch(state1_batch)

                q_batch_a = np.max(target_q1_values[0], axis=-1)
                q_batch_b = np.max(target_q1_values[1], axis=(1, 2))[:, 0]

                q_batch_a = np.array(q_batch_a)
                q_batch_b = np.array(q_batch_b)

            targets_a = np.zeros((self.batch_size, self.nb_actions,))
            targets_b = np.zeros((self.batch_size, self.screen_size, self.screen_size, 1))

            dummy_targets_a = np.zeros((self.batch_size,))
            dummy_targets_b = np.zeros((self.batch_size,))

            masks_a = np.zeros((self.batch_size, self.nb_actions,))
            masks_b = np.zeros((self.batch_size, self.screen_size, self.screen_size, 1))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch_a = self.gamma * q_batch_a
            discounted_reward_batch_b = self.gamma * q_batch_b

            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch_a = discounted_reward_batch_a * terminal1_batch[:]
            discounted_reward_batch_b = discounted_reward_batch_b * terminal1_batch[:]
            # INFO: try np.einsum('ij,i->ij',A,b)
            # assert discounted_reward_batch.shape == reward_batch.shape nope
            Rs_a = reward_batch[:] + discounted_reward_batch_a
            Rs_b = reward_batch[:] + discounted_reward_batch_b
            for idx, (target_a, target_b, mask_a, mask_b, R_a, R_b, action) in \
                    enumerate(zip(targets_a, targets_b, masks_a, masks_b, Rs_a, Rs_b, action_batch)):
                target_a[action.action] = R_a  # update action with estimated accumulated reward
                if len(action.coords) != 2:
                    print(action.coords)
                target_b[action.coords] = R_b  # update action with estimated accumulated reward
                dummy_targets_a[idx] = R_a
                dummy_targets_b[idx] = R_b
                mask_a[action.action] = 1.  # enable loss for this specific action
                mask_b[action.coords] = 1.  # enable loss for this specific action
            targets_a = np.array(targets_a).astype('float32')
            targets_b = np.array(targets_b).astype('float32')
            masks_a = np.array(masks_a).astype('float32')
            masks_b = np.array(masks_b).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch

            pred = self.trainable_model.predict_on_batch(ins + [targets_a, targets_b, masks_a, masks_b])
            metrics = self.trainable_model.train_on_batch(ins + [targets_a, targets_b, masks_a, masks_b],
                                                          [np.zeros(self.batch_size), targets_a, targets_b])
            metrics = [metric for idx, metric in enumerate(metrics) if
                       idx not in (1, 2)]  # throw away individual losses

            # update priority batch
            prios = []
            for pre in zip(pred[1], pred[2]):
                loss = [target_a - pre[0],
                        target_b - pre[1]]
                loss[0] *= mask_a  # apply element-wise mask
                loss[1] *= mask_b  # apply element-wise mask
                sum_loss_a = np.sum(loss[0])
                sum_loss_b = np.sum(loss[1])
                prios.append(abs(np.sum([sum_loss_a, sum_loss_b], axis=-1)))

            self.memory.update_priorities(id_batch, prios)

            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 3
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)


# V1 -> works with standard keras memory


class AbstractSc2DQNAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, screen_size, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=10000,
                 delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):
        super(AbstractSc2DQNAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.screen_size = screen_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        # assert q_values.shape == (len(state_batch), self.nb_actions) (len(state_batch), 2)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state])
        # q_values = self.compute_batch_q_values([state]).flatten()
        # assert q_values.shape == (2, 1) ?
        return q_values

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'screen_size': self.screen_size,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }


# A modified version of the Keras-rl DQN Agent to handle a much larger actionspace (multiple outputs required)
class SC2DQNAgent(AbstractSc2DQNAgent):
    """
    # Arguments
        model__: A Keras model.
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enable target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn__: A boolean which enable dueling architecture proposed by Mnih et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581).
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)

    """

    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(SC2DQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) != 2:
            raise ValueError(
                'Model "{}" has more or less than two outputs. DQN expects a model that has exactly 2 outputs.'.format(
                    model))

        # no shape checks yet
        # if model.output[0]._keras_shape != (None, self.nb_actions):
        #     raise ValueError(
        #         'Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action'
        #         ', in this case {}.'.format(model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type

        # dueling not working // dueling not learning
        if self.enable_dueling_network:

            # linearOutput
            # get the second last layer of the model, abandon the last layer
            # layer = model.layers[-2]
            layer = model.layers[5]
            # nb_action = model.output._keras_shape[-1]
            nb_action = model.output[0]._keras_shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                lin_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                     output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                lin_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                                     output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                lin_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            # conv layer > include 1,1x1 conv (?) [yes didnt work now trying no]
            # conv_layer = model.layers[3].output
            # conv_size = model.output[1]._keras_shape[1]

            # conv_flat = Flatten()(conv_layer)
            # conv_value = Dense(1, activation="linear")(conv_flat)
            # conv_action = Conv2D(1, (1, 1), padding="same", activation="linear")(conv_layer)

            # conv_lambda_in = [conv_value, conv_action]
            #
            # if self.dueling_type == 'avg':
            #     conv_outputlayer = Lambda(
            #         lambda a: K.expand_dims(K.expand_dims(a[0], -1), -1) + a[1] - K.mean(a[1], keepdims=True)
            #                               )(conv_lambda_in)
            # elif self.dueling_type == 'max':
            #     conv_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
            #                          output_shape=(nb_action,))(y)
            # elif self.dueling_type == 'naive':
            #     conv_outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            # else:
            #     assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(inputs=model.input, outputs=[lin_outputlayer, model.outputs[1]])

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        # State.
        self.reset_states()

    def get_config(self):
        config = super(SC2DQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true_a, y_true_b, y_pred_a, y_pred_b, mask_a, mask_b = args
            loss = [huber_loss(y_true_a, y_pred_a, self.delta_clip),
                    huber_loss(y_true_b, y_pred_b, self.delta_clip)]
            loss[0] *= mask_a  # apply element-wise mask
            loss[1] *= mask_b  # apply element-wise mask
            sum_loss_a = K.sum(loss[0])
            sum_loss_b = K.sum(loss[1])
            return K.sum([sum_loss_a, sum_loss_b], axis=-1)

        def clipped_masked_error_v2(args):
            y_true_a, y_true_b, y_pred_a, y_pred_b, mask_a, mask_b = args
            loss = [huber_loss(y_true_a, y_pred_a, self.delta_clip),
                    huber_loss(y_true_b, y_pred_b, self.delta_clip)]
            loss[0] *= mask_a  # apply element-wise mask
            loss[1] *= mask_b  # apply element-wise mask
            sum_loss_a = K.sum(loss[0])
            sum_loss_a = sum_loss_a * 0
            sum_loss_b = K.sum(loss[1])
            return K.sum([sum_loss_a, sum_loss_b], axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.

        y_pred = self.model.output

        y_true_a = Input(name='y_true_a', shape=(self.nb_actions,))
        y_true_b = Input(name='y_true_b', shape=(self.screen_size, self.screen_size, 1))
        mask_a = Input(name='mask_a', shape=(self.nb_actions,))
        mask_b = Input(name='mask_b', shape=(self.screen_size, self.screen_size, 1))

        # Layer loss was called with an input that isn't a symbolic tensor. Received type: <class 'list'>.
        # Full input:
        # [<tf.Tensor 'y_true_a:0' shape=(?, 2, 1) dtype=float32>,
        # <tf.Tensor 'y_true_b:0' shape=(?, 16, 16, 1) dtype=float32>,
        # [<tf.Tensor 'dense_2/BiasAdd:0' shape=(?, 2) dtype=float32>,
        # <tf.Tensor 'conv2d_3/Relu:0' shape=(?, 16, 16, 1) dtype=float32>],
        # <tf.Tensor 'mask:0' shape=(?, 2, 1) dtype=float32>,
        # <tf.Tensor 'mask_1:0' shape=(?, 16, 16, 1) dtype=float32>].
        # All inputs to the layer should be tensors.
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true_a, y_true_b, y_pred[0], y_pred[1], mask_a, mask_b])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input

        trainable_model = Model(inputs=ins + [y_true_a, y_true_b, mask_a, mask_b], outputs=[loss_out, y_pred[0], y_pred[1]])
        print(trainable_model.summary())
        # assert len(trainable_model.output_names) == 2 what is this??
        # combined_metrics = {trainable_model.output_names[1]: metrics} i dunno ??
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses)  # metrics=combined_metrics
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    # modded
    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q1_values = self.model.predict_on_batch(state1_batch)

                actions_a = np.argmax(q1_values[0], -1)
                actions_b = []
                for ac_b in q1_values[1]:
                    actions_b.append(np.unravel_index(ac_b.argmax(), ac_b.shape)[0:2])
                actions_b = np.array(actions_b)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q1_values = self.target_model.predict_on_batch(state1_batch)

                q_batch_a = target_q1_values[0][range(self.batch_size), actions_a]
                q_batch_b = []
                for (i, square_q) in enumerate(target_q1_values[1]):
                    q_batch_b.append(square_q[:, :, 0][actions_b[i][0], actions_b[i][1]])

                q_batch_b = np.array(q_batch_b)
            else:

                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                # target_q_values = self.target_model.predict_on_batch(state1_batch)

                target_q1_values = self.target_model.predict_on_batch(state1_batch)

                q_batch_a = np.max(target_q1_values[0], axis=-1)
                q_batch_b = np.max(target_q1_values[1], axis=(1, 2))[:, 0]

                q_batch_a = np.array(q_batch_a)
                q_batch_b = np.array(q_batch_b)

            targets_a = np.zeros((self.batch_size, self.nb_actions,))
            targets_b = np.zeros((self.batch_size, self.screen_size, self.screen_size, 1))

            dummy_targets_a = np.zeros((self.batch_size,))
            dummy_targets_b = np.zeros((self.batch_size,))

            masks_a = np.zeros((self.batch_size, self.nb_actions,))
            masks_b = np.zeros((self.batch_size, self.screen_size, self.screen_size, 1))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch_a = self.gamma * q_batch_a
            discounted_reward_batch_b = self.gamma * q_batch_b

            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch_a = discounted_reward_batch_a * terminal1_batch[:]
            discounted_reward_batch_b = discounted_reward_batch_b * terminal1_batch[:]
            # INFO: try np.einsum('ij,i->ij',A,b)
            # assert discounted_reward_batch.shape == reward_batch.shape nope
            Rs_a = reward_batch[:] + discounted_reward_batch_a
            Rs_b = reward_batch[:] + discounted_reward_batch_b
            for idx, (target_a, target_b, mask_a, mask_b, R_a, R_b, action) in \
                    enumerate(zip(targets_a, targets_b, masks_a, masks_b, Rs_a, Rs_b, action_batch)):
                target_a[action.action] = R_a  # update action with estimated accumulated reward
                if len(action.coords) != 2:
                    print(action.coords)
                target_b[action.coords] = R_b  # update action with estimated accumulated reward
                dummy_targets_a[idx] = R_a
                dummy_targets_b[idx] = R_b
                mask_a[action.action] = 1.  # enable loss for this specific action
                mask_b[action.coords] = 1.  # enable loss for this specific action
            targets_a = np.array(targets_a).astype('float32')
            targets_b = np.array(targets_b).astype('float32')
            masks_a = np.array(masks_a).astype('float32')
            masks_b = np.array(masks_b).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets_a, targets_b, masks_a, masks_b],
                                                          [np.zeros(self.batch_size), targets_a, targets_b])
            metrics = [metric for idx, metric in enumerate(metrics) if
                       idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 3
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)