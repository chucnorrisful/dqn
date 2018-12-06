from rl.core import Processor


class Sc2Processor(Processor):

    # observation, reward, done, info = env.step(action)
    def process_observation(self, observation):

        # small_observation = observation[0].observation["feature_screen"][5]

        # print(smallObservation, observation[0].reward, observation[0].last(), "lol")

        # small_observation = small_observation.reshape(1, small_observation.shape[0], small_observation.shape[0], 1)

        # print(smallObservation.shape)

        # print(smallObservation, observation[0].reward, observation[0].last(), "lol")

        return observation

        # assert observation.ndim == 3  # (height, width, channel)
        # img = Image.fromarray(observation)
        # img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        # processed_observation = np.array(img)
        # assert processed_observation.shape == INPUT_SHAPE
        # return processed_observation.astype('uint8')  # saves storage in experience memory
