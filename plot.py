import json
import matplotlib.pyplot as plt
import numpy as np

with open('/home/benjamin/PycharmProjects/dqn/finalWeights/seq_v4_good_first_log.json') as f:
    data = json.load(f)

with open('/home/benjamin/PycharmProjects/dqn/dqn_MoveToBeacon_log.json') as f2:
    data2 = json.load(f2)

rew = data["episode_reward"] + data2["episode_reward"]
loss = data["loss"] + data2["loss"]

smoother = 100

smooth = []
for (i, re) in enumerate(rew):
    start = i - smoother
    if start < 0:
        start = 0

    end = i + smoother
    if end > len(rew) - 1:
        end = len(rew) - 1

    mean = np.mean(rew[start:end])
    smooth.append(mean)

plt.plot(rew)
plt.plot(smooth)
plt.plot(loss)
plt.show()