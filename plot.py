import json
import matplotlib.pyplot as plt
import numpy as np


def single_plot(path, smoother=100, zero_scale=10):
    with open(path) as f:
        data = json.load(f)

    rew = data["episode_reward"]
    loss = data["loss"]

    smoother = smoother

    smooth = []
    zero_rate = []
    for (i, re) in enumerate(rew):
        start = i - smoother
        if start < 0:
            start = 0

        end = i + smoother
        if end > len(rew) - 1:
            end = len(rew) - 1

        mean = np.mean(rew[start:end])
        smooth.append(mean)

        zero = (len(rew[start:end]) - np.count_nonzero(rew[start:end])) * zero_scale / len(rew[start:end])
        zero_rate.append(zero)

    plt.plot(rew, 'kx', label='reward')
    plt.plot(loss, 'g-', label='loss')
    plt.plot(smooth, '-', color='orange', label='mean_reward')
    plt.plot(zero_rate, 'r-', label='zero_rate')
    plt.legend()
    plt.show()


def multi_plot(paths: list, smoother: int = 100, zero_scale: int = 10) -> None:
    rew = []
    loss = []
    mae = []
    mean_q = []
    mean_eps = []
    nb_steps = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
            loss += data["loss"]
            rew += data["episode_reward"]
            # mae += data["mean_absolute_error"]
            # mean_q += data["mean_q"]
            mean_eps += data["mean_eps"]
            nb_steps += data["nb_steps"]

    smoother = smoother

    smooth = []
    zero_rate = []
    for (i, re) in enumerate(rew):
        start = i - smoother
        if start < 0:
            start = 0

        end = i + smoother
        if end > len(rew) - 1:
            end = len(rew) - 1

        mean = np.mean(rew[start:end])
        smooth.append(mean)

        zero = (len(rew[start:end]) - np.count_nonzero(rew[start:end])) * zero_scale / len(rew[start:end])
        zero_rate.append(zero)

    # find step count by finding steps in nb_steps array
    step_coll = 0
    for i in range(1, len(nb_steps) - 1):
        if nb_steps[i-1] > nb_steps[i]:
            step_coll += nb_steps[i-1]

    print(step_coll)

    plt.plot(loss, 'g-', label='loss')
    plt.plot(rew, 'kx', label='reward')
    plt.plot(smooth, '-', color='orange', label='mean_reward')
    plt.plot(zero_rate, 'r-', label='zero_rate')
    plt.legend()
    plt.show()


multi_plot(["/home/benjamin/PycharmProjects/dqn/weights/fullyConv_v4_CollectMineralShards_03/dqn_log.json"],
           zero_scale=20, smoother=150)
