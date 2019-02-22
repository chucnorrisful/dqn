import json, csv
import matplotlib.pyplot as plt
import numpy as np


def multi_plot(paths: list, smoother: int = 100, zero_scale: int = 10, hw_stats=False, compare=None) -> None:
    rew = []
    loss = []
    mae = []
    mean_q = []
    mean_eps = []
    nb_steps = []

    fan_speed = []
    mem_used = []
    gpu_util = []
    mem_util = []
    gpu_temp = []
    gpu_power = []

    cpu_util = []
    ram_util = []
    swap_util = []

    cmp_rew = []

    for path in paths:
        with open(path) as f:
            data = json.load(f)
            loss += data["loss"]
            rew += data["episode_reward"]
            # mae += data["mean_absolute_error"]
            # mean_q += data["mean_q"]
            # mean_eps += data["mean_eps"]
            nb_steps += data["nb_steps"]

        if hw_stats:
            gpu_path = path[:len(path)-5] + "_gpu.json"
            with open(gpu_path) as f2:
                data = json.load(f2)
                fan_speed += data["fan_speed"]
                gpu_util += data["gpu_util"]
                mem_util += data["mem_util"]
                gpu_temp += data["gpu_temp"]
                gpu_power += data["gpu_power"]
                cpu_util += data["cpu_util"]
                ram_util += data["ram_util"]
                swap_util += data["swap_util"]

    if compare:
        for comp in compare:
            with open(comp) as f:
                data = json.load(f)
                cmp_rew += data["episode_reward"]

    smoother = smoother

    smooth = []
    zero_rate = []
    sigmas = []
    # rew = rew[400:]
    # loss = loss[400:]
    for (i, re) in enumerate(rew):
        start = i - smoother
        if start < 0:
            start = 0

        end = i + smoother
        if end > len(rew) - 1:
            end = len(rew) - 1

        mean = np.mean(rew[start:end])
        smooth.append(mean)

        sigma = 0
        for ree in rew[start:end]:
            sigma += (ree-mean) ** 2

        sigma = (sigma/len(rew[start:end])) ** 0.5
        sigmas.append(sigma)

        zero = (len(rew[start:end]) - np.count_nonzero(rew[start:end])) * zero_scale / len(rew[start:end])
        zero_rate.append(zero)

    cmp_smooth = []
    if compare:
        cmp_rew = cmp_rew[:len(rew)]
        for (i, re) in enumerate(cmp_rew):
            start = i - smoother
            if start < 0:
                start = 0

            end = i + smoother
            if end > len(cmp_rew) - 1:
                end = len(cmp_rew) - 1

            mean = np.mean(cmp_rew[start:end])
            cmp_smooth.append(mean)

    # find step count by finding steps in nb_steps array
    step_coll = 0
    for i in range(1, len(nb_steps) - 1):
        if nb_steps[i-1] > nb_steps[i]:
            step_coll += nb_steps[i-1]

    # print(step_coll)

    plt.plot(loss, 'g-', label='loss')
    plt.plot(rew, 'kx', label='reward')
    plt.plot(smooth, '-', color='orange', label='mean_reward')
    plt.plot(sigmas, 'r-', label='sigma')
    if compare:
        plt.plot(cmp_smooth, '-', color='blue', label='reward_cmp')
    if hw_stats:
        plt.plot(gpu_power, ',-', label='gpu_power')
        plt.plot(gpu_util, 'c,-', label='gpu_util')
        plt.plot(mem_util, 'm,-', label='mem_util')
        plt.plot(ram_util, ',-', label='ram_util')
        plt.plot(swap_util, ',-', label='swap_util')
        plt.plot(cpu_util, ',-', label='cpu_util')
    plt.legend()
    plt.show()

    max = np.argmax(rew)
    max_mean = np.argmax(smooth)

    print(rew[max])
    print(smooth[max_mean])
    print(sigmas[max_mean])


# CMS "/home/benjamin/PycharmProjects/dqn/weights/CollectMineralShards/fullyConv_v7/08/dqn_log_01.json",
#     "/home/benjamin/PycharmProjects/dqn/weights/CollectMineralShards/fullyConv_v7/08/dqn_log.json"
# MTB /home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fullyConv_v7/06/dqn_log.json

multi_plot(["/home/benjamin/PycharmProjects/dqn/weights/CollectMineralShards/fullyConv_v10/01/dqn_log.json"],
           zero_scale=20, smoother=100, hw_stats=False,)
# compare=["/home/benjamin/PycharmProjects/dqn/weights/CollectMineralShards/fullyConv_v7/08/dqn_log_01.json",
#         "/home/benjamin/PycharmProjects/dqn/weights/CollectMineralShards/fullyConv_v7/08/dqn_log.json"])
