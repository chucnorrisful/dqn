import json, csv
import os

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

    # plt.plot(np.array(mean_q) * 1, 'g-', label='mean_q')
    plt.scatter(y=rew, x=np.arange(0, len(rew)*1920, 1920), s=1, color="k", label='reward')
    plt.plot(np.arange(0, len(rew)*1920, 1920), smooth, '-', color='orange', label='mean_reward')
    plt.plot(np.arange(0, len(rew)*1920, 1920), sigmas, 'r-', label='sigma')
    if compare:
        plt.plot(cmp_smooth, '-', color='blue', label='reward_cmp')
    if hw_stats:
        plt.plot(gpu_power, ',-', label='gpu_power')
        plt.plot(gpu_util, 'c,-', label='gpu_util')
        plt.plot(mem_util, 'm,-', label='mem_util')
        plt.plot(ram_util, ',-', label='ram_util')
        plt.plot(swap_util, ',-', label='swap_util')
        plt.plot(cpu_util, ',-', label='cpu_util')
    plt.axvline(x=12605*1920, label="3mio steps")
    plt.legend()
    # plt.show()
    directory = "dqn/plots/CollectMineralShards"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/fcV10_4.png', dpi=150)

    max = np.argmax(rew)
    max_mean = np.argmax(smooth)

    print(rew[max])
    print(smooth[max_mean])
    print(sigmas[max_mean])


def test_plot(rewards):

    sigmas = np.std(rewards)
    maxi = np.max(rewards)
    mean = np.mean(rewards)
    # median = np.median(rewards)

    print(sigmas, maxi, mean)


def std_plot(paths, smoother, std=True):
    rew = []
    loss = []

    for path in paths:
        with open(path) as f:
            data = json.load(f)
            loss.append(data["loss"])
            rew.append(data["episode_reward"])

    smooth_x = []
    zero_rate_x = []
    sigmas_x = []

    for curve in rew:
        smooth = []
        zero_rate = []
        sigmas = []

        for (i, re) in enumerate(curve):
            start = i - smoother
            if start < 0:
                start = 0

            end = i + smoother
            if end > len(curve) - 1:
                end = len(curve) - 1

            mean = np.mean(curve[start:end])
            smooth.append(mean)

            sigma = 0
            for ree in curve[start:end]:
                sigma += (ree - mean) ** 2

            sigma = (sigma/len(curve[start:end])) ** 0.5
            sigmas.append(sigma)

            # zero = (len(rew[start:end]) - np.count_nonzero(rew[start:end])) * zero_scale / len(rew[start:end])
            # zero_rate.append(zero)

        smooth_x.append(smooth)
        zero_rate_x.append(zero_rate)
        sigmas_x.append(sigmas)

    plt.figure()

    for i in range(len(smooth_x)):
        plt.plot(np.array(smooth_x[i]), '-', label=get_label(i))
        if std:
            plt.fill_between(np.arange(0, len(smooth_x[i]), 1),
                             np.array(smooth_x[i]) + np.array(sigmas_x[i]),
                             np.array(smooth_x[i]) - np.array(sigmas_x[i]),
                             alpha=0.15)
    plt.legend()
    # plt.show()
    directory = "dqn/plots/MoveToBeacon"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # print(os.path.dirname(os.path.realpath(directory + '/rainbow_catchy.png')))
    plt.savefig(directory + '/lol.png', dpi=150)


def avg_std_plot(paths_a, paths_b, smoother, normalize=None, one=False):
    rew_a = []
    rew_b = []

    for path in paths_a:
        with open(path) as f:
            data = json.load(f)
            rew_a.append(data["episode_reward"])

    for path in paths_b:
        with open(path) as f:
            data = json.load(f)
            rew_b.append(data["episode_reward"])

    smooth_x_a = []
    sigmas_x_a = []

    for curve in rew_a:
        smooth = []
        sigmas = []

        for (i, re) in enumerate(curve):
            start = i - smoother
            if start < 0:
                start = 0

            end = i + smoother
            if end > len(curve) - 1:
                end = len(curve) - 1

            mean = np.mean(curve[start:end])
            smooth.append(mean)

            sigma = 0
            for ree in curve[start:end]:
                sigma += (ree - mean) ** 2

            sigma = (sigma/len(curve[start:end])) ** 0.5
            sigmas.append(sigma)

        smooth_x_a.append(smooth)
        sigmas_x_a.append(sigmas)

    if not one:
        smooth_x_b = []
        sigmas_x_b = []

        for curve in rew_b:
            smooth = []
            sigmas = []

            for (i, re) in enumerate(curve):
                start = i - smoother
                if start < 0:
                    start = 0

                end = i + smoother
                if end > len(curve) - 1:
                    end = len(curve) - 1

                mean = np.mean(curve[start:end])
                smooth.append(mean)

                sigma = 0
                for ree in curve[start:end]:
                    sigma += (ree - mean) ** 2

                sigma = (sigma / len(curve[start:end])) ** 0.5
                sigmas.append(sigma)

            smooth_x_b.append(smooth)
            sigmas_x_b.append(sigmas)

    smooth_x_a = np.average(np.array(smooth_x_a), axis=0)
    sigmas_x_a = np.average(np.array(sigmas_x_a), axis=0)

    if not one:
        smooth_x_b = np.average(np.array(smooth_x_b), axis=0)
        sigmas_x_b = np.average(np.array(sigmas_x_b), axis=0)

    if normalize == "MoveToBeacon":
        random = (1, 6)
        human = (28, 28)
    if normalize == "CollectMineralShards":
        random = (17, 35)
        human = (177, 179)

    if normalize is not None:
        smooth_x_a = np.array(smooth_x_a) - random[0]
        if not one:
            smooth_x_b = np.array(smooth_x_b) - random[0]
        fac = 100 / (human[0] - random[0])
        smooth_x_a = smooth_x_a * fac
        sigmas_x_a = sigmas_x_a * fac
        if not one:
            smooth_x_b = smooth_x_b * fac
            sigmas_x_b = sigmas_x_b * fac

    avg = [(smooth_x_a, sigmas_x_a)]
    if not one:
        avg.append((smooth_x_b, sigmas_x_b))

    plt.figure()

    for enemy in avg:
        plt.plot(enemy[0], '-', color="xkcd:orange")  # , color="xkcd:orange"
        sm_plus = enemy[0] + enemy[1]
        sm_minus = enemy[0] - enemy[1]
        plt.fill_between(np.arange(0, len(enemy[0]), 1),
                         sm_plus,
                         sm_minus,
                         alpha=0.15, color="xkcd:orange")  # , color="xkcd:orange"
    # plt.legend()
    # plt.show()        i
    directory = "dqn/plots/MoveToBeacon"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/fake_rainbow_baseline_v10_avg.svg')


def get_label(i):
    if i == 0:
        return "DQN"
    elif i == 1:
        return "DDQN"
    elif i == 2:
        return "Dueling DQN"
    elif i == 3:
        return "PER DQN"
    elif i == 4:
        return "Noisy DQN"
    elif i == 5:
        return "MultiStep DQN"
    else:
        return "FullyConv V10"


def avg_std_plot_2(paths_all, smoother, normalize=None):
    rew_all = []

    for paths in paths_all:
        rews = []
        for path in paths:
            with open(path) as f:
                data = json.load(f)
                rews.append(data["episode_reward"])
        rew_all.append(rews)

    smooth_x_all = []
    sigmas_x_all = []

    for rew_group in rew_all:
        sigmas_x = []
        smooth_x = []
        for curve in rew_group:
            smooth = []
            sigmas = []

            for (i, re) in enumerate(curve):
                start = i - smoother
                if start < 0:
                    start = 0

                end = i + smoother
                if end > len(curve) - 1:
                    end = len(curve) - 1

                mean = np.mean(curve[start:end])
                smooth.append(mean)

                sigma = 0
                for ree in curve[start:end]:
                    sigma += (ree - mean) ** 2

                sigma = (sigma / len(curve[start:end])) ** 0.5
                sigmas.append(sigma)

            smooth_x.append(smooth)
            sigmas_x.append(sigmas)
        smooth_x_all.append(smooth_x)
        sigmas_x_all.append(sigmas_x)

    for group in range(len(sigmas_x_all)):
        sigmas_x_all[group] = np.average(np.array(sigmas_x_all[group]), axis=0)
        smooth_x_all[group] = np.average(np.array(smooth_x_all[group]), axis=0)

    if normalize == "MoveToBeacon":
        random = (1, 6)
        human = (28, 28)
    if normalize == "CollectMineralShards":
        random = (17, 35)
        human = (177, 179)

    if normalize is not None:
        smooth_x_all = np.array(smooth_x_all) - random[0]
        sigmas_x_all = np.array(sigmas_x_all)

        fac = 100 / (human[0] - random[0])
        smooth_x_all = smooth_x_all * fac
        sigmas_x_all = sigmas_x_all * fac

    avg = zip(smooth_x_all, sigmas_x_all)

    plt.figure()

    for i, enemy in enumerate(avg):
        plt.plot(enemy[0], '-', label=get_label(i))  # , color="xkcd:orange"
        sm_plus = enemy[0] + enemy[1]
        sm_minus = enemy[0] - enemy[1]
        plt.fill_between(np.arange(0, len(enemy[0]), 1),
                         sm_plus,
                         sm_minus,
                         alpha=0.15)  # , color="xkcd:orange"
    plt.legend()
    # plt.show()
    directory = "dqn/plots/MoveToBeacon"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/only.png', dpi=150)


# avg_std_plot_2([["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/1/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/3/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/4/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/5/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/logs_for_plots_simon/without_double_1_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/logs_for_plots_simon/without_double_2_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/logs_for_plots_simon/without_dueling_1_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/logs_for_plots_simon/without_dueling_2_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/logs_for_plots_simon/without_per_1_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/logs_for_plots_simon/without_per_2_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/without_noisy_v10/1/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/without_noisy_v10/2/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/without_multi_step_v10/1/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/without_multi_step_v10/2/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/1/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/2/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/3/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/4/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/5/dqn_log.json"]], smoother=100, normalize="MoveToBeacon")


multi_plot(["/home/benjamin/PycharmProjects/dqn/weights/CollectMineralShards/fullyConv_v10/01/dqn_log.json"],
           zero_scale=20, smoother=100, hw_stats=False)

# multi_plot(["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_multi_step_v10/01/dqn_log.json"],
#            zero_scale=20, smoother=100, hw_stats=False)

# Catchy Only Plot
# std_plot(["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/5/dqn_log.json",
#           "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_double_v10/1/dqn_log.json",
#           "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_dueling_v10/01/dqn_log.json",
#           "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_prio_v10/01/dqn_log.json",
#           "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_noisy_v10/01/dqn_log.json",
#           "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_multi_step_v10/01/dqn_log.json",
#           "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/2/dqn_log.json"], 200, std=False)

# avg_std_plot_2([["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/1/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/3/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/4/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/dqn_baseline_v10/5/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_double_v10/1/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_dueling_v10/01/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_prio_v10/01/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_noisy_v10/01/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/only_multi_step_v10/01/dqn_log.json"],
#
#                 ["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/1/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/2/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/3/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/4/dqn_log.json",
#                  "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/5/dqn_log.json"]], smoother=100, normalize="MoveToBeacon")

# avg_std_plot(["/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/1/dqn_log.json",
#               "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/2/dqn_log.json",
#               "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/3/dqn_log.json",
#               "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/4/dqn_log.json",
#               "/home/benjamin/PycharmProjects/dqn/weights/MoveToBeacon/fake_rainbow_baseline_v10/5/dqn_log.json"],
#
#              [],
#              100, one=True)

# compare=["/home/benjamin/PycharmProjects/dqn/weights/CollectMineralShards/fullyConv_v7/08/dqn_log_01.json",
#         "/home/benjamin/PycharmProjects/dqn/weights/CollectMineralShards/fullyConv_v7/08/dqn_log.json"])
