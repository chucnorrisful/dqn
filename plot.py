import json
import os

import matplotlib.pyplot as plt
import numpy as np


# Eine Reihe an Hilfsmethoden für verschiedene Visualisierungen des Lernprozesses.
# Ganz unten in dieser Datei sind Beispiele.
# Achtung: Erst nach 100 Episoden sind genug Daten in den Logfiles, um einen ersten Plot anzuzeigen.


# Standard Plot des Lernfortschritts eines Agents aus dessen LogFile (auch wärend dieser lernt zu verwenden).
# paths ist eine list an pfaden(absolut) zu logfiles, welche normalerweise genau ein Element enthält.
# Mehrere Einträge in paths sind möglich, um den Verlauf eines unterbrochenen Testlaufs zu plotten, von welchem
# man zwei Logfiles hat.
# smoother gibt an, über wie viele Einträge jeweils der Durchschnitt gebildet werden soll für die mean_reward Kurve.
# Schreibt MAX, BEST_MEAN und Standardabweichung in der Umgebung des BEST_MEAN in die Kommandozeile.
# Wenn GPU-Logging benutzt wurde, können diese mit hw_stats=True geplottet werden.
# compare erwartet den gleichen Input wie paths, hier kann ein Pfad zu einem zweiten Logfile eingegeben werden,
# welcher dann zum Vergleich mit geplottet wird.
def multi_plot(paths: list, smoother: int = 100, zero_scale: int = 10, hw_stats=False, compare=None) -> None:
    rew = []
    loss = []
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

    plt.scatter(y=rew, x=np.arange(0, len(rew), 1), s=1, color="k", label='reward')
    plt.plot(smooth, '-', color='orange', label='mean_reward')
    plt.plot(sigmas, 'r-', label='sigma')
    plt.plot(zero_rate, '-', label='zero_rate')
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
    # Einkommentieren und Pfad ändern, um Plot direkt in ein Verzeichnis zu schreiben.
    # directory = "dqn/plots"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # plt.savefig(directory + '/my_plot.png', dpi=150)

    max = np.argmax(rew)
    max_mean = np.argmax(smooth)

    print("Max: ", rew[max])
    print("Best Mean:", smooth[max_mean])
    print("Standardabweichung: ", sigmas[max_mean], " [In der Umgebung des Best Mean]")


# Hilfsmethode für Testlauf in exec.py
def test_plot(rewards):

    sigmas = np.std(rewards)
    maxi = np.max(rewards)
    mean = np.mean(rewards)
    # median = np.median(rewards)

    print(sigmas, maxi, mean)


# paths sind hier verschiedene Testläufe, die in die gleiche Grafik geplottet werden sollen.
# im Gegensatz zu multi_plot wird außerdem die Standardabweichung mit eingezeichnet.
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


# Aus den Daten verschiedener Testläufe aus paths_a und paths_b wird jeweils ein Durchschnitt berechnet, und diese
# beiden Durchschnitte beide geplottet. Normalize kann "MoveToBeacon" oder "CollectMineralShards" sein und normalisiert,
# falls übergeben, zu der menschlichen Baseline eines StarCraft Grandmasters nach dem SC2LE Paper.
# wenn one=True, kann man paths_b weglassen.
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


# Benennungs-hilfs-methode
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


# paths_all ist eine zweidimensionale Liste, welche Listen von Pfaden zu Plots enthält. Aus jeder Sub-Liste wird der
# Durchschnitt gebildet, und anschließend alle zusammen geplottet.
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
        plt.plot(enemy[0], '-', label=i+1)  # , color="xkcd:orange"
        sm_plus = enemy[0] + enemy[1]
        sm_minus = enemy[0] - enemy[1]
        plt.fill_between(np.arange(0, len(enemy[0]), 1),
                         sm_plus,
                         sm_minus,
                         alpha=0.15)  # , color="xkcd:orange"
    plt.legend()
    # plt.show()
    directory = "dqn/plots/CollectMineralShards"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/cms_dqn.png', dpi=150)


# Beispeilhafte Verwendung:

# Lernverlauf live plotten.
# Erst nach den ersten 100 Episoden stehen genug Daten im Logfile, vorher crasht diese Methode!
# multi_plot(["/PathToDqn/dqn/weights/CollectMineralShards/my_first_run/1/dqn_log.json"],
#            zero_scale=20, smoother=100, hw_stats=False)

# Vergleichen zweier Durchschnitte über jeweils zwei Testläufe.
# avg_std_plot_2([["/PathToDqn/dqn/weights/MoveToBeacon/my_first_run/1/dqn_log.json",
#                  "/PathToDqn/dqn/weights/MoveToBeacon/my_first_run/2/dqn_log.json"],
#
#                 ["/PathToDqn/dqn/weights/MoveToBeacon/my_second_run/1/dqn_log.json",
#                  "/PathToDqn/dqn/weights/MoveToBeacon/my_second_run/2/dqn_log.json"]],
#                  smoother=100, normalize="MoveToBeacon")