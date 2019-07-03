# Rainbow-DQN for Keras-rl in SC2

Applying the DQN-Agent from [keras-rl](https://github.com/keras-rl/keras-rl) to [Starcraft 2 Learning Environment](https://github.com/deepmind/pysc2) 
and modding it to to use the [Rainbow-DQN](https://arxiv.org/abs/1710.02298) algorithms.

---
### Current state of the project

#### Final Paper (german): [read here](https://github.com/chucnorrisful/dqn/blob/master/RainbowInSC2.pdf)

- [x] Naive DQN with basic keras-rl dqn agent
- [x] Fully-conv network with 2 outputs (described in [this deepmind paper](https://deepmind.com/documents/110/sc2le.pdf))
- [x] Double DQN (described [here](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389))
- [x] Dueling DQN (described [here](https://arxiv.org/abs/1511.06581))
- [x] Prioritized experience replay (described [here](https://arxiv.org/abs/1511.05952))
- [x] Multi-step learning (described [here](https://arxiv.org/pdf/1710.02298.pdf))
- [x] Noisy nets (described [here](https://arxiv.org/abs/1706.10295))
- [ ] Distributional RL - working, but not learning (described [here](https://dl.acm.org/citation.cfm?id=3305428))
- [x] Final [rainbow agent](https://arxiv.org/pdf/1710.02298.pdf) without Distributional RL

---
### Installation:

Make sure, you have Python 3.6.

Follow the instructions on the [pysc2](https://github.com/deepmind/pysc2) repository 
for installing it as well as for installing StarCraft2 and the required mini_games Maps.

Follow the instructions on the [keras-rl](https://github.com/keras-rl/keras-rl) repository for installation.

Follow the instructions on the [baselines](https://github.com/openai/baselines) repository for installation.

You will also need the following python packages installed:
- tensorflow 1.12 (newer is currently not working with CUDA support for me)
- keras 2.2.4
- numpy
- matplotlib

If you want to use a CUDA-able GPU, install tensorflow-gpu and keras-gpu as well. You need to make sure to 
have a compatible driver and CUDA-toolkit (9.0 works for me) and the cudnn library (7.1.2 works for me) installed. 
This provides a 5x to 20x SpeedUp and therefor is recommended for training.

Running it on Linux is recommended for training as well, because it is required for running the game headless 
with up to 2x speedup.

Download the project files:
```bash
git clone https://github.com/chucnorrisful/dqn.git
```

The executable is located in exec.py - just set some Hyperparameters and run it!

The plot.py file provides some visualisation, but you have to manually enter the 
path to a (created by execution) log file.


--- 
### Challenges and Benchmarks (Deepmind SC2 minigames)

- [x] MoveToBeacon [mean: 25,64, max: 34]
- [x] CollectMineralShards [mean: 89, max: 120]
- [ ] FindAndDefeatZerglings
- [ ] DefeatRoaches
- [ ] DefeatZerglingsAndBanelings
- [ ] CollectMineralsAndGas
- [ ] BuildMarines
