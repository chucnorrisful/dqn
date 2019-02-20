# Rainbow-DQN for Keras-rl in SC2

Applying the DQN-Agent from [keras-rl](https://github.com/keras-rl/keras-rl) to [Starcraft 2 Learning Environment](https://github.com/deepmind/pysc2) 
and modding it to to use the [Rainbow-DQN](https://arxiv.org/abs/1710.02298) algorithms.

---
### Current state of the project

- [x] Naive DQN with basic keras-rl dqn agent
- [x] Fully-conv network with 2 outputs (described in [this deepmind paper](https://deepmind.com/documents/110/sc2le.pdf))
- [x] Double DQN
- [x] Dueling DQN
- [x] Prioritized experience replay
- [x] Multi-step learning
- [x] Noisy nets
- [ ] Distributional RL
- [ ] Final [rainbow agent](https://arxiv.org/pdf/1710.02298.pdf)

--- 
### Challenges and Benchmarks (Deepmind SC2 minigames)

- [x] MoveToBeacon [mean: 26, max: 32]
- [x] CollectMineralShards [mean: 89, max: 120]
- [ ] FindAndDefeatZerglings
- [ ] DefeatRoaches
- [ ] DefeatZerglingsAndBanelings
- [ ] CollectMineralsAndGas
- [ ] BuildMarines
