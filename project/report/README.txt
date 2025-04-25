# Reinforcement Learning Project

This repository contains the implementation of Proximal Policy Optimization (PPO) and Rainbow DQN algorithms for our reinforcement learning project. It includes the main training scripts, ablation study components, and video generation code.

## File Structure

### Main Training Scripts
- **PPO with hyperparameters**: [`ppo.ipynb`]  
  Implementation of Proximal Policy Optimization algorithm with tuned hyperparameters.

- **Rainbow DQN (Regular version)**: [`rainbow_dqn_Regular copy.ipynb`]  
  Standard implementation of Rainbow DQN with baseline hyperparameters.

- **Rainbow DQN (Extended tests)**: [`Rainbow_Extended_seeds copy.ipynb`]  
  Extended version of Rainbow DQN with seed tests and additional experiments.

### Rainbow DQN Ablation Study
- **Rainbow components**: [`rainbow_dqn_components.py`]  
  Core implementation of Rainbow DQN with modular components for ablation studies.

- **Color perturbation study**: [`rainbow_dqn_color_perturbation.py`]  
  Implementation testing color perturbation robustness in Rainbow DQN.

### Visualization
- **Video generation**: [`test_rainbow_dqn.py`]  
  Script to generate videos of the trained Rainbow DQN agent in action.