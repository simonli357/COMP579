# test_ppo.py
import gym
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Categorical
import os

from ppo import PPOAgent, PreprocessFrame, ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('MsPacman-v0')
env = PreprocessFrame(env)
env = gym.wrappers.FrameStack(env, num_stack=4)

current_dir = os.path.dirname(os.path.abspath(__file__))
from gym.wrappers import RecordVideo
env = RecordVideo(env, video_folder=os.path.join(current_dir, "videos"), episode_trigger=lambda episode_id: True)

num_actions = env.action_space.n
input_channels = 4  # Because we're stacking 4 grayscale frames

agent = PPOAgent(env, input_channels, num_actions, rollout_length=256, update_epochs=4, mini_batch_size=32)

agent.actor_critic.load_state_dict(torch.load(os.path.join(current_dir, "ppo_weights.pth"), map_location=device))
agent.actor_critic.eval()

def test_episode(env, agent):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            action, _, _ = agent.select_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

num_test_episodes = 5
for ep in range(num_test_episodes):
    ep_reward = test_episode(env, agent)
    print(f"Test Episode {ep+1}: Total Reward = {ep_reward}")

env.close()
