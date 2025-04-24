# test_rainbow_dqn.py
import gym
import torch
import json
import os
from rainbow_dqn import RainbowDQNAgent, PreprocessFrame, build_rainbow, PrioritizedReplayBuffer
from gym.wrappers import RecordVideo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Setup environment ---
env = gym.make('MsPacman-v0')
env = PreprocessFrame(env)
env = gym.wrappers.FrameStack(env, num_stack=4)

# --- Video recording ---
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "videos")
os.makedirs(video_path, exist_ok=True)

env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda ep_id: True)
env.reset()

# --- Load agent parameters ---
param_path = os.path.join(current_dir, "weights", "rainbow_dqn_params_50000.json")
with open(param_path, "r") as f:
    agent_params = json.load(f)

# --- Create and load agent ---
input_channels = agent_params["input_channels"]
num_actions = agent_params["num_actions"]

net = build_rainbow(input_channels, num_actions,
                    noisy=True,
                    dueling=True,
                    num_atoms=agent_params.get("num_atoms", 51))

agent = RainbowDQNAgent(
    env, net,
    replay_buffer_cls=PrioritizedReplayBuffer,
    **agent_params
)

weights_path = os.path.join(current_dir, "weights", "rainbow_dqn_weights_50000.pth")
agent.online_net.load_state_dict(torch.load(weights_path, map_location=device))
agent.online_net.eval()

# --- Run test episodes ---
num_test_episodes = 30

for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

env.close()
print(f"Videos saved to {video_path}")
