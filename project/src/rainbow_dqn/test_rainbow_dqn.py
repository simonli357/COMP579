# test_rainbow_dqn.py
import gym
import torch
from rainbow_dqn import RainbowDQNAgent, PreprocessFrame  # import your agent and wrappers
from torch import device as torch_device
import os
from gym.wrappers import RecordVideo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

env = gym.make('MsPacman-v0')
env = PreprocessFrame(env)
env = gym.wrappers.FrameStack(env, num_stack=4)
current_dir = os.path.dirname(os.path.abspath(__file__))
env = RecordVideo(env, video_folder=os.path.join(current_dir, "videos"), episode_trigger=lambda episode_id: True)

num_actions = env.action_space.n
input_channels = 4 
agent = RainbowDQNAgent(
    env, 
    input_channels, 
    num_actions,
    num_atoms=51, 
    v_min=-10, 
    v_max=10,
    learning_rate=1e-4, 
    gamma=0.99,
    buffer_size=100000, 
    batch_size=32, 
    multi_step=3,
    update_target_every=1000, 
    alpha=0.6,
    beta_start=0.4, 
    beta_frames=100000
)

agent.online_net.load_state_dict(torch.load(os.path.join(current_dir, "rainbow_dqn_weights.pth"), map_location=device))
agent.online_net.eval()

num_test_episodes = 5

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
