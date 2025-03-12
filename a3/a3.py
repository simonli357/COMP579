import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# --------------------------
# Q-Network definition
# --------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):
        """
        Constructs an MLP to approximate Q(x) with num_layers (2 or 3 layers recommended).
        The final output dimension equals the number of discrete actions.
        """
        super(QNetwork, self).__init__()
        layers = []
        # first layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # add extra hidden layers if needed
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.init_weights()
        
    def init_weights(self):
        # Initialize all linear layers uniformly between -0.001 and 0.001
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.001, b=0.001)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-0.001, b=0.001)
                    
    def forward(self, x):
        return self.model(x)

# --------------------------
# Replay Buffer
# --------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# --------------------------
# Preprocess State (Normalization)
# --------------------------
def preprocess_state(env, state):
    """
    For environments such as Assault-ram-v5 where observations are uint8,
    convert to float and normalize by dividing by 255.
    """
    if isinstance(state, np.ndarray) and state.dtype == np.uint8:
        return state.astype(np.float32) / 255.0
    else:
        return np.array(state, dtype=np.float32)

# --------------------------
# Single-transition update (no replay)
# --------------------------
def update_q_network(q_network, optimizer, state, action, reward, next_state, done, gamma, epsilon, algorithm, device):
    q_network.train()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = q_network(state_tensor)
    # Ensure q_value is a 1D tensor (shape [1])
    q_value = q_values[0, action].unsqueeze(0)
    
    # Compute target value
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    with torch.no_grad():
        next_q_values = q_network(next_state_tensor)
    
    if done:
        target = reward
    else:
        if algorithm == 'q_learning':
            target = reward + gamma * next_q_values.max().item()
        elif algorithm == 'expected_sarsa':
            num_actions = next_q_values.shape[1]
            best_action = next_q_values.argmax().item()
            expected_value = 0.0
            for a in range(num_actions):
                # Under ε–greedy: best action gets (1-ε+ε/num_actions), others get ε/num_actions
                if a == best_action:
                    prob = 1 - epsilon + epsilon / num_actions
                else:
                    prob = epsilon / num_actions
                expected_value += prob * next_q_values[0, a].item()
            target = reward + gamma * expected_value
        else:
            raise ValueError("Unknown algorithm type.")
    
    target_tensor = torch.FloatTensor([target]).to(device)
    loss = nn.MSELoss()(q_value, target_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --------------------------
# Batch update (with replay)
# --------------------------
def update_q_network_batch(q_network, optimizer, states, actions, rewards, next_states, dones, gamma, epsilon, algorithm, device):
    q_network.train()
    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    next_states_tensor = torch.FloatTensor(next_states).to(device)
    dones_tensor = torch.FloatTensor(dones).to(device)
    
    q_values = q_network(states_tensor)
    q_value = q_values.gather(1, actions_tensor).squeeze(1)
    
    with torch.no_grad():
        next_q_values = q_network(next_states_tensor)
    
    if algorithm == 'q_learning':
        target = rewards_tensor + gamma * (1 - dones_tensor) * next_q_values.max(1)[0]
    elif algorithm == 'expected_sarsa':
        num_actions = next_q_values.shape[1]
        best_actions = next_q_values.argmax(dim=1)
        # Build probability distribution for each next state
        probs = torch.ones_like(next_q_values) * (epsilon / num_actions)
        for i in range(len(best_actions)):
            probs[i, best_actions[i]] = 1 - epsilon + epsilon / num_actions
        expected_q = (next_q_values * probs).sum(dim=1)
        target = rewards_tensor + gamma * (1 - dones_tensor) * expected_q
    else:
        raise ValueError("Unknown algorithm type.")
    
    loss = nn.MSELoss()(q_value, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --------------------------
# Run one episode (either online update or with replay)
# --------------------------
def train_episode(env, q_network, optimizer, gamma, epsilon, algorithm, replay_buffer=None, batch_size=64, device='cpu'):
    state = env.reset()
    state = preprocess_state(env, state)
    total_reward = 0
    done = False
    while not done:
        # ε–greedy action selection using the current Q–network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_network(state_tensor)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = q_values.argmax().item()
            
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(env, next_state)
        total_reward += reward
        
        # Update: if not using replay, perform update immediately; otherwise add to buffer and update from batch
        if replay_buffer is None:
            update_q_network(q_network, optimizer, state, action, reward, next_state, done, gamma, epsilon, algorithm, device)
        else:
            replay_buffer.add(state, action, reward, next_state, done)
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                update_q_network_batch(q_network, optimizer, states, actions, rewards, next_states, dones, gamma, epsilon, algorithm, device)
        
        state = next_state
    return total_reward

# --------------------------
# Run one experiment (one seed) for given configuration
# --------------------------
def run_experiment(env_name, algorithm, use_replay, epsilon, step_size, num_episodes=1000, seed=0, num_layers=2, hidden_dim=256, gamma=0.99, batch_size=64, device='cpu'):
    # Create environment and set seeds for reproducibility
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine input dimension from observation space
    if hasattr(env.observation_space, 'shape'):
        input_dim = env.observation_space.shape[0]
    else:
        raise ValueError("Unknown observation space format.")
    output_dim = env.action_space.n
    
    # Initialize Q-network and optimizer (using SGD with the given step size)
    q_network = QNetwork(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.SGD(q_network.parameters(), lr=step_size)
    
    # Optionally use a replay buffer (capacity 1e6)
    replay_buffer = ReplayBuffer(capacity=int(1e6)) if use_replay else None
    
    episode_rewards = []
    import tqdm
    # for ep in range(num_episodes):
    for ep in tqdm.tqdm(range(num_episodes)):
        ep_reward = train_episode(env, q_network, optimizer, gamma, epsilon, algorithm, replay_buffer, batch_size, device)
        episode_rewards.append(ep_reward)
    env.close()
    return episode_rewards

# Setup parameters and initialize results dictionary
num_seeds = 10          # For debugging: fewer seeds
num_episodes = 100      # Fewer episodes for quick testing
epsilons = [0.1, 0.2, 0.3]
step_sizes = [0.25, 0.125, 0.0625]  # 1/4, 1/8, 1/16
algorithms = ['q_learning', 'expected_sarsa']
replay_settings = [False, True]
env_names = ["Acrobot-v1", "Assault-ram-v5"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.99
batch_size = 64
num_layers = 2  # or 3 as desired

# Dictionary to hold all results:
results = {}  # structure: results[env][use_replay][epsilon][step_size][algorithm] = np.array (seeds x episodes)

# Create directory to save plots
os.makedirs("plots", exist_ok=True)

print("Setup complete.")

env_name = "Acrobot-v1"
results[env_name] = {}          # Initialize for Acrobot-v1
use_replay = False              # No replay setting for this block
results[env_name][use_replay] = {}

for eps in epsilons:
    results[env_name][use_replay][eps] = {}
    for step in step_sizes:
        results[env_name][use_replay][eps][step] = {}
        for alg in algorithms:
            print(f"Acrobot-v1 | Replay: {use_replay} | ε: {eps} | step size: {step} | Alg: {alg}")
            all_rewards = []
            for seed in range(num_seeds):
                ep_rewards = run_experiment(
                    env_name, alg, use_replay, eps, step, num_episodes=num_episodes,
                    seed=seed, num_layers=num_layers, gamma=gamma, batch_size=batch_size, device=device
                )
                all_rewards.append(ep_rewards)
            results[env_name][use_replay][eps][step][alg] = np.array(all_rewards)
            
            # Plotting for the current epsilon and step size (both algorithms together)
            plt.figure()
            episodes = np.arange(1, num_episodes + 1)
            for a, color, ls in zip(algorithms, ['green', 'red'], ['-', '--']):
                data = results[env_name][use_replay][eps][step][a]
                mean_reward = data.mean(axis=0)
                std_reward = data.std(axis=0)
                plt.plot(episodes, mean_reward, color=color, linestyle=ls, label=f"{a}")
                plt.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward,
                                 color=color, alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            replay_text = "Replay" if use_replay else "No Replay"
            plt.title(f"{env_name} | {replay_text} | ε={eps} | step size={step}")
            plt.legend()
            plt.tight_layout()
            plot_filename = f"plots/{env_name}_{'replay' if use_replay else 'noreplay'}_eps{eps}_step{step}.png"
            plt.savefig(plot_filename)
            plt.close()

print("Acrobot-v1 training & plotting (No Replay) completed.")

env_name = "Acrobot-v1"
# 'results[env_name]' is already created; now use replay = True
use_replay = True
results[env_name][use_replay] = {}

for eps in epsilons:
    results[env_name][use_replay][eps] = {}
    for step in step_sizes:
        results[env_name][use_replay][eps][step] = {}
        for alg in algorithms:
            print(f"Acrobot-v1 | Replay: {use_replay} | ε: {eps} | step size: {step} | Alg: {alg}")
            all_rewards = []
            for seed in range(num_seeds):
                ep_rewards = run_experiment(
                    env_name, alg, use_replay, eps, step, num_episodes=num_episodes,
                    seed=seed, num_layers=num_layers, gamma=gamma, batch_size=batch_size, device=device
                )
                all_rewards.append(ep_rewards)
            results[env_name][use_replay][eps][step][alg] = np.array(all_rewards)
            
            # Plotting for the current epsilon and step size (both algorithms together)
            plt.figure()
            episodes = np.arange(1, num_episodes + 1)
            for a, color, ls in zip(algorithms, ['green', 'red'], ['-', '--']):
                data = results[env_name][use_replay][eps][step][a]
                mean_reward = data.mean(axis=0)
                std_reward = data.std(axis=0)
                plt.plot(episodes, mean_reward, color=color, linestyle=ls, label=f"{a}")
                plt.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward,
                                 color=color, alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            replay_text = "Replay" if use_replay else "No Replay"
            plt.title(f"{env_name} | {replay_text} | ε={eps} | step size={step}")
            plt.legend()
            plt.tight_layout()
            plot_filename = f"plots/{env_name}_{'replay' if use_replay else 'noreplay'}_eps{eps}_step{step}.png"
            plt.savefig(plot_filename)
            plt.close()

print("Acrobot-v1 training & plotting (Replay) completed.")

env_name = "Assault-ram-v5"
results[env_name] = {}          # Initialize for Assault-ram-v5
use_replay = False              # No replay setting for this block
results[env_name][use_replay] = {}

for eps in epsilons:
    results[env_name][use_replay][eps] = {}
    for step in step_sizes:
        results[env_name][use_replay][eps][step] = {}
        for alg in algorithms:
            print(f"Assault-ram-v5 | Replay: {use_replay} | ε: {eps} | step size: {step} | Alg: {alg}")
            all_rewards = []
            for seed in range(num_seeds):
                ep_rewards = run_experiment(
                    env_name, alg, use_replay, eps, step, num_episodes=num_episodes,
                    seed=seed, num_layers=num_layers, gamma=gamma, batch_size=batch_size, device=device
                )
                all_rewards.append(ep_rewards)
            results[env_name][use_replay][eps][step][alg] = np.array(all_rewards)
            
            # Plotting for the current epsilon and step size (both algorithms together)
            plt.figure()
            episodes = np.arange(1, num_episodes + 1)
            for a, color, ls in zip(algorithms, ['green', 'red'], ['-', '--']):
                data = results[env_name][use_replay][eps][step][a]
                mean_reward = data.mean(axis=0)
                std_reward = data.std(axis=0)
                plt.plot(episodes, mean_reward, color=color, linestyle=ls, label=f"{a}")
                plt.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward,
                                 color=color, alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            replay_text = "Replay" if use_replay else "No Replay"
            plt.title(f"{env_name} | {replay_text} | ε={eps} | step size={step}")
            plt.legend()
            plt.tight_layout()
            plot_filename = f"plots/{env_name}_{'replay' if use_replay else 'noreplay'}_eps{eps}_step{step}.png"
            plt.savefig(plot_filename)
            plt.close()

print("Assault-ram-v5 training & plotting (No Replay) completed.")

env_name = "Assault-ram-v5"
# 'results[env_name]' is already initialized; now use replay = True
use_replay = True
results[env_name][use_replay] = {}

for eps in epsilons:
    results[env_name][use_replay][eps] = {}
    for step in step_sizes:
        results[env_name][use_replay][eps][step] = {}
        for alg in algorithms:
            print(f"Assault-ram-v5 | Replay: {use_replay} | ε: {eps} | step size: {step} | Alg: {alg}")
            all_rewards = []
            for seed in range(num_seeds):
                ep_rewards = run_experiment(
                    env_name, alg, use_replay, eps, step, num_episodes=num_episodes,
                    seed=seed, num_layers=num_layers, gamma=gamma, batch_size=batch_size, device=device
                )
                all_rewards.append(ep_rewards)
            results[env_name][use_replay][eps][step][alg] = np.array(all_rewards)
            
            # Plotting for the current epsilon and step size (both algorithms together)
            plt.figure()
            episodes = np.arange(1, num_episodes + 1)
            for a, color, ls in zip(algorithms, ['green', 'red'], ['-', '--']):
                data = results[env_name][use_replay][eps][step][a]
                mean_reward = data.mean(axis=0)
                std_reward = data.std(axis=0)
                plt.plot(episodes, mean_reward, color=color, linestyle=ls, label=f"{a}")
                plt.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward,
                                 color=color, alpha=0.2)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            replay_text = "Replay" if use_replay else "No Replay"
            plt.title(f"{env_name} | {replay_text} | ε={eps} | step size={step}")
            plt.legend()
            plt.tight_layout()
            plot_filename = f"plots/{env_name}_{'replay' if use_replay else 'noreplay'}_eps{eps}_step{step}.png"
            plt.savefig(plot_filename)
            plt.close()

print("Assault-ram-v5 training & plotting (Replay) completed.")

