import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import tqdm

# --------------------------
# Preprocessing (Normalization)
# --------------------------
def preprocess_state(env, state):
    """
    For environments like Assault-ram-v5 where observations are uint8,
    convert to float and normalize by dividing by 255.
    """
    if isinstance(state, np.ndarray) and state.dtype == np.uint8:
        return state.astype(np.float32) / 255.0
    else:
        return np.array(state, dtype=np.float32)

# --------------------------
# Neural Network for Policy (z(s))
# --------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):
        super(PolicyNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output: one logit per discrete action
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.001, b=0.001)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-0.001, b=0.001)
    
    def forward(self, x):
        return self.model(x)

# --------------------------
# Neural Network for Value (for Actor-Critic)
# --------------------------
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super(ValueNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output: scalar value estimate
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.001, b=0.001)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-0.001, b=0.001)
    
    def forward(self, x):
        return self.model(x)

# --------------------------
# Helper: Select action using Boltzmann Policy
# --------------------------
def select_action(policy_net, state, temperature, env, device):
    """
    Given a state, compute logits z(s) with gradient tracking enabled,
    then form a Boltzmann policy using temperature.
    Returns the sampled action and its log probability.
    """
    policy_net.train()  # Ensure gradients are tracked
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    logits = policy_net(state_tensor).squeeze(0)  # Now computed with grad tracking
    # Scale logits by temperature T
    scaled_logits = logits / temperature
    # Compute probabilities with softmax
    probs = torch.softmax(scaled_logits, dim=0)
    # Sample an action using the Categorical distribution
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

# --------------------------
# REINFORCE: Run one episode (Monte Carlo updates)
# --------------------------
def run_episode_reinforce(env, policy_net, optimizer, gamma, temp_config, episode_idx, num_episodes, device):
    """
    temp_config: dict with keys "type" (either "fixed" or "decreasing"),
                 and for fixed: "T_fixed", for decreasing: "T_initial", "T_final"
    """
    state = env.reset()
    state = preprocess_state(env, state)
    log_probs = []
    rewards = []
    total_reward = 0
    done = False

    while not done:
        # Determine temperature based on configuration
        if temp_config["type"] == "fixed":
            temperature = temp_config["T_fixed"]
        elif temp_config["type"] == "decreasing":
            # Linear decrease from T_initial to T_final
            temperature = temp_config["T_initial"] - (temp_config["T_initial"] - temp_config["T_final"]) * (episode_idx / num_episodes)
        else:
            raise ValueError("Unknown temperature configuration type.")
        
        action, log_prob = select_action(policy_net, state, temperature, env, device)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(env, next_state)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        state = next_state

    # Compute returns (discounted sum of rewards) from the episode
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns).to(device)
    
    # Normalize returns (optional but often helps)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Compute policy loss
    policy_loss = 0
    for log_prob, G in zip(log_probs, returns):
        policy_loss += -log_prob * G
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    return total_reward

# --------------------------
# Actor-Critic: Run one episode with one-step updates
# --------------------------
def run_episode_actor_critic(env, policy_net, value_net, optimizer_policy, optimizer_value, gamma, temp_config, episode_idx, num_episodes, device):
    state = env.reset()
    state = preprocess_state(env, state)
    total_reward = 0
    done = False

    while not done:
        # Determine temperature
        if temp_config["type"] == "fixed":
            temperature = temp_config["T_fixed"]
        elif temp_config["type"] == "decreasing":
            temperature = temp_config["T_initial"] - (temp_config["T_initial"] - temp_config["T_final"]) * (episode_idx / num_episodes)
        else:
            raise ValueError("Unknown temperature configuration type.")
        
        # Compute state value
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        value = value_net(state_tensor)
        
        # Select action using policy network
        action, log_prob = select_action(policy_net, state, temperature, env, device)
        
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(env, next_state)
        total_reward += reward
        
        # Compute TD target and TD error (δ)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        with torch.no_grad():
            next_value = value_net(next_state_tensor)
        target = reward + (gamma * next_value if not done else 0)
        delta = target - value
        
        # Actor loss: policy gradient weighted by TD error (treated as advantage)
        actor_loss = -log_prob * delta.detach()
        # Critic loss: Mean Squared Error
        critic_loss = delta.pow(2)
        
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        (actor_loss + critic_loss).backward()
        optimizer_policy.step()
        optimizer_value.step()
        
        state = next_state
        
    return total_reward

# --------------------------
# Run one experiment (one seed) for REINFORCE
# --------------------------
def run_experiment_reinforce(env_name, temp_config, num_episodes, gamma, lr_policy, seed, num_layers=2, hidden_dim=256, device='cpu'):
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Get state and action dimensions
    if hasattr(env.observation_space, 'shape'):
        input_dim = env.observation_space.shape[0]
    else:
        raise ValueError("Unknown observation space format.")
    output_dim = env.action_space.n
    
    policy_net = PolicyNetwork(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    
    rewards_per_episode = []
    # for ep in range(num_episodes):
    for ep in tqdm.tqdm(range(num_episodes)):
        ep_reward = run_episode_reinforce(env, policy_net, optimizer, gamma, temp_config, ep, num_episodes, device)
        rewards_per_episode.append(ep_reward)
    env.close()
    return rewards_per_episode

# --------------------------
# Run one experiment (one seed) for Actor-Critic
# --------------------------
def run_experiment_actor_critic(env_name, temp_config, num_episodes, gamma, lr_policy, lr_value, seed, num_layers=2, hidden_dim=256, device='cpu'):
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Get state and action dimensions
    if hasattr(env.observation_space, 'shape'):
        input_dim = env.observation_space.shape[0]
    else:
        raise ValueError("Unknown observation space format.")
    output_dim = env.action_space.n
    
    policy_net = PolicyNetwork(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    value_net = ValueNetwork(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr_policy)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr_value)
    
    rewards_per_episode = []
    for ep in range(num_episodes):
        ep_reward = run_episode_actor_critic(env, policy_net, value_net, optimizer_policy, optimizer_value,
                                             gamma, temp_config, ep, num_episodes, device)
        rewards_per_episode.append(ep_reward)
    env.close()
    return rewards_per_episode

def run_experiment_env(env_name, temp_configs, num_episodes, gamma, lr_policy, lr_value, num_seeds, num_layers, hidden_dim, device, results):
    results[env_name] = {}
    for temp_key, temp_config in temp_configs.items():
        results[env_name][temp_key] = {"reinforce": None, "actor_critic": None}
        all_rewards_reinforce = []
        all_rewards_actor_critic = []
        print(f"Running {env_name} | Temperature config: {temp_key}")
        for seed in range(num_seeds):
            # REINFORCE experiment
            ep_rewards_reinforce = run_experiment_reinforce(env_name, temp_config, num_episodes, gamma,
                                                            lr_policy, seed, num_layers, hidden_dim, device)
            all_rewards_reinforce.append(ep_rewards_reinforce)
            # Actor-Critic experiment
            ep_rewards_actor_critic = run_experiment_actor_critic(env_name, temp_config, num_episodes, gamma,
                                                                    lr_policy, lr_value, seed, num_layers, hidden_dim, device)
            all_rewards_actor_critic.append(ep_rewards_actor_critic)
        results[env_name][temp_key]["reinforce"] = np.array(all_rewards_reinforce)
        results[env_name][temp_key]["actor_critic"] = np.array(all_rewards_actor_critic)
        
        # Plot training curves for this temperature configuration on the given environment
        episodes = np.arange(1, num_episodes+1)
        plt.figure()
        # REINFORCE: green, solid
        data_reinforce = results[env_name][temp_key]["reinforce"]
        mean_reinforce = data_reinforce.mean(axis=0)
        std_reinforce = data_reinforce.std(axis=0)
        plt.plot(episodes, mean_reinforce, color='green', linestyle='-', label="REINFORCE")
        plt.fill_between(episodes, mean_reinforce - std_reinforce, mean_reinforce + std_reinforce,
                            color='green', alpha=0.2)
        # Actor-Critic: red, dashed
        data_ac = results[env_name][temp_key]["actor_critic"]
        mean_ac = data_ac.mean(axis=0)
        std_ac = data_ac.std(axis=0)
        plt.plot(episodes, mean_ac, color='red', linestyle='--', label="Actor-Critic")
        plt.fill_between(episodes, mean_ac - std_ac, mean_ac + std_ac,
                            color='red', alpha=0.2)
        
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{env_name} | Temperature: {temp_key}")
        plt.legend()
        plt.tight_layout()
        plot_filename = f"plots_policy/{env_name}_{temp_key}.png"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.show()
        plt.savefig(plot_filename)
        plt.close()
    return results

# Experiment settings
num_seeds = 2           # For quick testing, reduce this number.
num_episodes = 2      # For debugging, you may reduce episodes.
gamma = 0.99
lr_policy = 1e-3         # Learning rate for policy networks (tweak as desired)
lr_value = 1e-3          # Learning rate for value network in Actor-Critic
num_layers = 2
hidden_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Temperature configurations: fixed and decreasing
temp_configs = {
    "fixed": {"type": "fixed", "T_fixed": 1.0},
    "decreasing": {"type": "decreasing", "T_initial": 1.0, "T_final": 0.1}
}

env_names = ["Acrobot-v1", "ALE/Assault-ram-v5"]
# Dictionary for storing results:
# results[env][temp_config]["reinforce"] and ["actor_critic"] will be arrays of shape (num_seeds, num_episodes)
results = {}

# Create directory for plots
os.makedirs("plots_policy", exist_ok=True)

for env_name in env_names:
    run_experiment_env(env_name, temp_configs, num_episodes, gamma, lr_policy, lr_value, num_seeds, num_layers, hidden_dim, device, results)
print("All policy-based experiments completed and plots saved in the 'plots_policy' directory.")
