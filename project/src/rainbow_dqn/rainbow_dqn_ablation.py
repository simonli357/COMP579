#%%
from tqdm.std import tqdm as _std_tqdm
import tqdm
tqdm.tqdm = _std_tqdm
globals()['tqdm'] = _std_tqdm

#%%
import gym
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")



#%%
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, height=84, width=84, grayscale=True):
        super(PreprocessFrame, self).__init__(env)
        self.height = height
        self.width = width
        self.grayscale = grayscale
        self.observation_space = gym.spaces.Box(low=0, high=1.0,
                                                shape=(self.height, self.width),
                                                dtype=np.float32)
        
    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        return obs



#%%
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)



#%%
class RainbowDQN(nn.Module):
    def __init__(self, input_channels, num_actions, num_atoms=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc_value = NoisyLinear(linear_input_size, 512)
        self.fc_advantage = NoisyLinear(linear_input_size, 512)
        self.value_stream = NoisyLinear(512, num_atoms)
        self.advantage_stream = NoisyLinear(512, num_actions * num_atoms)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        value = F.relu(self.fc_value(x))
        advantage = F.relu(self.fc_advantage(x))
        value = self.value_stream(value)  # shape: (batch, num_atoms)
        advantage = self.advantage_stream(advantage)  # shape: (batch, num_actions * num_atoms)
        advantage = advantage.view(-1, self.num_actions, self.num_atoms)
        
        q_atoms = value.unsqueeze(1) + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        q_dist = q_dist.clamp(min=1e-3)  # for numerical stability
        return q_dist
    
    def reset_noise(self):
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()




#%%
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority





#%%
class RainbowDQNAgent:
    def __init__(self, env, input_channels, num_actions,
                 num_atoms=51, v_min=-10, v_max=10,
                 learning_rate=1e-4, gamma=0.99,
                 buffer_size=100000, batch_size=32, multi_step=3,
                 update_target_every=1000, alpha=0.6,
                 beta_start=0.4, beta_frames=100000):
        
        self.env = env
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.multi_step = multi_step
        self.update_target_every = update_target_every
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 0
        self.beta = beta_start
        
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms).to(device)
        
        self.online_net = RainbowDQN(input_channels, num_actions, num_atoms, v_min, v_max).to(device)
        self.target_net = RainbowDQN(input_channels, num_actions, num_atoms, v_min, v_max).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
    
    def select_action(self, state):
        state = np.array(state)
        if state.ndim == 3 and state.shape[-1] == 4:
            state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
        elif state.ndim == 3 and state.shape[0] == 4:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        else:
            raise ValueError("Unexpected state shape: {}".format(state.shape))
        with torch.no_grad():
            self.online_net.reset_noise()
            q_dist = self.online_net(state)  # (1, num_actions, num_atoms)
            q_values = torch.sum(q_dist * self.support, dim=2)
            action = q_values.argmax(1).item()
        return action
    
    def projection_distribution(self, next_state, reward, done):
        """
        Compute the projection of the target distribution onto the fixed support.
        """
        with torch.no_grad():
            self.target_net.reset_noise()
            next_state = torch.FloatTensor(next_state).to(device)
            if next_state.ndim == 3:
                next_state = next_state.unsqueeze(0)
            next_dist = self.target_net(next_state)  # shape: (1, num_actions, num_atoms)
            q_values = torch.sum(next_dist * self.support, dim=2)
            next_action = q_values.argmax(1).item()  # Convert to scalar
            next_dist = next_dist[0, next_action] 
        
        Tz = reward + (1 - done) * (self.gamma ** self.multi_step) * self.support.cpu().numpy()
        Tz = np.clip(Tz, self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        l = np.floor(b).astype(np.int64)
        u = np.ceil(b).astype(np.int64)
        
        m = np.zeros(self.num_atoms, dtype=np.float32)
        for i in range(self.num_atoms):
            # Distribute probability mass to l and u
            if l[i] == u[i]:
                m[l[i]] += next_dist[i].item()
            else:
                m[l[i]] += next_dist[i].item() * (u[i] - b[i])
                m[u[i]] += next_dist[i].item() * (b[i] - l[i])
        return torch.FloatTensor(m).to(device)
    
    def update(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        # Anneal beta towards 1
        self.beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size, self.beta)
        
        states = torch.FloatTensor(states).to(device)
        if states.ndim == 4 and states.shape[-1] == 4:
            states = states.permute(0, 3, 1, 2)
        next_states = torch.FloatTensor(next_states).to(device)
        if next_states.ndim == 4 and next_states.shape[-1] == 4:
            next_states = next_states.permute(0, 3, 1, 2)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        self.online_net.reset_noise()
        dist = self.online_net(states)  # shape: (batch, num_actions, num_atoms)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
        dist = dist.gather(1, actions).squeeze(1)  # shape: (batch, num_atoms)
        dist = dist.clamp(min=1e-3)
        
        target_dist = []
        for i in range(self.batch_size):
            target_m = self.projection_distribution(next_states[i].cpu().numpy(),
                                          rewards[i].item(), dones[i].item())

            target_dist.append(target_m)
        target_dist = torch.stack(target_dist)  # shape: (batch, num_atoms)
        
        log_p = torch.log(dist)
        sample_losses = - (target_dist * log_p).sum(1)
        loss = (sample_losses * weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        new_priorities = sample_losses.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)
        
        if self.frame_idx % self.update_target_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
    
    def train(self, num_frames):
        state = self.env.reset()
        episode_reward = 0
        all_rewards = []
        pbar = tqdm(total=num_frames, desc="Training", initial=self.frame_idx)
        while self.frame_idx < num_frames:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            self.update()
            self.frame_idx += 1
            pbar.update(1)

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                pbar.set_postfix({'Episode Reward': episode_reward})
                print(f"Frame: {self.frame_idx}, Episode Reward: {episode_reward}")
                episode_reward = 0
        pbar.close()
        print("▶Training finished!")  # new add
        return all_rewards

#%%
def run_experiment(hparams):
    env = gym.make('MsPacman-v0')
    env = PreprocessFrame(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    
    num_actions = env.action_space.n
    input_channels = 4 
    
    agent = RainbowDQNAgent(
        env,
        input_channels,
        num_actions,
        num_atoms=hparams.get('num_atoms', 51),
        v_min=hparams.get('v_min', -10),
        v_max=hparams.get('v_max', 10),
        learning_rate=hparams['learning_rate'],
        gamma=hparams['gamma'],
        buffer_size=hparams['buffer_size'],
        batch_size=hparams['batch_size'],
        multi_step=hparams['multi_step'],
        update_target_every=hparams['update_target_every'],
        alpha=hparams['alpha'],
        beta_start=hparams['beta_start'],
        beta_frames=hparams['beta_frames']
    )

    # Train the agent and collect rewards
    rewards = agent.train(hparams['total_frames'])
    return rewards
    

    
    """current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # load saved model
    name = '0'
    weights_path = os.path.join(current_dir, "weights",  f"rainbow_dqn_weights_{name}.pth")
    if os.path.exists(weights_path):
        print("Loading saved model weights...")
        agent.online_net.load_state_dict(torch.load(weights_path))
        agent.target_net.load_state_dict(agent.online_net.state_dict())
    else:
        print("No saved model weights found. Starting training from scratch.")
    if os.path.exists(os.path.join(current_dir, "weights", f"rainbow_dqn_state_{name}.pkl")):
        print("Loading training state...")
        with open(os.path.join(current_dir, "weights", f"rainbow_dqn_state_{name}.pkl"), "rb") as f:
            state = pickle.load(f)
            agent.frame_idx = state["frame_idx"]
            agent.replay_buffer = state["replay_buffer"]
        print(f"Resuming training from frame {agent.frame_idx}.")
    else:
        print("No saved training state found. Starting training from scratch.")
    
    os.makedirs(os.path.join(current_dir, "weights"), exist_ok=True)
    rewards = agent.train(total_frames)
    
    # save model
    with open(os.path.join(current_dir, "weights", f"rainbow_dqn_state_{total_frames}.pkl"), "wb") as f:
        pickle.dump({
            "frame_idx": agent.frame_idx,
            "replay_buffer": agent.replay_buffer
        }, f)
    torch.save(agent.online_net.state_dict(), os.path.join(current_dir, "weights", f"rainbow_dqn_weights_{total_frames}.pth"))"""
#%%
# Storage for all experiment results
all_results = {}
all_results_large = {}

"""
lrs       = [1e-5, 5e-5]
multi_ss  = [3,    5]
alphas    = [0.4,  0.6]

"""

#%%
# Experiment A: lr=1e-5, multi_step=3, alpha=0.4
hparams_A = {
    'learning_rate': 1e-5,
    'gamma': 0.99,
    'buffer_size': 100000,
    'batch_size': 32,
    'multi_step': 3,
    'update_target_every': 1000,
    'alpha': 0.4,
    'beta_start': 0.4,
    'beta_frames': 100000,
    'total_frames': 30000
}

# Run and plot A
rewards_A = run_experiment(hparams_A)
all_results['A'] = (hparams_A, rewards_A)
#%%
plt.figure(figsize=(8, 4))
plt.plot(rewards_A, label='A: lr=1e-5,ms=3,a=0.4')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Experiment A')
plt.legend()
plt.show()




#%%
# Experiment B: lr=1e-5, multi_step=3, alpha=0.6
hparams_B = hparams_A.copy()
hparams_B['alpha'] = 0.6

rewards_B = run_experiment(hparams_B)
all_results['B'] = (hparams_B, rewards_B)
#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_B, label='B: lr=1e-5, ms=3, α=0.6')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment B')
plt.legend()
plt.show()



#%%
# Experiment C: lr=1e-5, multi_step=5, alpha=0.4
hparams_C = hparams_A.copy()
hparams_C['multi_step'] = 5

rewards_C = run_experiment(hparams_C)
all_results['C'] = (hparams_C, rewards_C)
#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_C, label='C: lr=1e-5, ms=5, α=0.4')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment C')
plt.legend()
plt.show()


#%%
# Experiment D: lr=1e-5, multi_step=5, alpha=0.6
hparams_D = hparams_A.copy()
hparams_D['multi_step'] = 5
hparams_D['alpha'] = 0.6

rewards_D = run_experiment(hparams_D)
all_results['D'] = (hparams_D, rewards_D)
#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_D, label='D: lr=1e-5, ms=5, α=0.6')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment D')
plt.legend()
plt.show()


#%%
# Experiment E: lr=5e-5, multi_step=3, alpha=0.4
hparams_E = hparams_A.copy()
hparams_E['learning_rate'] = 5e-5

rewards_E = run_experiment(hparams_E)
all_results['E'] = (hparams_E, rewards_E)
#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_E, label='E: lr=5e-5, ms=3, α=0.4')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment E')
plt.legend()
plt.show()


#%%
# Experiment F: lr=5e-5, multi_step=3, alpha=0.6
hparams_F = hparams_A.copy()
hparams_F['learning_rate'] = 5e-5
hparams_F['alpha'] = 0.6

rewards_F = run_experiment(hparams_F)
all_results['F'] = (hparams_F, rewards_F)
#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_F, label='F: lr=5e-5, ms=3, α=0.6')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment F')
plt.legend()
plt.show()


#%%
# Experiment G: lr=5e-5, multi_step=5, alpha=0.4
hparams_G = hparams_A.copy()
hparams_G['learning_rate'] = 5e-5
hparams_G['multi_step'] = 5

rewards_G = run_experiment(hparams_G)
all_results['G'] = (hparams_G, rewards_G)
#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_G, label='G: lr=5e-5, ms=5, α=0.4')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment G')
plt.legend()
plt.show()


#%%
# Experiment H: lr=5e-5, multi_step=5, alpha=0.6
hparams_H = hparams_A.copy()
hparams_H['learning_rate'] = 5e-5
hparams_H['multi_step'] = 5
hparams_H['alpha'] = 0.6

rewards_H = run_experiment(hparams_H)
all_results['H'] = (hparams_H, rewards_H)
#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_H, label='H: lr=5e-5, ms=5, α=0.6')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment H')
plt.legend()
plt.show()


#%%
plt.figure(figsize=(10,6))
for key, (h, rew) in all_results.items():
    plt.plot(rew, label=f"{key}: lr={h['learning_rate']}, ms={h['multi_step']}, α={h['alpha']}")
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('All Experiments A–H')
plt.legend()
plt.show()








#%%
import random

# New function: runs experiments across multiple seeds
def run_experiment_seeds(hparams, seeds):
    """
    Run the RainbowDQNAgent for each seed in `seeds` using the same hyperparameters.
    Returns a list of reward trajectories (one per seed).
    """
    all_runs = []  # will hold reward lists for each seed
    for seed in seeds:
        # 1) set global random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 2) create and seed the environment
        env = gym.make('MsPacman-v0')
        env.seed(seed)
        env = PreprocessFrame(env)
        env = gym.wrappers.FrameStack(env, num_stack=4)

        # 3) read action space and channels
        num_actions = env.action_space.n
        input_channels = 4  # fixed by FrameStack

        # 4) instantiate the agent with provided hyperparameters
        agent = RainbowDQNAgent(
            env,
            input_channels,
            num_actions,
            num_atoms=hparams.get('num_atoms', 51),
            v_min=hparams.get('v_min', -10),
            v_max=hparams.get('v_max', 10),
            learning_rate=hparams['learning_rate'],
            gamma=hparams['gamma'],
            buffer_size=hparams['buffer_size'],
            batch_size=hparams['batch_size'],
            multi_step=hparams['multi_step'],
            update_target_every=hparams['update_target_every'],
            alpha=hparams['alpha'],
            beta_start=hparams['beta_start'],
            beta_frames=hparams['beta_frames']
        )

        # 5) train and collect rewards
        rewards = agent.train(hparams['total_frames'])
        all_runs.append(rewards)

    return all_runs

# Storage for all experiment results
all_results_seeds = {}




#%%
# Example: run Experiment A across 3 seeds
hparams_A_seed = {
    'learning_rate': 1e-5,
    'gamma': 0.99,
    'buffer_size': 100000,
    'batch_size': 32,
    'multi_step': 3,
    'update_target_every': 1000,
    'alpha': 0.4,
    'beta_start': 0.4,
    'beta_frames': 100000,
    'total_frames': 30000
}
seeds = [0, 1, 2]
runs_A_seed = run_experiment_seeds(hparams_A_seed, seeds)
all_results_seeds['A_seed'] = (hparams_A_seed, runs_A_seed)





# %%
# Find the shortest episode length across all seeds
min_len = min(len(r) for r in runs_A_seed)
runs_trunc = [r[:min_len] for r in runs_A_seed]

episodes = np.arange(min_len)
mean_rewards = np.mean(runs_trunc, axis=0)
std_rewards  = np.std(runs_trunc, axis=0)

plt.figure(figsize=(8,4))

# Mean reward curve
plt.plot(episodes, mean_rewards, color='black', linewidth=2, label='Mean')

# Shaded ±1σ region
plt.fill_between(
    episodes,
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    color='gray',
    alpha=0.2,
    label='±1σ'
)

for i, r in enumerate(runs_trunc):
    plt.plot(episodes, r, linestyle='--', alpha=0.4, label=f"Seed {seeds[i]}")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Experiment A: lr=1e-5, multi_step=3, alpha=0.4 (3 Seeds)")
plt.legend()
plt.show()


#%%
# Example: run Experiment E across 3 seeds
hparams_E_seed = {
    'learning_rate': 5e-5,
    'gamma': 0.99,
    'buffer_size': 1000000,
    'batch_size': 32,
    'multi_step': 3,
    'update_target_every': 5000,
    'alpha': 0.4,
    'beta_start': 0.4,
    'beta_frames': 500000,
    'total_frames': 30000
}
seeds = [0, 1, 2]
runs_E_seed = run_experiment_seeds(hparams_E_seed, seeds)
all_results_seeds['E_seed'] = (hparams_E_seed, runs_E_seed)


# %%
min_len = min(len(r) for r in runs_E_seed)
runs_trunc = [r[:min_len] for r in runs_E_seed]

episodes = np.arange(min_len)
mean_rewards = np.mean(runs_trunc, axis=0)
std_rewards  = np.std(runs_trunc, axis=0)

plt.figure(figsize=(8,4))
plt.plot(episodes, mean_rewards, color='black', linewidth=2, label='Mean')
plt.fill_between(
    episodes,
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    color='gray',
    alpha=0.2,
    label='±1σ'
)

for i, r in enumerate(runs_trunc):
    plt.plot(episodes, r, linestyle='--', alpha=0.4, label=f"Seed {seeds[i]}")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Experiment E: lr=5e-5, multi_step=3, alpha=0.4 (3 Seeds)")
plt.legend()
plt.show()


#%%
# Example: run Experiment F across 3 seeds
hparams_F_seed = {
    'learning_rate': 5e-5,
    'gamma': 0.99,
    'buffer_size': 1000000,
    'batch_size': 32,
    'multi_step': 3,
    'update_target_every': 5000,
    'alpha': 0.6,
    'beta_start': 0.4,
    'beta_frames': 500000,
    'total_frames': 30000
}
seeds = [0, 1, 2]
runs_F_seed = run_experiment_seeds(hparams_F_seed, seeds)
all_results_seeds['F_seed'] = (hparams_F_seed, runs_F_seed)




#%%
hparams_A = {
    'learning_rate': 1e-5,
    'gamma': 0.99,
    'buffer_size': 100000,
    'batch_size': 32,
    'multi_step': 3,
    'update_target_every': 1000,
    'alpha': 0.4,
    'beta_start': 0.4,
    'beta_frames': 100000,
    'total_frames': 30000
}

#%%
#Choose Better performance to test large 
# 'buffer_size': 100000 -> 1000000
#'update_target_every': 1000 -> 5000
#'beta_frames': 100000 ->500000
# Experiment E: lr=5e-5, multi_step=3, alpha=0.4 
hparams_E_large = hparams_A.copy()
hparams_E_large['learning_rate'] = 5e-5
hparams_E_large['buffer_size'] = 1000000
hparams_E_large['update_target_every'] = 5000
hparams_E_large['beta_frames'] = 500000

rewards_E_large = run_experiment(hparams_E_large)
#%%
all_results_large['E'] = (hparams_E_large, rewards_E_large)


#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_E_large, label='E_Large: lr=5e-5, ms=3, α=0.4')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment E Large')
plt.legend()
plt.show()


#%%
#Choose Better performance to test large 
# 'buffer_size': 100000 -> 1000000
#'update_target_every': 1000 -> 5000
#'beta_frames': 100000 ->500000
# Experiment F: lr=5e-5, multi_step=3, alpha=0.6 
hparams_F_large = hparams_A.copy()
hparams_F_large['learning_rate'] = 5e-5
hparams_F_large['alpha'] = 0.6
hparams_F_large['buffer_size'] = 1000000
hparams_F_large['update_target_every'] = 5000
hparams_F_large['beta_frames'] = 500000

rewards_F_large = run_experiment(hparams_F_large)
all_results_large['F'] = (hparams_F_large, rewards_F_large)

#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_F_large, label='F_Large: lr=5e-5, ms=3, α=0.6')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment F Large')
plt.legend()
plt.show()


#%%
#Choose Better performance to test large 
# 'buffer_size': 100000 -> 1000000
#'update_target_every': 1000 -> 5000
#'beta_frames': 100000 ->500000
# Experiment G: lr=5e-5, multi_step=5, alpha=0.4 
hparams_G_large = hparams_A.copy()
hparams_G_large['learning_rate'] = 5e-5
hparams_G_large['multi_step'] = 5
hparams_G_large['buffer_size'] = 1000000
hparams_G_large['update_target_every'] = 5000
hparams_G_large['beta_frames'] = 500000

rewards_G_large = run_experiment(hparams_G_large)

#%%
all_results_large['G'] = (hparams_G_large, rewards_G_large)

#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_G_large, label='G_Large: lr=5e-5, ms=5, α=0.4')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment G Large')
plt.legend()
plt.show()

# %%
# Experiment H: lr=5e-5, multi_step=5, alpha=0.6 
hparams_H_large = hparams_A.copy()
hparams_H_large['learning_rate'] = 5e-5
hparams_H_large['multi_step'] = 5
hparams_H_large['alpha'] = 0.6
hparams_H_large['buffer_size'] = 1000000
hparams_H_large['update_target_every'] = 5000
hparams_H_large['beta_frames'] = 500000

rewards_H_large = run_experiment(hparams_H_large)
#%%
all_results_large['H'] = (hparams_H_large, rewards_H_large)

#%%
plt.figure(figsize=(8,4))
plt.plot(rewards_H_large, label='H_Large: lr=5e-5, ms=5, α=0.6')
plt.xlabel('Episode'); plt.ylabel('Total Reward')
plt.title('Experiment H Large')
plt.legend()
plt.show()

# %%
# Experiment A large: lr=1e-5, multi_step=3, alpha=0.4 
hparams_A_large = hparams_A.copy()
hparams_A_large['buffer_size'] = 1000000
hparams_A_large['update_target_every'] = 5000
hparams_A_large['beta_frames'] = 500000

rewards_A_large = run_experiment(hparams_A_large)

#%%
all_results_large['A'] = (hparams_A_large, rewards_A_large)


# %%
# Experiment B large: lr=1e-5, multi_step=3, alpha=0.6 
hparams_B_large = hparams_A.copy()
hparams_B_large['alpha'] = 0.6
hparams_B_large['buffer_size'] = 1000000
hparams_B_large['update_target_every'] = 5000
hparams_B_large['beta_frames'] = 500000

rewards_B_large = run_experiment(hparams_B_large)

#%%
all_results_large['B'] = (hparams_B_large, rewards_B_large)


# %%
# Experiment C large: lr=1e-5, multi_step=5, alpha=0.4
hparams_C_large = hparams_A.copy()
hparams_C_large['multi_step'] = 5
hparams_C_large['buffer_size'] = 1000000
hparams_C_large['update_target_every'] = 5000
hparams_C_large['beta_frames'] = 500000

rewards_C_large = run_experiment(hparams_C_large)

#%%
all_results_large['C'] = (hparams_C_large, rewards_C_large)


# %%
# Experiment D large: lr=1e-5, multi_step=5, alpha=0.6
hparams_D_large = hparams_A.copy()
hparams_D_large['multi_step'] = 5
hparams_D_large['alpha'] = 0.6
hparams_D_large['buffer_size'] = 1000000
hparams_D_large['update_target_every'] = 5000
hparams_D_large['beta_frames'] = 500000

rewards_D_large = run_experiment(hparams_D_large)

#%%
all_results_large['D'] = (hparams_D_large, rewards_D_large)




#%%
# Overlay Experiments E–H on one plot
plt.figure(figsize=(10, 6))

for key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
    h, rewards = all_results_large[key]  

    plt.plot(
        rewards,
        label=f"{key}: lr={h['learning_rate']}, ms={h['multi_step']}, α={h['alpha']}"
    )

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("All Experiments A–H (Extended)")
plt.legend()
plt.show()




#%%
# Experiment E with 100k frames: lr=5e-5, multi_step=3, alpha=0.4 
hparams_E_large_100k = hparams_A.copy()
hparams_E_large_100k['learning_rate'] = 5e-5
hparams_E_large_100k['buffer_size'] = 1000000
hparams_E_large_100k['update_target_every'] = 5000
hparams_E_large_100k['beta_frames'] = 500000
hparams_E_large_100k['total_frames'] = 100000

rewards_E_large_100k = run_experiment(hparams_E_large_100k)
#all_results_large['E'] = (hparams_E_large_100k, rewards_E_large_100k)