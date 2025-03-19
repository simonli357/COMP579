import gym
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        self.frame_idx += 1
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
        pbar = tqdm(total=num_frames, desc="Training")
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
        return all_rewards

if __name__ == "__main__":
    env = gym.make('MsPacman-v0')
    env = PreprocessFrame(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    
    num_actions = env.action_space.n
    input_channels = 4 
    
    agent = RainbowDQNAgent(env, input_channels, num_actions,
                            num_atoms=51, v_min=-10, v_max=10,
                            learning_rate=1e-4, gamma=0.99,
                            buffer_size=100000, batch_size=32, multi_step=3,
                            update_target_every=1000, alpha=0.6,
                            beta_start=0.4, beta_frames=100000)
    
    total_frames = 50000
    rewards = agent.train(total_frames)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(agent.online_net.state_dict(), os.path.join(current_dir, "rainbow_dqn_weights.pth"))
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance of Rainbow DQN on Ms. Pac-Man")
    plt.savefig(os.path.join(current_dir, "rainbow_dqn_training.png"))
    plt.show()
