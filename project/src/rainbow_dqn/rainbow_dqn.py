import gym
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import pathlib, json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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

def _conv_out(size, k, s):       
    return (size - k) // s + 1
class RainbowBody(nn.Module):
    """
    Classic three‑layer Atari CNN used by DQN/Rainbow.
    Output shape: (batch, 64, 7, 7)  => 3136 features.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),    nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),    nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

def _fc(in_f, out_f, noisy):
    return NoisyLinear(in_f, out_f) if noisy else nn.Linear(in_f, out_f)

def build_rainbow(input_ch: int,
                  n_actions: int,
                  *,          # force keyword args
                  noisy: bool = True,
                  dueling: bool = True,
                  num_atoms: int = 51) -> nn.Module:
    """
    Creates a Rainbow‑style network with components toggled by flags.

    Args
    ----
    noisy      : if False → plain Linear layers
    dueling    : if False → single Q‑head (no value/advantage split)
    num_atoms  : support size for distributional RL (keep 51 for standard)
    """
    class RainbowNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = RainbowBody(input_ch)
            self.noisy, self.dueling, self.num_atoms = noisy, dueling, num_atoms
            self.flattened = 3136

            if dueling:
                self.fc_val = _fc(self.flattened, 512, noisy)
                self.fc_adv = _fc(self.flattened, 512, noisy)
                self.val_out = _fc(512, num_atoms, noisy)
                self.adv_out = _fc(512, n_actions * num_atoms, noisy)
            else:
                self.fc = _fc(self.flattened, 512, noisy)
                self.head = _fc(512, n_actions * num_atoms, noisy)

        # ------------------------------------------------------------------
        def forward(self, x):
            x = self.body(x)
            x = x.view(x.size(0), -1)

            if self.dueling:
                v = F.relu(self.fc_val(x))
                a = F.relu(self.fc_adv(x))
                v = self.val_out(v)                      # (B,  num_atoms)
                a = self.adv_out(a).view(-1, n_actions, self.num_atoms)
                q_atoms = v.unsqueeze(1) + a - a.mean(1, keepdim=True)
            else:
                x = F.relu(self.fc(x))
                q_atoms = self.head(x).view(-1, n_actions, self.num_atoms)

            return F.softmax(q_atoms, dim=2).clamp(min=1e-3)

        def reset_noise(self):
            if self.noisy:                # only when NoisyLinear in use
                for m in self.modules():
                    if isinstance(m, NoisyLinear):
                        m.reset_noise()

    return RainbowNet()
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
class UniformReplayBuffer(PrioritizedReplayBuffer):
    """
    Drop‑in replacement that ignores priorities and beta annealing.
    Keeps API identical to PrioritizedReplayBuffer.
    """
    def sample(self, batch_size, beta):
        indices = np.random.choice(len(self.buffer), batch_size)
        samples  = [self.buffer[i] for i in indices]
        batch = list(zip(*samples))
        states, actions, rewards, next_states, dones = map(np.array, batch)
        weights = np.ones(batch_size, dtype=np.float32)  # all 1s
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, *_):
        pass
class RainbowDQNAgent:
    def __init__(self, env, net, replay_buffer_cls,*,input_channels, num_actions,
                 num_atoms=51, v_min=-10, v_max=10,
                 learning_rate=1e-4, gamma=0.99,
                 buffer_size=100000, batch_size=32, multi_step=3,
                 update_target_every=1000, alpha=0.6,
                 beta_start=0.4, beta_frames=100000):
        
        self.env  = env
        self.online_net = net.to(device)
        self.target_net = type(net)().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.support     = torch.linspace(-10, 10, net.num_atoms).to(device)
        self.delta_z     = (10 - (-10)) / (net.num_atoms - 1)
        self.num_atoms   = net.num_atoms
        self.num_actions = env.action_space.n

        self.gamma, self.multi_step = gamma, multi_step
        self.batch_size             = batch_size
        self.update_target_every    = update_target_every

        self.replay_buffer = replay_buffer_cls(buffer_size, alpha)
        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        self.frame_idx, self.beta_start, self.beta_frames = 0, beta_start, beta_frames
        self.beta = beta_start
        
        self.v_min = v_min
        self.v_max = v_max
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
    
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
        return all_rewards

if __name__ == "__main__":
    env = gym.make('MsPacman-v0')
    env = PreprocessFrame(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    
    num_actions = env.action_space.n
    input_channels = 4 
    
    agent_params = {
        "input_channels": input_channels,
        "num_actions": num_actions,
        "num_atoms": 51,
        "v_min": -10,
        "v_max": 10,
        "learning_rate": 5e-5,
        "gamma": 0.99,
        "buffer_size": 1000000,
        "batch_size": 32,
        "multi_step": 5,
        "update_target_every": 5000,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 500000
    }
    
    agent = RainbowDQNAgent(env, build_rainbow(input_channels, num_actions), 
                        replay_buffer_cls=PrioritizedReplayBuffer,
                        **agent_params)
    
    total_frames = 50000
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # load saved model
    # name = '10000'
    # weights_path = os.path.join(current_dir, "weights",  f"rainbow_dqn_weights_{name}.pth")
    # if os.path.exists(weights_path):
    #     print("Loading saved model weights...")
    #     agent.online_net.load_state_dict(torch.load(weights_path))
    #     agent.target_net.load_state_dict(agent.online_net.state_dict())
    # else:
    #     print("No saved model weights found. Starting training from scratch.")
    # if os.path.exists(os.path.join(current_dir, "weights", f"rainbow_dqn_state_{name}.pkl")):
    #     print("Loading training state...")
    #     with open(os.path.join(current_dir, "weights", f"rainbow_dqn_state_{name}.pkl"), "rb") as f:
    #         state = pickle.load(f)
    #         agent.frame_idx = state["frame_idx"]
    #         agent.replay_buffer = state["replay_buffer"]
    #     print(f"Resuming training from frame {agent.frame_idx}.")
    # else:
    #     print("No saved training state found. Starting training from scratch.")
    
    os.makedirs(os.path.join(current_dir, "weights"), exist_ok=True)
    
    params_path = os.path.join(current_dir, "weights", f"rainbow_dqn_params_{total_frames}.json")
    with open(params_path, "w") as f:
        json.dump(agent_params, f, indent=4)
        
    rewards = agent.train(total_frames)
    
    # save model
    with open(os.path.join(current_dir, "weights", f"rainbow_dqn_state_{total_frames}.pkl"), "wb") as f:
        pickle.dump({
            "frame_idx": agent.frame_idx,
            "replay_buffer": agent.replay_buffer
        }, f)
    torch.save(agent.online_net.state_dict(), os.path.join(current_dir, "weights", f"rainbow_dqn_weights_{total_frames}.pth"))
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance of Rainbow DQN on Ms. Pac-Man")
    plt.savefig(os.path.join(current_dir, "rainbow_dqn_training.png"))
    plt.show()
