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

torch.manual_seed(357)
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
class ColorPreprocessFrame(gym.ObservationWrapper):
    """
    Convert to grayscale then tint into an RGB triplet.
    Outputs a float32 array in [0,1] of shape (H, W, 3).
    """
    def __init__(self, env, height=84, width=84,
                 color_map=None,  # one of cv2.COLORMAP_*, or None
                 tint=(1.0, 0.0, 0.0)):  # RGB scale factors
        super().__init__(env)
        self.h, self.w      = height, width
        self.color_map      = color_map
        self.tint           = tint
        # now 3 channels
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.h, self.w, 3),
            dtype=np.float32
        )

    def observation(self, obs):
        # convert to 8‑bit gray
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # apply colormap or manual tint
        if self.color_map is not None:
            colored = cv2.applyColorMap(gray, self.color_map)
            obs_col = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        else:
            g = gray.astype(np.float32) / 255.0
            R = g * self.tint[0]
            G = g * self.tint[1]
            B = g * self.tint[2]
            obs_col = np.stack([R, G, B], axis=2)
        # resize + normalize
        obs_col = cv2.resize(obs_col, (self.w, self.h),
                             interpolation=cv2.INTER_AREA)
        if obs_col.dtype == np.uint8:
            obs_col = obs_col.astype(np.float32) / 255.0
        return obs_col
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
    def __init__(self,
                 env,
                 net,                 # pre‑built Rainbow / ablated network
                 replay_buffer_cls,   # PrioritizedReplayBuffer  or UniformReplayBuffer
                 *,
                 buffer_size=100_000,
                 batch_size=32,
                 learning_rate=1e-4,
                 gamma=0.99,
                 multi_step=3,
                 update_target_every=1_000,
                 alpha=0.6,
                 beta_start=0.4,
                 beta_frames=100_000,
                 v_min=-10,
                 v_max=10):

        # ---------- external handles ----------
        self.env          = env
        self.online_net   = net.to(device)
        self.target_net   = type(net)().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # ---------- RL hyper‑params ----------
        self.gamma        = gamma
        self.multi_step   = multi_step
        self.batch_size   = batch_size
        self.update_target_every = update_target_every

        # ---------- distributional support ----------
        self.v_min, self.v_max   = v_min, v_max
        self.num_atoms           = net.num_atoms
        self.support             = torch.linspace(v_min, v_max, self.num_atoms).to(device)
        self.delta_z             = (v_max - v_min) / (self.num_atoms - 1)

        # ---------- replay buffer ----------
        self.replay_buffer = replay_buffer_cls(buffer_size, alpha)
        self.beta_start, self.beta_frames = beta_start, beta_frames
        self.beta         = beta_start

        # ---------- optimiser ----------
        self.optimizer    = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # ---------- bookkeeping ----------
        self.frame_idx    = 0
        self.num_actions  = env.action_space.n
    # ---------------------------------------------------------------------
    def _to_tensor(self, arr):
        arr = np.array(arr)  # Make sure it's a proper NumPy array
        t = torch.FloatTensor(arr).to(device)

        if t.ndim == 3:
            # Example shapes:
            #   grayscale stack: (4, 84, 84)
            #   RGB image: (84, 84, 3)
            if t.shape[0] == 84 and t.shape[2] == 3:
                # Single RGB image → (H, W, 3)
                t = t.permute(2, 0, 1)
            elif t.shape[0] == 4:
                # Stack of grayscale frames → (4, 84, 84)
                pass
            else:
                raise ValueError(f"Unexpected 3D shape: {t.shape}")
            t = t.unsqueeze(0)

        elif t.ndim == 4:
            # Example shapes:
            #   RGB FrameStack: (4, 84, 84, 3)
            #   Grayscale batch: (B, 4, 84, 84)
            if t.shape[-1] == 3:
                # (stack, H, W, 3)
                S, H, W, C = t.shape
                t = t.permute(0, 3, 1, 2)      # → (S, 3, H, W)
                t = t.reshape(1, S * C, H, W)  # → (1, 12, H, W)
            elif t.shape[1] == 4 and t.shape[2] == 84:
                # already (B, 4, H, W)
                pass
            else:
                raise ValueError(f"Unexpected 4D shape: {t.shape}")

        elif t.ndim == 5:
            # batched RGB frame stacks: (B, stack, H, W, 3)
            B, S, H, W, C = t.shape
            t = t.permute(0, 1, 4, 2, 3)       # (B, S, C, H, W)
            t = t.reshape(B, S * C, H, W)      # (B, 12, H, W)

        else:
            raise ValueError(f"Unexpected input shape: {t.shape}")

        return t
    def select_action(self, state):
        state_t = self._to_tensor(state)
        with torch.no_grad():
            if hasattr(self.online_net, "reset_noise"):
                self.online_net.reset_noise()
            q_dist   = self.online_net(state_t)
            q_values = (q_dist * self.support).sum(dim=2)
        return q_values.argmax(1).item()
    # ---------------------------------------------------------------------
    def projection_distribution(self, next_state, reward, done):
        with torch.no_grad():
            if hasattr(self.target_net, "reset_noise"):
                self.target_net.reset_noise()

            next_state = torch.FloatTensor(next_state).to(device)
            if next_state.ndim == 3:          # (C,H,W)
                next_state = next_state.unsqueeze(0)

            next_dist = self.target_net(next_state)            # (1, A, atoms)
            q_values  = torch.sum(next_dist * self.support, 2) # (1, A)
            next_action = q_values.argmax(1).item()
            next_dist   = next_dist[0, next_action]            # (atoms,)

        Tz = reward + (1 - done) * (self.gamma**self.multi_step) * self.support.cpu().numpy()
        Tz = np.clip(Tz, self.v_min, self.v_max)
        b  = (Tz - self.v_min) / self.delta_z
        l, u = np.floor(b).astype(np.int64), np.ceil(b).astype(np.int64)

        m = np.zeros(self.num_atoms, dtype=np.float32)
        for i in range(self.num_atoms):
            if l[i] == u[i]:
                m[l[i]] += next_dist[i].item()
            else:
                m[l[i]] += next_dist[i].item() * (u[i] - b[i])
                m[u[i]] += next_dist[i].item() * (b[i] - l[i])
        return torch.FloatTensor(m).to(device)
    # ---------------------------------------------------------------------
    def update(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        # anneal importance‑sampling β toward 1
        self.beta = min(1.0,
                        self.beta_start + self.frame_idx *
                        (1.0 - self.beta_start) / self.beta_frames)

        (states, actions, rewards,
         next_states, dones,
         idxs, weights) = self.replay_buffer.sample(self.batch_size, self.beta)

        states      = self._to_tensor(states)
        next_states = self._to_tensor(next_states)
        actions     = torch.LongTensor(actions).to(device)
        rewards     = torch.FloatTensor(rewards).to(device)
        dones       = torch.FloatTensor(dones).to(device)
        weights     = torch.FloatTensor(weights).to(device)

        if hasattr(self.online_net, "reset_noise"):
            self.online_net.reset_noise()

        dist  = self.online_net(states)                         # (B, A, atoms)
        dist  = dist.gather(1, actions.unsqueeze(1).unsqueeze(1).expand(-1,1,self.num_atoms))
        dist  = dist.squeeze(1).clamp(min=1e-3)                 # (B, atoms)

        target_dist = []
        for i in range(self.batch_size):
            target_dist.append(self.projection_distribution(next_states[i].cpu().numpy(),
                                                            rewards[i].item(),
                                                            dones[i].item()))
        target_dist = torch.stack(target_dist)                  # (B, atoms)

        loss_per_sample = -(target_dist * torch.log(dist)).sum(1)
        loss = (loss_per_sample * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update PER priorities
        new_prios = loss_per_sample.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(idxs, new_prios)

        if self.frame_idx % self.update_target_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
    # ---------------------------------------------------------------------
    def train(self, num_frames):
        state = self.env.reset()
        ep_reward, all_rewards = 0, []
        pbar = tqdm(total=num_frames, desc="Training")

        while self.frame_idx < num_frames:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state, ep_reward = next_state, ep_reward + reward
            self.update()
            self.frame_idx += 1
            pbar.update(1)

            if done:
                state = self.env.reset()
                all_rewards.append(ep_reward)
                pbar.set_postfix({'Episode reward': ep_reward})
                ep_reward = 0

        pbar.close()
        return all_rewards

def plot_rewards(rewards, save_path=None, smooth_window=5, title="Episode Rewards"):
    plt.figure(figsize=(10, 5))
    
    # Smooth curve
    if len(rewards) >= smooth_window:
        smoothed = np.convolve(rewards, np.ones(smooth_window)/smooth_window, mode='valid')
        plt.plot(smoothed, label=f"Smoothed (window={smooth_window})")
    plt.plot(rewards, alpha=0.4, label="Raw")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved reward plot to {save_path}")
    else:
        plt.show()

    plt.close()
        
def run_experiment(cfg: dict, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    frame_preprocessors = [
        ColorPreprocessFrame,
    ]
    for preprocessor in frame_preprocessors:
        env = gym.make("MsPacman-v0", render_mode=None)
        env = preprocessor(env)
        env = gym.wrappers.FrameStack(env, 4)

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:   # (stack, H, W) for grayscale
            input_ch = obs_shape[0]
        elif len(obs_shape) == 4: # (stack, H, W, 3) for RGB
            input_ch = obs_shape[0] * obs_shape[3]
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")

        net = build_rainbow(input_ch, env.action_space.n,
                            noisy=cfg["noisy"],
                            dueling=True,
                            num_atoms=51)

        buffer_cls = PrioritizedReplayBuffer if cfg["prioritized"] else UniformReplayBuffer

        agent = RainbowDQNAgent(env, net, buffer_cls,
                                buffer_size=1000,
                                batch_size=16,
                                learning_rate=cfg["lr"],
                                multi_step=cfg["multi_step"],
                                gamma=0.99)

        rewards = agent.train(cfg["frames"])
        print("Rewards:", rewards)
        print("Length of rewards:", len(rewards))
        # Save plot of rewards
        plot_path = out_dir / "rewards_plot.png"
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        print("out_dir", out_dir)
        plot_rewards(rewards, save_path=plot_path, title=f"Rewards - {cfg['name']}")

        # Save data as npy + weights
        np.save(out_dir / "rewards.npy", np.array(rewards))
        torch.save(agent.online_net.state_dict(), out_dir / "weights_final.pth")

if __name__ == "__main__":
    sweep = [
        {"name": "color_test", "noisy":True,  "prioritized":True,  "multi_step":3, "lr":1e-4, "frames":30000},
    ]
    root = pathlib.Path("experiments")
    for cfg in sweep:
        run_experiment(cfg, root / cfg["name"])