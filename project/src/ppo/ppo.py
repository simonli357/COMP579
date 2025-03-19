import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, height=84, width=84, grayscale=True):
        super(PreprocessFrame, self).__init__(env)
        self.height = height
        self.width = width
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(low=0, high=1.0,
                                                shape=(self.height, self.width),
                                                dtype=np.float32)
        
    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        return obs
      
class ActorCritic(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Linear(linear_input_size, 512)
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)
        
    def forward(self, x):
        # x shape expected: (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.policy(x), self.value(x)

class PPOAgent:
    def __init__(self, env, input_channels, num_actions, 
                 lr=2.5e-4, gamma=0.99, lam=0.95, clip_epsilon=0.1, 
                 update_epochs=4, mini_batch_size=32, rollout_length=256):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_length = rollout_length
        
        self.num_actions = num_actions
        self.actor_critic = ActorCritic(input_channels, num_actions).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
    def select_action(self, obs):
        obs_array = np.array(obs)
        # Expected shapes:
        #   Channel-last: (84, 84, 4)
        #   Channel-first: (4, 84, 84)
        if obs_array.ndim == 3 and obs_array.shape[-1] == 4:
            obs_tensor = torch.FloatTensor(obs_array).permute(2, 0, 1).unsqueeze(0).to(device)
        elif obs_array.ndim == 3 and obs_array.shape[0] == 4:
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(device)
        else:
            raise ValueError("Unexpected observation shape: {}".format(obs_array.shape))
        
        policy_logits, value = self.actor_critic(obs_tensor)
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns
    
    def collect_rollout(self):
        obs = self.env.reset()
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(self.rollout_length):
            states.append(obs)
            action, log_prob, value = self.select_action(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            obs, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            dones.append(done)
            if done:
                obs = self.env.reset()
        return states, actions, log_probs, rewards, dones, values
    
    def ppo_update(self, states, actions, log_probs, returns, advantages):
        states = np.array(states)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert states from (num_steps, H, W, C) or (num_steps, C, H, W) to (num_steps, C, H, W)
        if states.shape[-1] == 4:  # channel-last case
            states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
        else:
            states_tensor = torch.FloatTensor(states).to(device)
        
        dataset_size = states_tensor.size(0)
        indices = np.arange(dataset_size)
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]
                
                batch_states = states_tensor[mb_idx]
                batch_actions = actions[mb_idx]
                batch_old_log_probs = old_log_probs[mb_idx]
                batch_returns = returns[mb_idx]
                batch_advantages = advantages[mb_idx]
                
                policy_logits, values = self.actor_critic(batch_states)
                dist = Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                loss = policy_loss + 0.5 * value_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def train(self, total_updates=100):
        episode_rewards = []
        all_rewards = []
        for update in range(1, total_updates + 1):
            states, actions, log_probs, rewards, dones, values = self.collect_rollout()
            if dones[-1]:
                next_value = 0
            else:
                last_obs = states[-1]
                obs_array = np.array(last_obs)
                if obs_array.ndim == 3 and obs_array.shape[-1] == 4:
                    obs_tensor = torch.FloatTensor(obs_array).permute(2, 0, 1).unsqueeze(0).to(device)
                elif obs_array.ndim == 3 and obs_array.shape[0] == 4:
                    obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(device)
                else:
                    raise ValueError("Unexpected observation shape: {}".format(obs_array.shape))
                _, next_value = self.actor_critic(obs_tensor)
                next_value = next_value.item()
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            self.ppo_update(states, actions, log_probs, returns, advantages)
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            all_rewards.append(total_reward)
            
            if update % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Update {update}: Average Reward: {avg_reward:.2f}")
        return all_rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make('MsPacman-v0')
env = PreprocessFrame(env)
env = gym.wrappers.FrameStack(env, num_stack=4)

num_actions = env.action_space.n
input_channels = 4 

agent = PPOAgent(env, input_channels, num_actions, rollout_length=256, update_epochs=4, mini_batch_size=32)

training_rewards = agent.train(total_updates=100)
current_dir = os.path.dirname(os.path.abspath(__file__))
torch.save(agent.actor_critic.state_dict(), os.path.join(current_dir, "ppo_weights.pth"))

plt.figure(figsize=(8,4))
plt.plot(training_rewards)
plt.xlabel("Update")
plt.ylabel("Total Reward per Rollout")
plt.title("Training Performance of PPO Agent on Ms. Pac-Man")
plt.show()
