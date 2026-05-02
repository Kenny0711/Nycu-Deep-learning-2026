# tune_dqn.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import time

gym.register_envs(ale_py)

# ==========================================================
# DQN 模型與預處理器 (與你滿分的 task3_11 完全一致)
# ==========================================================
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        feature_dim = 64 * 7 * 7
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, x):
        features = self.feature_extractor(x / 255.0)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        cropped = gray[34:194, :]
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        _, thresholded = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        return thresholded

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_steps=1_000_000, n_multi_step=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / beta_steps
        self.n_multi_step = n_multi_step
        self.gamma = gamma
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity  

    def sample(self, batch_size):
        self.beta = min(1.0, self.beta + self.beta_increment)
        prios = self.priorities[:len(self.buffer)]
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        adjusted_samples = []
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            cumulative_reward = reward
            n_step_next_state = next_state
            n_step_done = done
            
            if not done:
                for n in range(1, self.n_multi_step):
                    if i + n < len(self.buffer):
                        _, _, next_reward, next_next_state, next_done = self.buffer[i + n]
                        cumulative_reward += (self.gamma ** n) * next_reward
                        n_step_next_state = next_next_state
                        n_step_done = next_done
                        if next_done: break
                    else: break
            adjusted_samples.append((state, action, cumulative_reward, n_step_next_state, n_step_done))
            
        return adjusted_samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            new_prio = (error + 1e-6) ** self.alpha
            self.priorities[idx] = new_prio
            self.max_priority = max(self.max_priority, new_prio)

    def __len__(self):
        return len(self.buffer)

# ==========================================================
# 封裝成 Optuna 評估用的 Agent
# ==========================================================
class TuningAgent:
    def __init__(self, trial_params):
        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        self.test_env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        
        self.q_net = DQN(4, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(4, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # 載入 Optuna 挑選的參數
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=trial_params['lr'], eps=1.5e-4)
        self.train_per_step = trial_params['train_per_step']
        self.epsilon_decay_steps = trial_params['epsilon_decay_steps']
        self.target_update_frequency = trial_params['target_update_frequency']
        
        # 固定參數
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decrement = (1.0 - 0.01) / self.epsilon_decay_steps
        self.replay_start_size = 5000
        self.n_multi_step = 3
        
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21
        
        self.memory = PrioritizedReplayBuffer(capacity=100000, alpha=0.6, beta=0.4, 
                                              beta_steps=600_000, n_multi_step=3, gamma=self.gamma)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def evaluate(self, num_episodes=5): # Tuning 時為了省時間，先測 5 局
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, term, trunc, _ = self.test_env.step(action)
                done = term or trunc
                total_reward += reward
                state = self.preprocessor.step(next_obs)
            total_rewards.append(total_reward)
        return np.mean(total_rewards)

    def train(self):
        if len(self.memory) < self.replay_start_size: return 
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement
            self.epsilon = max(self.epsilon, self.epsilon_min)
        self.train_count += 1
       
        batch, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        states, actions, rewards, next_states, dones = zip(*batch)    

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        rewards = torch.clamp(rewards, -1, 1)
        with torch.no_grad():
            a_star = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, a_star.unsqueeze(1)).squeeze(1)
            target_q = rewards + (self.gamma ** self.n_multi_step) * next_q * (1 - dones)

        td_errors = target_q - q_values
        loss = (weights * nn.functional.smooth_l1_loss(q_values, target_q, reduction='none')).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, abs(td_errors).detach().cpu().numpy())
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def run_tuning(self, trial):
        max_env_steps = 600_000
        
        while self.env_count < max_env_steps:
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            
            while not done and self.env_count < max_env_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)

                self.memory.add((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                self.env_count += 1
            
            # 每隔 100 局 (大約 2萬步) 測一次，如果過 19 分就提早結束
            if self.env_count > self.replay_start_size and self.env_count % 50000 < 5000:
                 eval_reward = self.evaluate(num_episodes=5)
                 print(f"[Trial {trial.number}] Env Step: {self.env_count} | Eval Reward: {eval_reward:.1f} | Eps: {self.epsilon:.3f}")
                 
                 # 💥 Early Stopping 達成目標！
                 if eval_reward >= 19.0:
                     print(f"🔥 BINGO! Found winning parameters at {self.env_count} steps!")
                     trial.set_user_attr("steps_to_19", self.env_count)
                     return eval_reward

                 # 讓 Optuna 決定要不要提早砍掉爛參數 (Pruning)
                 trial.report(eval_reward, self.env_count)
                 if trial.should_prune():
                     raise optuna.exceptions.TrialPruned()

        # 如果跑到 600K 都沒達到 19，回傳最後的最高分
        final_reward = self.evaluate(num_episodes=10)
        return final_reward

# ==========================================================
# Optuna Objective Function (定義探索範圍)
# ==========================================================
def objective(trial):
    # 定義要嘗試的參數範圍
    params = {
        'lr': trial.suggest_categorical('lr', [1e-4, 6.25e-5, 5e-5]),
        'epsilon_decay_steps': trial.suggest_int('epsilon_decay_steps', 100_000, 300_000, step=50_000),
        'train_per_step': trial.suggest_categorical('train_per_step', [1, 2]),
        'target_update_frequency': trial.suggest_categorical('target_update_frequency', [1000, 2000, 4000])
    }
    
    print(f"\n🚀 Starting Trial {trial.number} with params: {params}")
    agent = TuningAgent(trial_params=params)
    reward = agent.run_tuning(trial)
    return reward

if __name__ == "__main__":
    # 建立一個 Optuna 研究，目標是最大化 Reward
    study = optuna.create_study(
        direction="maximize", 
        study_name="pong-600k-hunter",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=100000) # 給 10萬步的暖身期
    )
    
    # 執行搜尋 (這裡設定跑 20 組不同參數組合，你可以依時間調整 n_trials)
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*50)
    print("🏆 Tuning Finished!")
    print("="*50)
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Highest Reward: {trial.value}")
    if "steps_to_19" in trial.user_attrs:
        print(f"  Reached 19 at step: {trial.user_attrs['steps_to_19']}")
    print("  Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*50)
    
    # 將結果存成檔案
    with open("best_params_report.txt", "w") as f:
        f.write(f"Highest Reward: {trial.value}\n")
        f.write(f"Best Parameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")