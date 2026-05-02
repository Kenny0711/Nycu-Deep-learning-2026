# Spring 2026, 535518 Deep Learning
# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

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
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self,input_dim, num_actions):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        feature_dim = 64 * 7 * 7

        # Value stream：估計 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream：估計 A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        features = self.feature_extractor(x / 255.0)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Dueling DQN 合併公式
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        #切掉上下方分數
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
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_steps=1_000_000,n_multi_step=1, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        #0501
        self.beta_increment = (1.0 - beta) / beta_steps
        # self.buffer = deque(maxlen=capacity)
        # self.priorities = deque(maxlen=capacity)
        self.n_multi_step = n_multi_step
        self.gamma = gamma
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, transition):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity  
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        prios = self.priorities[:len(self.buffer)]
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # O(1) 光速讀取
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        if self.n_multi_step > 1:
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
                            if next_done:
                                break
                        else:
                            break
                adjusted_samples.append((state, action, cumulative_reward, n_step_next_state, n_step_done))
        else:
            adjusted_samples = samples      
            
        return adjusted_samples, indices, weights
        ########## END OF YOUR CODE (for Task 3) ########## 
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-6) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        #0429
       # input_dim = self.env.observation_space.shape[0]
        input_dim=4
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(input_dim,self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(input_dim,self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
       # self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        #
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_decrement = (args.epsilon_start - args.epsilon_min) / self.epsilon_decay_steps
        #
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21 # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        #0501
        self.n_multi_step = 3
        self.memory = PrioritizedReplayBuffer(capacity=args.memory_size, 
                                              alpha=0.6, 
                                              beta=0.4, 
                                              beta_steps=600_000,
                                              n_multi_step=3, 
                                              gamma=args.discount_factor)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)

                #max_priority = np.array(self.memory.priorities).max() if len(self.memory) > 0 else 1.0
                self.memory.add((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                milestones = [600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]
                if self.env_count in milestones:
                    # 請記得把檔名中的 StudentID 換成你的真實學號！
                    milestone_path = os.path.join(self.save_dir, f"LAB5_314553044_task3_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), milestone_path)
                    print(f"\n🌟 [Milestone Reached] Saved model at {self.env_count} steps: {milestone_path}\n")
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                
                    ########## END OF YOUR CODE ##########   
                    # print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    # wandb.log({
                    #     "Episode": ep,
                    #     "Total Reward": total_reward,
                    #     "Env Step Count": self.env_count,
                    #     "Update Count": self.train_count,
                    #     "Epsilon": self.epsilon
                    # })
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                if eval_reward >= 19.5:
                    # 檔名會包含步數和分數，例如: LAB5_StudentID_task3_step450000_score20.0.pt
                    # (請記得把 StudentID 換成你的學號)
                    perfect_model_path = os.path.join(self.save_dir, f"LAB5_314553044_task3_step{self.env_count}_score{eval_reward:.1f}.pt")
                    torch.save(self.q_net.state_dict(), perfect_model_path)
                    print(f"🔥 [CRITICAL MILESTONE] Scored {eval_reward:.2f}! Model saved as {perfect_model_path}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    # def evaluate(self, episodes=20):
    #     obs, _ = self.test_env.reset()
    #     state = self.preprocessor.reset(obs)
    #     done = False
    #     total_reward = 0

    #     while not done:
    #         state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
    #         with torch.no_grad():
    #             action = self.q_net(state_tensor).argmax().item()
    #         next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
    #         done = terminated or truncated
    #         total_reward += reward
    #         state = self.preprocessor.step(next_obs)

    #     return total_reward
    def evaluate(self, num_episodes=20):
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
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.preprocessor.step(next_obs)

            total_rewards.append(total_reward)

        return np.mean(total_rewards)

    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement
            self.epsilon = max(self.epsilon, self.epsilon_min)
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        # batch = random.sample(self.memory, self.batch_size)
        # states, actions, rewards, next_states, dones = zip(*batch)
        batch, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        states, actions, rewards, next_states, dones = zip(*batch)    
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN asnd the gradient updates 
        rewards = torch.clamp(rewards, -1, 1)
        with torch.no_grad():
            a_star = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, a_star.unsqueeze(1)).squeeze(1)
            target_q = rewards + (self.gamma ** self.n_multi_step) * next_q * (1 - dones)

        # '''
        # DQN loss function
        # '''
        # with torch.no_grad():
        #     next_q = self.target_net(next_states).max(dim=1)[0]
        #     target_q = rewards + self.gamma * next_q * (1 - dones)
        td_errors = target_q - q_values
        # Clip for stability reasons
        loss = (weights * nn.functional.smooth_l1_loss(
                q_values, target_q, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        #0501梯度裁減
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        clip_error = abs(td_errors)

        self.memory.update_priorities(indices, clip_error.detach().cpu().numpy())
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        # NOTE: Enable this part if "loss" is defined
        
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
        wandb.log({
            "Loss": loss.item(),
            "Env Step Count": self.env_count
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    #parser.add_argument("--epsilon-decay", type=float, default=0.99995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=150000, help="Epsilon 從 start 衰減到 min 所需的總步數")
    parser.add_argument("--target-update-frequency", type=int, default=2000)
    parser.add_argument("--replay-start-size", type=int, default=5000)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--train-per-step", type=int, default=1)#1
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong_tk3", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run(episodes=args.num_episodes)