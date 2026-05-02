import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse
import logging

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
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

        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
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
        
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    total_sum = 0
    
    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    num_actions = env.action_space.n
    
    # 初始化 Task 3 專用模型與預處理器
    preprocessor = AtariPreprocessor()
    input_dim = 4 # 因為有 4 張 frame 疊加
    model = DQN(input_dim, num_actions).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0
        
        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # 🚨 關鍵修改：移除了原版中的 state_tensor = state_tensor / 255.0
            # 因為 Dueling DQN 裡面已經除過了，再除一次會讓特徵消失！
            
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            state = preprocessor.step(next_obs)
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"task3_eval_ep{ep}.mp4")
        # 將過程存成影片
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
                
        print(f"Test weight file: {args.model_path}, seed = {args.seed + ep}, eval reward = {total_reward}")
        total_sum += total_reward
        
    avg_score = total_sum / args.episodes
    print(f"\nAverage score over {args.episodes} episodes: {avg_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="ALE/Pong-v5")
    # 請將這裡換成你 Task 3 滿分權重檔的路徑
    parser.add_argument("--model_path", type=str, default="./results/best_model.pt") 
    parser.add_argument("--output_dir", type=str, default="./videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    evaluate(args)