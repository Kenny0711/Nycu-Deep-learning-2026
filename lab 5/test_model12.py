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
    def __init__(self, num_actions, game_mode):
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
        if game_mode == 'CartPole-v1':
            self.network = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            )
        elif game_mode in ('Pong-v5', 'ALE/Pong-v5'):
            #copied from paper
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 9, 9)
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 7, 7)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        else:
            raise NotImplementedError(f"wrong game name")
        ########## END OF YOUR CODE ##########
    def forward(self, x):
        return self.network(x)
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked
        
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
    
    if args.env_name == "CartPole-v1":
        preprocessor = None
        model = DQN(num_actions, args.env_name).to(device)
    elif args.env_name in ("Pong-v5", "ALE/Pong-v5"):
        preprocessor = AtariPreprocessor()
        model = DQN(num_actions, args.env_name).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if args.env_name == "CartPole-v1":
            state = np.asarray(obs, dtype=np.float32)
        else:
            state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0
        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            if args.env_name != "CartPole-v1":
                state_tensor = state_tensor / 255.0
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if args.env_name == "CartPole-v1":
                state = np.asarray(next_obs, dtype=np.float32)
            else:
                state = preprocessor.step(next_obs)
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        print(f"Test weight file: {args.model_path}, seed = {args.seed + ep}, eval reward = {total_reward}")
        total_sum += total_reward
    avg_score = total_sum / args.episodes
    print(f"\nAverage score over {args.episodes} episodes: {avg_score:.2f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=314553044, help="Random seed for evaluation")
    parser.add_argument("--env-name", type=str, default="CartPole-v1")#Pong-v5
    args = parser.parse_args()
    evaluate(args)